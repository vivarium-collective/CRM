from cobra.io import read_sbml_model
from basico import *
from cobra.io import read_sbml_model
import matplotlib.pyplot as plt
import tellurium as te
import re
import os


def generate_crm_config_from_sbml(model_file_dict,
                                   default_tau=1.0,
                                   default_m=0.1,
                                   default_r=0.5,
                                   default_K=10.0,
                                   normalize_c=True,
                                   resource_mode='logistic'):
    """
    Convert SBML model dictionary into a Vivarium CRM process config.

    Args:
        model_file_dict (dict): {species_name: sbml_path}
        default_tau (float): default adaptation time
        default_m (float): default maintenance rate
        default_r (float): default resource replenishment rate
        default_K (float): default carrying capacity
        normalize_c (bool): normalize uptake matrix
        resource_mode (str): 'logistic', 'external', or 'tilman'

    Returns:
        dict: config compatible with the CRM process
    """

    species_names = list(model_file_dict.keys())
    resource_ids = set()

    # First pass to collect resource IDs
    for key, path in model_file_dict.items():
        model = read_sbml_model(path)
        solution = model.optimize()
        for rxn in model.exchanges:
            if solution.fluxes[rxn.id] < 0:
                resource_ids.add(rxn.id)

    resource_names = sorted(resource_ids)

    # Initialize CRM parameter maps
    tau = {sp: default_tau for sp in species_names}
    maintenance = {sp: default_m for sp in species_names}
    resource_value = {r: 1.0 for r in resource_names}
    carrying_capacity = {r: default_K for r in resource_names}
    uptake_rate = {r: default_r for r in resource_names}
    resource_uptake_rate = {sp: {r: 0.0 for r in resource_names} for sp in species_names}

    # Fill uptake matrix
    for sp in species_names:
        model = read_sbml_model(model_file_dict[sp])
        solution = model.optimize()
        for r in resource_names:
            if r in solution.fluxes and solution.fluxes[r] < 0:
                resource_uptake_rate[sp][r] = abs(solution.fluxes[r])

    # Normalize uptake if requested
    if normalize_c:
        for sp in species_names:
            values = list(resource_uptake_rate[sp].values())
            total = sum(values)
            if total > 0:
                for r in resource_names:
                    resource_uptake_rate[sp][r] /= total

    config = {
        "species_number": len(species_names),
        "resource_number": len(resource_names),
        "tau": tau,
        "maintenance": maintenance,
        "resource_value": resource_value,
        "resource_uptake_rate": resource_uptake_rate,
        "carrying_capacity": carrying_capacity,
        "uptake_rate": uptake_rate,
        "resource_mode": resource_mode
    }

    return config, species_names, resource_names

# This function create a dictionary of model names and their paths.


def xml_dict(path, ext='.xml'):
    path = path
    ext = ext

    # List all files in the directory with the specified extension
    files = [file for file in os.listdir(path) if file.endswith(ext)]

    # get all the files into a dictionary
    microbiome = {}

    for xml in files:
        splitenames = xml.split('.')  # split file into component
        filename = splitenames[0]
        microbiome[filename] = path + xml

    return microbiome


# This function create a species data as well as the resources information associated with it
def species_data(model_file):
    species_data_dict = {}

    for key, value in model_file.items():
        model = read_sbml_model(value)
        solution = model.optimize()

        # Identify uptake reactions (negative fluxes in exchange reactions)
        uptake_resources = {
            reaction.id: solution.fluxes[reaction.id]
            for reaction in model.exchanges
            if solution.fluxes[reaction.id] < 0
        }

        # Build species data entry
        species_data_dict[key] = {
            'name': key,
            'model': value,
            'uptake_resources': uptake_resources,
            'death': 'delta',
            'constraints': {
                'E_star': 1.0
            },
            'resource_params': {
                res_id: {
                    'c': 1,
                    'K': 0.5,
                    'v': 1,
                    'a': 1,
                    'adaptation': 1,
                    'kinetics': 'Monod'
                } for res_id in uptake_resources.keys()
            }
        }

    return species_data_dict

import numpy as np


def generate_crm_parameters_from_sbml(model_file_dict,
                                      default_tau=1.0,
                                      default_m=0.1,
                                      default_r=0.5,
                                      default_K=10.0,
                                      normalize_c=True):
    """
    Generate CRM-compatible parameter dictionary from a dictionary of SBML models.

    Args:
        model_file_dict (dict): {species_name: sbml_path}
        default_tau (float): default tau per species
        default_m (float): default maintenance rate per species
        default_r (float): default resource regen rate
        default_K (float): default resource carrying capacity
        normalize_c (bool): if True, normalize c rows to sum to 1

    Returns:
        dict: parameters compatible with simulate_crm
    """
    from cobra.io import read_sbml_model

    species_names = list(model_file_dict.keys())
    resource_ids = set()

    # First pass: collect all uptake resource IDs
    for key, path in model_file_dict.items():
        model = read_sbml_model(path)
        solution = model.optimize()
        for rxn in model.exchanges:
            if solution.fluxes[rxn.id] < 0:
                resource_ids.add(rxn.id)

    resource_ids = sorted(list(resource_ids))
    num_species = len(species_names)
    num_resources = len(resource_ids)

    tau = np.full(num_species, default_tau)
    m = np.full(num_species, default_m)
    w = np.full(num_resources, 1.0)  # resource quality (can also use avg flux magnitude)
    r = np.full(num_resources, default_r)
    K = np.full(num_resources, default_K)
    c = np.zeros((num_species, num_resources))

    # Second pass: fill in consumer matrix c
    for i, species in enumerate(species_names):
        model = read_sbml_model(model_file_dict[species])
        solution = model.optimize()

        for j, res in enumerate(resource_ids):
            if res in solution.fluxes and solution.fluxes[res] < 0:
                c[i, j] = abs(solution.fluxes[res])  # preference strength = uptake flux

    if normalize_c:
        row_sums = c.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid divide by zero
        c /= row_sums

    params = {
        "tau": tau,
        "m": m,
        "w": w,
        "c": c,
        "r": r,
        "K": K
    }

    return params, resource_ids, species_names


def basico_model(model_name, species_data):
    basico.new_model(name=model_name)

    for species in species_data.keys():
        sp_id = species.replace(" ", "_")

        # Add death reaction
        death_rxn_id = f"Death_of_{sp_id}"
        add_reaction(death_rxn_id, f"{sp_id} ->")

        # Add growth reactions for each resource
        for resource in species_data[species]['resource_params'].keys():
            rxn_id = f"Growth_{sp_id}_on_{resource}"
            add_reaction(rxn_id, f"{sp_id} + {resource} -> 2 {sp_id}")


def spec_to_antimony_inputs(spec):
    crm_config = spec['crm']['config']
    init_species = spec['species']
    init_resources = spec['concentrations']

    # Get names
    species_names = list(init_species.keys())
    resource_names = list(init_resources.keys())

    # Get initial conditions
    initial_N = [init_species[s] for s in species_names]
    initial_R = [init_resources[r] for r in resource_names]

    # Get parameters
    tau = [crm_config['tau'][s] for s in species_names]
    m = [crm_config['maintenance'][s] for s in species_names]
    w = [crm_config['resource_value'][r] for r in resource_names]
    r = [crm_config['uptake_rate'][r] for r in resource_names]
    K = [crm_config['carrying_capacity'][r] for r in resource_names]
    c = [[float(crm_config['resource_uptake_rate'][s][r]) for r in resource_names]
         for s in species_names]

    params = {
        'tau': tau,
        'm': m,
        'w': w,
        'r': r,
        'K': K,
        'c': c
    }

    return species_names, resource_names, params, initial_N, initial_R, crm_config['resource_mode']


def sanitize(name):
    """Sanitize a name to be Antimony-compatible."""
    return re.sub(r'\W|^(?=\d)', '_', name)


def generate_antimony_crm_multi(species_names, resource_names, params,
                                      initial_N=None, initial_R=None, resource_mode='logistic'):
    """
    Generate Antimony model using real species and resource names.

    Args:
        species_names (list): List of species names (e.g. from SBML keys)
        resource_names (list): List of resource names (e.g. exchange rxn IDs)
        params (dict): CRM parameter dictionary
        initial_N (list): Initial species biomass
        initial_R (list): Initial resource concentrations
        resource_mode (str): 'logistic', 'external', or 'tilman'

    Returns:
        str: Antimony model string
    """
    num_species = len(species_names)
    num_resources = len(resource_names)

    tau = params["tau"]
    m = params["m"]
    w = params["w"]
    c = params["c"]
    r = params["r"]
    K_or_kappa = params["K"] if "K" in params else params["kappa"]

    initial_N = initial_N if initial_N is not None else [0.1] * num_species
    initial_R = initial_R if initial_R is not None else [5.0] * num_resources

    # Sanitize names
    s_names = [sanitize(n) for n in species_names]
    r_names = [sanitize(rn) for rn in resource_names]

    model = "model crm_named()\n\n"

    # Species declaration
    species_decl = ", ".join(s_names + r_names)
    model += f"    species {species_decl};\n\n"

    # Parameters
    for i, s in enumerate(s_names):
        model += f"    tau_{s} = {tau[i]}; m_{s} = {m[i]};\n"
    for j, rname in enumerate(r_names):
        model += f"    w_{rname} = {w[j]}; r_{rname} = {r[j]}; K_{rname} = {K_or_kappa[j]};\n"
    for i, s in enumerate(s_names):
        for j, rname in enumerate(r_names):
            model += f"    c_{s}_{rname} = {c[i][j]};\n"

    model += "\n"

    # Initial conditions
    for i, s in enumerate(s_names):
        model += f"    {s} = {initial_N[i]};\n"
    for j, rname in enumerate(r_names):
        model += f"    {rname} = {initial_R[j]};\n"

    model += "\n"

    # Biomass growth
    for i, s in enumerate(s_names):
        growth_terms = " + ".join([f"c_{s}_{r} * w_{r} * {r}" for r in r_names])
        model += f"    {s}' = ({s} / tau_{s}) * ({growth_terms} - m_{s});\n"

    # Resource dynamics
    for j, r in enumerate(r_names):
        if resource_mode == 'logistic':
            regen = f"(r_{r} / K_{r}) * (K_{r} - {r}) * {r}"
            consumption = " + ".join([f"{s} * c_{s}_{r} * {r}" for s in s_names])
        elif resource_mode == 'external':
            regen = f"r_{r} * (K_{r} - {r})"
            consumption = " + ".join([f"{s} * c_{s}_{r} * {r}" for s in s_names])
        elif resource_mode == 'tilman':
            regen = f"r_{r} * (K_{r} - {r})"
            consumption = " + ".join([f"{s} * c_{s}_{r}" for s in s_names])
        else:
            raise ValueError("Unsupported resource_mode")

        model += f"    {r}' = {regen} - ({consumption});\n"

    model += "\nend\n"
    return model


def run_basico_simulation_from_antimony(ant_str, duration=200, steps=1000):
    """
    Load an Antimony string, convert to SBML, simulate using Basico.

    Args:
        ant_str (str): Antimony model string
        duration (float): Simulation duration (e.g. 200 hours)
        steps (int): Number of time steps

    Returns:
        pandas.DataFrame: Simulation results from Basico
    """
    # Convert Antimony → SBML
    r = te.loada(ant_str)
    sbml_str = r.getSBML()

    # Load into Basico
    model = basico.load_model_from_string(sbml_str)

    # Simulate using Basico
    result_df = basico.run_time_course(model=model, duration=duration, steps=steps)

    return result_df


def save_antimony_as_sbml(model_str: str, filename: str = "adaptive_strategy_model.xml", path: str = None):
    """
    Convert an Antimony model string to SBML and save it to a file.

    Parameters:
    - model_str: str, Antimony-formatted model string.
    - filename: str, name of the output SBML file (default is 'adaptive_strategy_model.xml').
    - path: str, optional directory path to save the file.
    """
    try:
        r = te.loada(model_str)
        sbml_str = r.getSBML()

        # Handle path if provided
        if path is not None:
            os.makedirs(path, exist_ok=True)
            full_path = os.path.join(path, filename)
        else:
            full_path = filename

        with open(full_path, "w") as f:
            f.write(sbml_str)

        print(f"SBML model saved as '{full_path}'")
    except Exception as e:
        print(f"Failed to convert or save model: {e}")


def plot_species_and_resources(df, species_names, resource_names, title='CRM Dynamics',
                               species_to_plot=None, resources_to_plot=None):
    """
    Plot both species and resource dynamics from Basico simulation output.

    Args:
        df (pd.DataFrame): Result from basico.run_time_course()
        species_names (list): Original species names used in Antimony (unsanitized)
        resource_names (list): Original resource names used in Antimony (unsanitized)
        title (str): Title of the overall plot
        species_to_plot (list, optional): Subset of species names to plot
        resources_to_plot (list, optional): Subset of resource names to plot
    """

    def sanitize(name):
        return re.sub(r'\W|^(?=\d)', '_', name)

    # Sanitize names
    s_names_all = [sanitize(n) for n in species_names]
    r_names_all = [sanitize(r) for r in resource_names]

    # Determine which to plot
    s_names = [sanitize(n) for n in species_to_plot] if species_to_plot else s_names_all
    r_names = [sanitize(r) for r in resources_to_plot] if resources_to_plot else r_names_all

    time = df.index

    # Plot species
    plt.figure(figsize=(10, 5))
    for name in s_names:
        if name in df.columns:
            plt.plot(time, df[name], label=name)
    plt.xlabel("Time (hours)")
    plt.ylabel("Population (cells/mL)")
    plt.title(f"{title} - Species")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

