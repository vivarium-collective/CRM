from basico import *
from cobra.io import read_sbml_model
import os


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


def generate_antimony_crm_multi(num_species, num_resources, params=None,
                                 initial_N=None, initial_R=None, resource_mode='logistic'):
    """
    Generate an Antimony model string for a CRM with arbitrary numbers of species and resources.

    Args:
        num_species (int): Number of species
        num_resources (int): Number of resources
        params (dict, optional): Contains keys: tau, m, w, c, K/kappa, r
        initial_N (list or np.array, optional): Initial biomass for each species
        initial_R (list or np.array, optional): Initial resource concentrations
        resource_mode (str): One of ['logistic', 'external', 'tilman']

    Returns:
        str: Antimony model string
    """
    # If no params provided, generate defaults
    if params is None:
        np.random.seed(42)
        params = {
            "tau": np.ones(num_species),
            "m": np.full(num_species, 0.1),
            "w": np.ones(num_resources),
            "c": np.ones((num_species, num_resources)),
            "K": np.full(num_resources, 10.0),
            "r": np.full(num_resources, 1.0)
        }

    tau = params["tau"]
    m = params["m"]
    w = params["w"]
    c = params["c"]
    r = params["r"]
    K_or_kappa = params["K"] if "K" in params else params["kappa"]

    # Initial conditions
    initial_N = initial_N if initial_N is not None else [0.1] * num_species
    initial_R = initial_R if initial_R is not None else [5.0] * num_resources

    model = "model crm_dynamic()\n\n"

    # Species declarations
    species_str = ", ".join([f"N_{i}" for i in range(num_species)] + [f"R_{j}" for j in range(num_resources)])
    model += f"    species {species_str};\n\n"

    # Parameter declarations
    for i in range(num_species):
        model += f"    tau_{i} = {tau[i]}; m_{i} = {m[i]};\n"
    for j in range(num_resources):
        model += f"    w_{j} = {w[j]}; r_{j} = {r[j]}; K_{j} = {K_or_kappa[j]};\n"
    for i in range(num_species):
        for j in range(num_resources):
            model += f"    c_{i}_{j} = {c[i][j]};\n"
    model += "\n"

    # Initial conditions
    for i in range(num_species):
        model += f"    N_{i} = {initial_N[i]};\n"
    for j in range(num_resources):
        model += f"    R_{j} = {initial_R[j]};\n"

    model += "\n"

    # Species dynamics
    for i in range(num_species):
        growth_expr = " + ".join([f"c_{i}_{j} * w_{j} * R_{j}" for j in range(num_resources)])
        model += f"    N_{i}' = (N_{i} / tau_{i}) * ({growth_expr} - m_{i});\n"

    # Resource dynamics
    for j in range(num_resources):
        if resource_mode == 'logistic':
            regen = f"(r_{j} / K_{j}) * (K_{j} - R_{j}) * R_{j}"
            cons = " + ".join([f"N_{i} * c_{i}_{j} * R_{j}" for i in range(num_species)])
        elif resource_mode == 'external':
            regen = f"r_{j} * (K_{j} - R_{j})"
            cons = " + ".join([f"N_{i} * c_{i}_{j} * R_{j}" for i in range(num_species)])
        elif resource_mode == 'tilman':
            regen = f"r_{j} * (K_{j} - R_{j})"
            cons = " + ".join([f"N_{i} * c_{i}_{j}" for i in range(num_species)])
        else:
            raise ValueError("Unsupported resource_mode")

        model += f"    R_{j}' = {regen} - ({cons});\n"

    model += "\nend\n"
    return model


import tellurium as te
import basico


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
    result_df = basico.run_time_course(duration=duration, steps=steps)

    return result_df


import matplotlib.pyplot as plt

def plot_all_species_trajectories(df, species_prefix='N_', title='Population Dynamics', ylabel='Population (cells/mL)'):
    """
    Plot time series for all species in the DataFrame that match a given prefix.

    Args:
        df (pd.DataFrame): Output of basico.run_time_course()
        species_prefix (str): Filter species columns by this prefix (default 'N_')
        title (str): Title of the plot
        ylabel (str): Label for Y-axis
    """
    plt.figure(figsize=(8, 6))

    time = df.index
    species_cols = [col for col in df.columns if col.startswith(species_prefix)]

    if not species_cols:
        raise ValueError(f"No species found with prefix '{species_prefix}' in DataFrame.")

    for col in species_cols:
        plt.plot(time, df[col], label=col)

    plt.xlabel('Time (hours)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()