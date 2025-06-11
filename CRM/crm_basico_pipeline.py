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

def basico_model(model_name, species_data):
    new_model(name=model_name)

    for species in species_data.keys():
        sp_id = species.replace(" ", "_")

        # Add death reaction
        death_rxn_id = f"Death_of_{sp_id}"
        add_reaction(death_rxn_id, f"{sp_id} ->")

        # Add growth reactions for each resource
        for resource in species_data[species]['resource_params'].keys():
            rxn_id = f"Growth_{sp_id}_on_{resource}"
            add_reaction(rxn_id, f"{sp_id} + {resource} -> 2 {sp_id}")


def generate_antimony(resources, Vmax, Km, v, delta, Y_map=None, initial_conditions=None):
    """
    Generate a Tellurium Antimony CRM model for ANY number of resources.

    Args:
        resources (list): list of resource names, e.g., ['glucose', 'acetate']
        Vmax (list): Vmax for each resource
        Km (list): Km for each resource
        v (list): Growth value for each resource
        delta (float): death rate
        Y_map (dict, optional): {produced_resource: (source_resource, yield_value)}
            E.g., {'acetate': ('glucose', 1.0)}
        initial_conditions (dict, optional): {'biomass': value, 'resource_name': value}
            If not provided, default initial conditions will be used.
    Returns:
        str: Antimony model string
    """
    model = "model crm_fixed()\n"

    # Define species
    model += "    species N"
    for res in resources:
        model += f", C_{res}"
    model += ";\n\n"

    # Define parameters
    for i, res in enumerate(resources):
        model += f"    Vmax_{res} = {Vmax[i]};\n"
        model += f"    Km_{res} = {Km[i]};\n"
        model += f"    v_{res} = {v[i]};\n"
    model += f"    delta = {delta};\n"
    if Y_map:
        for prod_res, (source_res, Y_val) in Y_map.items():
            model += f"    Y_{prod_res}_from_{source_res} = {Y_val};\n"
    model += "\n"

    # Initial conditions
    model += "    N = "
    model += f"{initial_conditions.get('biomass', 0.1) if initial_conditions else 0.1};\n"
    for res in resources:
        init_val = initial_conditions.get(res, 10.0) if initial_conditions else 10.0
        model += f"    C_{res} = {init_val};\n"
    model += "\n"

    # Uptake rates
    for res in resources:
        model += f"    r_{res} := Vmax_{res} * (C_{res} / (Km_{res} + C_{res}));\n"
    model += "\n"

    # Biomass growth
    growth_terms = " + ".join([f"v_{res} * r_{res}" for res in resources])
    model += f"    N' = N * ({growth_terms} - delta);\n"

    # Resource dynamics
    for res in resources:
        if Y_map and res in Y_map:
            # This resource is produced by another resource
            source_res, _ = Y_map[res]
            model += f"    C_{res}' = N * (Y_{res}_from_{source_res} * r_{source_res} - r_{res});\n"
        else:
            # Normal consumption only
            model += f"    C_{res}' = -N * r_{res};\n"

    model += "end\n"

    return model