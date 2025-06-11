from typing import Tuple, List, Optional, Dict
import numpy as np

def simulate_ecosystem(
    total_species: int = 1000,
    community_species: int = 100,
    num_groups: int = 4,
    num_resources: int = 10,
    total_dirichlet: float = 100,
    specialist_mean: float = 0.5,
    specialist_std: float = 0.01,
    metabolism_mode: str = 'rand',
    influx_index: int = 0,
    non_normal: bool = True,
    plot_results: bool = True
) -> Dict:
    """
    Simulate a microbial consumer-resource model (MCRM) ecosystem.

    Args:
        total_species (int): Total number of species available globally.
        community_species (int): Number of species to sample into the ecosystem.
        num_groups (int): Number of metabolic families.
        num_resources (int): Total number of resources.
        total_dirichlet (float): Dirichlet hyperparameter for metabolic variability.
        specialist_mean (float): Mean specialization for consumers.
        specialist_std (float): Standard deviation of specialization.
        metabolism_mode (str): Mode for generating stoichiometry matrix ('rand', 'thermo', 'sparse').
        influx_index (int): Index of resource being supplied.
        non_normal (bool): Whether to relax total enzyme constraint per species.
        plot_results (bool): Whether to generate plots.

    Returns:
        dict: Simulation results including species and resource dynamics.
    """
    # Generate metabolism matrix
    D = get_metabolism(num_resources, metabolism_mode, 1/num_resources)

    # Generate consumer priors
    priors = [
        get_consumer_priors(100, x, specialist_mean, specialist_std, num_resources)
        for x in range(num_groups)
    ]

    # Construct consumers
    out, _ = make_phylo_consumers(
        total_species, num_resources, num_groups, priors,
        total_dirichlet, non_normal
    )

    # Sample a subpopulation
    k = np.random.choice(total_species, community_species, replace=False)
    C_sample = out['C'][k]
    group_labels = out['group'][k]

    # Initialize model parameters
    params = mcrm_params(influx_index, community_species, C_sample, D, 'eye', '')

    # Run simulation
    result = run_mcrm(params)

    if plot_results:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(result['species'])
        axs[0].set_title("Species Abundance Over Time")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Abundance")

        final_abundance = result['communityStruct']
        x = coarse_grain_community_structure(final_abundance, group_labels, num_groups)
        x[x < 1e-10] = 0
        axs[1].pie(x + 1e-20, labels=[f"Family {i+1}" for i in range(num_groups)], autopct='%1.1f%%')
        axs[1].set_title("Community Composition by Family")
        plt.tight_layout()
        plt.show()

    return {
        "result": result,
        "group_labels": group_labels,
        "params": params,
        "D": D,
        "consumer_matrix": C_sample,
        "k_indices": k
    }