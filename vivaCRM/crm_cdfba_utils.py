from typing import Dict, List, Optional
from cdFBA.utils import get_substrates, SHARED_ENVIRONMENT


def get_initial_counts(
    species_names: List[str],
    substrates: List[str],
    *,
    default_biomass: float = 0.5,
    default_resource: float = 20.0,
    biomass_overrides: Optional[Dict[str, float]] = None,
    resource_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build a single dict of initial counts for both resources and species.

    - species_names: list of species keys (e.g., 'E.coli', 'Yeast')
    - substrates: list of resource keys (e.g., 'EX_glc__D_e', 'EX_ac_e')
    - default_biomass: fallback biomass for any species not in biomass_overrides
    - default_resource: fallback concentration/counts for any resource not in resource_overrides
    - biomass_overrides: {species -> biomass} to override defaults
    - resource_overrides: {resource -> value} to override defaults
    """
    if not substrates:
        raise ValueError("Must provide list of exchange reaction ids (resources).")

    biomass_overrides = biomass_overrides or {}
    resource_overrides = resource_overrides or {}

    conditions: Dict[str, float] = {}

    # resources first
    for ex_id in substrates:
        conditions[ex_id] = float(resource_overrides.get(ex_id, default_resource))

    # species biomasses
    for sp in species_names:
        conditions[sp] = float(biomass_overrides.get(sp, default_biomass))

    return conditions


def initial_environment(
    *,
    volume: float = 1.0,
    initial_counts: Optional[Dict[str, float]] = None,
    species_list: Optional[List[str]] = None,
    substrates: Optional[List[str]] = None,
    default_biomass: float = 0.5,
    default_resource: float = 20.0,
    biomass_overrides: Optional[Dict[str, float]] = None,
    resource_overrides: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Construct the shared_environment store with *both* counts and concentrations.

    Provide either:
      - initial_counts (a complete dict), OR
      - species_list & substrates (+ optional overrides/defaults)
    """
    if initial_counts is None:
        if species_list is None or substrates is None:
            raise ValueError("Provide initial_counts OR (species_list AND substrates).")
        initial_counts = get_initial_counts(
            species_list,
            substrates,
            default_biomass=default_biomass,
            default_resource=default_resource,
            biomass_overrides=biomass_overrides,
            resource_overrides=resource_overrides,
        )

    if volume <= 0:
        raise ValueError("volume must be > 0")

    concentrations = {k: v / volume for k, v in initial_counts.items()}

    return {
        "volume": float(volume),
        "counts": initial_counts,
        "concentrations": concentrations,
    }

if __name__ == "__main__":
    # 1) Simple, all defaults
    env = initial_environment(
        volume=1.0,
        species_list=["E.coli", "S.flexneri"],
        substrates=["EX_glc__D_e", "EX_ac_e"]
    )
    print(env)

    # 2) Per-item overrides
    env = initial_environment(
        volume=2.0,
        species_list=["E.coli", "Yeast"],
        substrates=["EX_glc__D_e", "EX_ac_e", "EX_o2_e"],
        biomass_overrides={"Yeast": 0.2},
        resource_overrides={"EX_glc__D_e": 40.0, "EX_o2_e": 100.0},
        default_resource=0.0,  # any resource not overridden -> 0
    )
    print(env)

    # 3) Provide the full dict yourself
    env = initial_environment(
        volume=1.5,
        initial_counts={"EX_glc__D_e": 60.0, "EX_ac_e": 0.0, "E.coli": 0.1}
    )
    print(env)
