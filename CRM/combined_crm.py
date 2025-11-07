#!/usr/bin/env python3
# combined_crm.py
# Unified simulator: MacArthur, Goldford (cross-feeding), Marsland (leakage), Pacciani–Mori (adaptive uptake)

import os
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Literal
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =========================================================
# Unified Consumer-Resource RHS
# =========================================================
def unified_consumer_resource_rhs(
    time: float,
    state_vector: NDArray[np.floating],
    *,
    num_species: int,
    num_resources: int,
    use_monod_kinetics: bool = False,
    maximum_uptake_rate_matrix: Optional[NDArray[np.floating]] = None,
    half_saturation_matrix: Optional[NDArray[np.floating]] = None,
    consumption_matrix: Optional[NDArray[np.floating]] = None,
    per_resource_growth_yield: Optional[NDArray[np.floating]] = None,
    species_maintenance_rate: NDArray[np.floating],
    renewal_mode: Literal["chemostat","logistic"] = "chemostat",
    resource_supply_rate: Optional[NDArray[np.floating]] = None,
    resource_decay_rate: Optional[NDArray[np.floating]] = None,
    resource_intrinsic_growth: Optional[NDArray[np.floating]] = None,
    resource_carrying_capacity: Optional[NDArray[np.floating]] = None,
    byproduct_matrix: Optional[NDArray[np.floating]] = None,
    leakage_fraction: float = 0.0,
    energy_accounting: Literal["none","strict","marsland"] = "none",
    use_adaptive_uptake: bool = False,
    adaptive_value_weight: Optional[NDArray[np.floating]] = None,
    adaptive_learning_rate: float = 1.0,
) -> NDArray[np.floating]:
    """Unified model: MacArthur + Crossfeeding + Leakage + Adaptive uptake"""

    S, P = num_species, num_resources
    populations = state_vector[:S]
    resources = state_vector[S:S+P]

    if use_adaptive_uptake:
        alpha = state_vector[S+P:].reshape(S, P)
    else:
        alpha = None

    if per_resource_growth_yield is None:
        per_resource_growth_yield = np.ones(P)

    # ----- uptake -----
    if use_monod_kinetics:
        assert maximum_uptake_rate_matrix is not None and half_saturation_matrix is not None
        base_uptake_rate = maximum_uptake_rate_matrix * (resources / (half_saturation_matrix + resources))
    else:
        assert consumption_matrix is not None
        base_uptake_rate = consumption_matrix * resources  # linear uptake

    if use_adaptive_uptake:
        effective_uptake_rate = alpha * base_uptake_rate
    else:
        effective_uptake_rate = base_uptake_rate

    uptake_per_resource = (populations @ effective_uptake_rate)

    # ----- species dynamics -----
    growth_input = (effective_uptake_rate * per_resource_growth_yield).sum(axis=1)
    if energy_accounting in ("strict", "marsland"):
        growth_input *= (1 - leakage_fraction)
    dN = populations * (growth_input - species_maintenance_rate)

    # ----- resource dynamics -----
    if renewal_mode == "chemostat":
        assert resource_supply_rate is not None and resource_decay_rate is not None
        renewal = resource_supply_rate - resource_decay_rate * resources
    else:
        assert resource_intrinsic_growth is not None and resource_carrying_capacity is not None
        renewal = resource_intrinsic_growth * (resource_carrying_capacity - resources) * resources

    if byproduct_matrix is not None and leakage_fraction > 0:
        leaked_return = leakage_fraction * (byproduct_matrix @ uptake_per_resource)
    else:
        leaked_return = 0.0

    dR = renewal - uptake_per_resource + leaked_return

    # ----- adaptive uptake -----
    if use_adaptive_uptake:
        assert adaptive_value_weight is not None
        benefit = adaptive_value_weight * base_uptake_rate
        mean_benefit = (alpha * benefit).sum(axis=1, keepdims=True)
        d_alpha = adaptive_learning_rate * alpha * (benefit - mean_benefit)
        return np.concatenate([dN, dR, d_alpha.flatten()])

    return np.concatenate([dN, dR])


# =========================================================
# Utilities
# =========================================================
def make_byproduct_matrix(p: int, seed: int = 1) -> NDArray[np.floating]:
    rng = np.random.default_rng(seed)
    D = rng.random((p, p))
    np.fill_diagonal(D, 0)
    D /= D.sum(axis=0, keepdims=True)
    return D


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# =========================================================
# Run Simulation (with lambda wrapper)
# =========================================================
def run_simulation(
    model_name: str,
    save_dir: str,
    *,
    num_species: int,
    num_resources: int,
    time_span: tuple[float, float],
    time_points: np.ndarray,
    initial_populations: np.ndarray,
    initial_resources: np.ndarray,
    **kwargs
):
    """Integrate and save results using a lambda to bind keyword args."""
    if kwargs.get("use_adaptive_uptake", False):
        alpha0 = np.random.dirichlet(np.ones(num_resources), size=num_species)
        y0 = np.concatenate([initial_populations, initial_resources, alpha0.flatten()])
    else:
        y0 = np.concatenate([initial_populations, initial_resources])

    rhs = lambda t, y: unified_consumer_resource_rhs(
        t, y, num_species=num_species, num_resources=num_resources, **kwargs
    )

    sol = solve_ivp(rhs, time_span, y0, t_eval=time_points, rtol=1e-7, atol=1e-9)

    ensure_dir(save_dir)
    np.savetxt(os.path.join(save_dir, f"{model_name}_time.csv"), sol.t, delimiter=",")
    np.savetxt(os.path.join(save_dir, f"{model_name}_populations.csv"),
               sol.y[:num_species, :].T, delimiter=",")
    np.savetxt(os.path.join(save_dir, f"{model_name}_resources.csv"),
               sol.y[num_species:num_species+num_resources, :].T, delimiter=",")

    if kwargs.get("use_adaptive_uptake", False) and sol.success:
        start = num_species + num_resources
        alpha = sol.y[start:, :].reshape(num_species, num_resources, -1)
        for s in range(num_species):
            np.savetxt(os.path.join(save_dir, f"{model_name}_alpha_species{s+1}.csv"),
                       alpha[s].T, delimiter=",")

    return sol


# =========================================================
# Plotting (save-only)
# =========================================================
def plot_series(filename: str, t: np.ndarray, Y: np.ndarray, label_prefix: str, ylabel: str, title: str):
    plt.figure(figsize=(9, 6))
    for i in range(Y.shape[0]):
        plt.plot(t, Y[i], label=f"{label_prefix} {i+1}")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_alpha(filename: str, t: np.ndarray, alpha_traj: np.ndarray):
    S, P, _ = alpha_traj.shape
    plt.figure(figsize=(10, 6))
    for s in range(S):
        for r in range(P):
            plt.plot(t, alpha_traj[s, r, :], label=f"α(s{s+1}, r{r+1})")
    plt.xlabel("Time")
    plt.ylabel("Allocation weight")
    plt.title("Adaptive uptake allocation (α) evolution")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# =========================================================
# Main simulation sequence
# =========================================================
if __name__ == "__main__":

    out_dir = ensure_dir("crm_simulation_results")
    num_species, num_resources = 3, 4
    time_span = (0, 150)
    time_points = np.linspace(0, 150, 800)

    consumption_matrix = np.array([
        [0.8, 0.3, 0.0, 0.2],
        [0.1, 0.7, 0.2, 0.4],
        [0.4, 0.1, 0.9, 0.3],
    ])
    species_maintenance_rate = np.full(num_species, 0.25)
    per_resource_growth_yield = np.ones(num_resources)
    resource_supply_rate = np.full(num_resources, 1.0)
    resource_decay_rate = np.full(num_resources, 1.0)
    byproduct_matrix = make_byproduct_matrix(num_resources, seed=2)

    initial_populations = np.full(num_species, 0.05)
    initial_resources = np.full(num_resources, 1.0)

    # -----------------------------
    # 1. Classical MacArthur
    # -----------------------------
    sol_mac = run_simulation(
        "MacArthur", out_dir,
        num_species=num_species,
        num_resources=num_resources,
        time_span=time_span,
        time_points=time_points,
        initial_populations=initial_populations,
        initial_resources=initial_resources,
        consumption_matrix=consumption_matrix,
        species_maintenance_rate=species_maintenance_rate,
        per_resource_growth_yield=per_resource_growth_yield,
        renewal_mode="chemostat",
        resource_supply_rate=resource_supply_rate,
        resource_decay_rate=resource_decay_rate,
        leakage_fraction=0.0,
        byproduct_matrix=None
    )
    plot_series(os.path.join(out_dir, "macarthur_species.png"), sol_mac.t, sol_mac.y[:num_species, :],
                "Species", "Abundance", "MacArthur Species")
    plot_series(os.path.join(out_dir, "macarthur_resources.png"), sol_mac.t,
                sol_mac.y[num_species:num_species+num_resources, :], "Resource", "Concentration", "MacArthur Resources")

    # -----------------------------
    # 2. Goldford (Cross-Feeding)
    # -----------------------------
    sol_gold = run_simulation(
        "Goldford", out_dir,
        num_species=num_species,
        num_resources=num_resources,
        time_span=time_span,
        time_points=time_points,
        initial_populations=initial_populations,
        initial_resources=initial_resources,
        consumption_matrix=consumption_matrix,
        species_maintenance_rate=species_maintenance_rate,
        per_resource_growth_yield=per_resource_growth_yield,
        renewal_mode="chemostat",
        resource_supply_rate=resource_supply_rate,
        resource_decay_rate=resource_decay_rate,
        byproduct_matrix=byproduct_matrix,
        leakage_fraction=0.3,
        energy_accounting="none"
    )
    plot_series(os.path.join(out_dir, "goldford_species.png"), sol_gold.t, sol_gold.y[:num_species, :],
                "Species", "Abundance", "Goldford Cross-feeding Species")

    # -----------------------------
    # 3. Marsland (Leakage)
    # -----------------------------
    sol_mars = run_simulation(
        "Marsland", out_dir,
        num_species=num_species,
        num_resources=num_resources,
        time_span=time_span,
        time_points=time_points,
        initial_populations=initial_populations,
        initial_resources=initial_resources,
        consumption_matrix=consumption_matrix,
        species_maintenance_rate=species_maintenance_rate,
        per_resource_growth_yield=per_resource_growth_yield,
        renewal_mode="chemostat",
        resource_supply_rate=resource_supply_rate,
        resource_decay_rate=resource_decay_rate,
        byproduct_matrix=byproduct_matrix,
        leakage_fraction=0.3,
        energy_accounting="marsland"
    )
    plot_series(os.path.join(out_dir, "marsland_species.png"), sol_mars.t, sol_mars.y[:num_species, :],
                "Species", "Abundance", "Marsland Leakage Species")

    # -----------------------------
    # 4. Pacciani–Mori (Adaptive)
    # -----------------------------
    sol_adapt = run_simulation(
        "Adaptive", out_dir,
        num_species=num_species,
        num_resources=num_resources,
        time_span=time_span,
        time_points=time_points,
        initial_populations=initial_populations,
        initial_resources=initial_resources,
        consumption_matrix=consumption_matrix,
        species_maintenance_rate=species_maintenance_rate,
        per_resource_growth_yield=per_resource_growth_yield,
        renewal_mode="chemostat",
        resource_supply_rate=resource_supply_rate,
        resource_decay_rate=resource_decay_rate,
        use_adaptive_uptake=True,
        adaptive_value_weight=np.array([1.0, 1.5, 0.8, 1.2]),
        adaptive_learning_rate=0.5
    )
    plot_series(os.path.join(out_dir, "adaptive_species.png"), sol_adapt.t, sol_adapt.y[:num_species, :],
                "Species", "Abundance", "Adaptive Uptake Species")
    plot_series(os.path.join(out_dir, "adaptive_resources.png"), sol_adapt.t,
                sol_adapt.y[num_species:num_species+num_resources, :], "Resource", "Concentration", "Adaptive Uptake Resources")

    if sol_adapt.success:
        start = num_species + num_resources
        alpha_traj = sol_adapt.y[start:, :].reshape(num_species, num_resources, -1)
        plot_alpha(os.path.join(out_dir, "adaptive_alpha.png"), sol_adapt.t, alpha_traj)

    print(f"✅ All results saved in {os.path.abspath(out_dir)}")