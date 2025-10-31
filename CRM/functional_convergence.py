import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# =========================================================
# helpers
# =========================================================
def get_metabolism(num_resources, p=0.1):
    """Random cross-feeding, same for everyone."""
    D = np.random.rand(num_resources, num_resources) * p
    return D


def make_unstructured_pool(total_species, num_resource_traits, alpha=5.0):
    """
    Build ONE big trait cloud:
    - every species ~ Dirichlet(alpha,...,alpha)
    - no families, no labels
    """
    a = np.full(num_resource_traits, alpha, dtype=float)
    C = np.random.dirichlet(a, size=total_species)
    return C  # shape (total_species, num_resource_traits)


def mcrm_params(resource_idx, C_sample, D):
    num_species = C_sample.shape[0]
    num_resources = C_sample.shape[1]

    var_idx = {
        "species": list(range(num_species)),
        "resources": list(range(num_species, num_species + num_resources)),
    }

    W = np.eye(num_resources)

    alpha = np.zeros(num_resources)
    alpha[resource_idx] = 5.0  # same supply for everyone

    x0 = np.random.rand(num_species + num_resources)

    return {
        "num_species": num_species,
        "num_resources": num_resources,
        "varIdx": var_idx,
        "C": C_sample,
        "D": D,
        "W": W,
        "alpha": alpha,
        "death_rate": np.zeros(num_species),
        "mu": 1.0,
        "T": 1.0,
        "tau": np.ones(num_resources),
        "x0": x0,
        "timeStep": 5000,
    }


def population_dynamics(t, x, params):
    v = params["varIdx"]
    C, D = params["C"], params["D"]
    alpha, tau = params["alpha"], params["tau"]

    N = x[v["species"]]
    R = x[v["resources"]]

    dx = np.zeros_like(x)

    # growth
    growth = C @ (params["W"] @ R) - params["T"]
    dx[v["species"]] = N * params["mu"] * growth  # no death in this version

    # resources
    consumption = (C * R).T @ N           # resource-wise
    production = D @ consumption          # leaked/byproducts
    dx[v["resources"]] = (alpha - R) / tau - consumption + production

    return dx


def run_mcrm(params, num_points=3000):
    x0 = params["x0"]
    t_eval = np.linspace(0, params["timeStep"], num_points)
    sol = solve_ivp(lambda t, x: population_dynamics(t, x, params),
                    [0, params["timeStep"]], x0,
                    method="LSODA", t_eval=t_eval)

    v = params["varIdx"]
    return {
        "species": sol.y[v["species"], -1],
        "resources": sol.y[v["resources"], -1],
    }


# =========================================================
# main experiment
# =========================================================
def main():
    np.random.seed(0)
    save_dir = "results_unstructured"
    os.makedirs(save_dir, exist_ok=True)

    # ---------------- experiment design ----------------
    num_supplied_resources = 5      # R0..R4 we will test
    total_species = 1000            # global pool
    subset_size = 50                # sampled into communities
    reps_per_resource = 30          # per resource
    num_resource_traits = 10        # dimension of C
    dirichlet_alpha = 5.0           # 1–5 is good: 1 = spiky, 5 = smoother

    # 1) global unstructured pool
    C_pool = make_unstructured_pool(
        total_species,
        num_resource_traits,
        alpha=dirichlet_alpha
    )

    # 2) fixed cross-feeding for everyone
    D_global = get_metabolism(num_resource_traits, p=1 / num_resource_traits)

    results = []

    # 3) simulate
    for r_idx in range(num_supplied_resources):
        for _ in range(reps_per_resource):
            # unbiased sampling: just pick 50 out of 1000
            spp_idx = np.random.choice(total_species, subset_size, replace=False)
            C_sample = C_pool[spp_idx]

            params = mcrm_params(r_idx, C_sample, D_global)
            sim = run_mcrm(params)
            N_final = sim["species"]
            R_final = sim["resources"]

            # functional outcome: how much of *supplied* resource can this random community use?
            uptake_vec = C_sample.T @ N_final
            func_scalar = uptake_vec[r_idx]

            results.append({
                "supplied_resource": r_idx,
                "uptake_scalar": func_scalar,
                "n_survivors": int(np.sum(N_final > 1e-6)),
                "species_abundance": N_final,
                "uptake_vec": uptake_vec
            })

    # ---------------- save + plots ----------------
    save_dir = "results_unstructured"
    os.makedirs(save_dir, exist_ok=True)

    # Convert C_pool to a dataframe
    df_pool = pd.DataFrame(C_pool, columns=[f"R{j + 1}" for j in range(C_pool.shape[1])])

    # 1️⃣ Violin plot: distribution of consumption preferences per resource
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=df_pool, inner="quartile", palette="viridis", cut=0)
    plt.xlabel("Resource")
    plt.ylabel("Preference Weight")
    plt.title("Global Pool: Distribution of Resource Preferences")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/global_pool_violin.png", dpi=300)
    plt.close()

    # 2️⃣ Heatmap: each row = species, columns = resource preferences
    plt.figure(figsize=(8, 5))
    sns.heatmap(df_pool, cmap="mako", cbar_kws={"label": "Preference Weight"})
    plt.xlabel("Resource")
    plt.ylabel("Species (index)")
    plt.title("Global Pool: Species × Resource Preference Matrix (C)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/global_pool_heatmap.png", dpi=300)
    plt.close()

    # 3️⃣ Histogram of specialization index (how uneven each species’ preferences are)
    specialization = np.sum(df_pool.values ** 2, axis=1)  # closer to 1 → specialist, 1/n → generalist
    plt.figure(figsize=(6, 4))
    sns.histplot(specialization, bins=20, kde=True, color="teal")
    plt.xlabel("∑(Cᵢⱼ²)  (Specialization Index)")
    plt.ylabel("Count")
    plt.title("Global Pool: Specialization Distribution Across Species")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/global_pool_specialization_hist.png", dpi=300)
    plt.close()

    print("✅ Saved plots: global_pool_violin.png, global_pool_heatmap.png, global_pool_specialization_hist.png")

    df = pd.DataFrame({
        "supplied_resource": [r["supplied_resource"] for r in results],
        "uptake_scalar": [r["uptake_scalar"] for r in results],
        "n_survivors": [r["n_survivors"] for r in results],
    })
    df.to_csv(f"{save_dir}/functional_convergence_unstructured.csv", index=False)

    # A) functional convergence box
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="supplied_resource", y="uptake_scalar")
    plt.xlabel("Supplied resource index")
    plt.ylabel("Total uptake of that resource")
    plt.title("Functional convergence – unstructured global pool")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fc_box_unstructured.png", dpi=300)
    plt.close()

    # B) richness box
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="supplied_resource", y="n_survivors")
    plt.xlabel("Supplied resource index")
    plt.ylabel("Final richness (survivors)")
    plt.title("Richness per supplied resource")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/richness_box_unstructured.png", dpi=300)
    plt.close()

    # C) histograms of uptake per resource
    for r_idx in range(num_supplied_resources):
        sub = df[df.supplied_resource == r_idx]
        plt.figure(figsize=(5, 3))
        plt.hist(sub["uptake_scalar"], bins=12, alpha=0.8)
        plt.xlabel("Total uptake")
        plt.ylabel("Count")
        plt.title(f"Uptake distribution – resource {r_idx}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/uptake_hist_r{r_idx}.png", dpi=300)
        plt.close()

    # D) diagnostic heatmaps
    species_mat = np.vstack([r["species_abundance"] for r in results])
    uptake_mat = np.vstack([r["uptake_vec"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(species_mat, ax=axes[0], cmap="mako")
    axes[0].set_title("Species-level structure")
    axes[0].set_xlabel("Species")
    axes[0].set_ylabel("Simulation")

    sns.heatmap(uptake_mat, ax=axes[1], cmap="mako")
    axes[1].set_title("Uptake vector")
    axes[1].set_xlabel("Resource")
    axes[1].set_ylabel("Simulation")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/heatmaps_unstructured.png", dpi=300)
    plt.close()

    print(f"✅ done. Saved plots + CSV in `{save_dir}/`")


if __name__ == "__main__":
    main()