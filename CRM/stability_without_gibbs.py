"""
compare_stability_all_models_v7.py

- 4 models (all 5×5):
    1. Classical MacArthur
    2. MacArthur + Crossfeeding (MCRM-style)
    3. MacArthur + Leakage (MiCRM-style)
    4. MacArthur + Adaptive Uptake (slowed, damped)

- 3-layer stability test:
    (i)   local stability via Jacobian eigenvalues
    (ii)  empirical basin via Monte Carlo ICs
    (iii) Hopf-style 1D sweep (parameter neighborhood robustness)

- Outputs:
    stability_results.csv
    stability_params.csv   <-- simulation / model params
    stability_results.png
    stability_local_vs_basin.png
    stability_hopf_safe.png
    stability_eig_sweeps.png
    stability_radar.png
    stability_basin_hist.png
    stability_local_vs_global_bar.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from scipy.integrate import solve_ivp

# adjust to your layout
from community_models import CommunityModels


# ============================================================
# 0. utility: per-variable tolerance
# ============================================================
def close_enough(x, x_star, abs_tol=1e-2, rel_tol=5e-2):
    """
    L_inf-style mixed tolerance:
        |x - x*| <= max(abs_tol, rel_tol * |x*|)
    """
    x = np.asarray(x, float)
    x_star = np.asarray(x_star, float)
    tol = np.maximum(abs_tol, rel_tol * np.abs(x_star))
    return np.all(np.abs(x - x_star) <= tol)


# ============================================================
# 1. common parameter generator
# ============================================================
def build_common_params(seed=1, S=5, R=5):
    rng = np.random.default_rng(seed)

    # smallish consumption
    C = rng.uniform(0.03, 0.12, size=(S, R))

    # cross-feeding base
    D = rng.uniform(0.0, 0.2, size=(R, R))
    np.fill_diagonal(D, 0.0)
    D = D / (D.sum(axis=0, keepdims=True) + 1e-9)

    return {
        "S": S,
        "R": R,
        "C": C,
        "D": D,
        "kappa": np.full(R, 2.0),
        "tau": np.full(R, 2.0),
        "m": np.full(S, 0.18),
        "w": np.ones(R),
        "N0": np.full(S, 0.1),
        "R0": np.full(R, 2.5),
        "rng": rng
    }


# ============================================================
# 2. numerical building blocks
# ============================================================
def integrate_to_equilibrium(f, x0, T=400.0, atol=1e-8, rtol=1e-6):
    sol = solve_ivp(f, (0.0, T), x0, method="BDF", atol=atol, rtol=rtol)
    return sol.y[:, -1]


def finite_diff_jacobian(f, x_star, eps=1e-6):
    x_star = np.asarray(x_star, float)
    n = x_star.size
    J = np.zeros((n, n), float)
    f0 = f(0.0, x_star)
    for j in range(n):
        x_pert = x_star.copy()
        x_pert[j] += eps
        fj = f(0.0, x_pert)
        J[:, j] = (fj - f0) / eps
    return J


def local_stability_score(f, x_star):
    J = finite_diff_jacobian(f, x_star)
    lam = eigvals(J)
    max_real = float(np.max(np.real(lam)))
    raw = max(0.0, -max_real)
    return raw, max_real


def basin_fraction(
    f,
    x_star,
    Omega_low,
    Omega_high,
    K=1000,
    T=400.0,
    abs_tol=1e-2,
    rel_tol=5e-2,
    verbose=False,
    seed=123,
    collect_dist=False,
):
    rng = np.random.default_rng(seed)
    n = x_star.size
    count = 0
    dists = []
    for k in range(K):
        x0 = rng.uniform(Omega_low, Omega_high, size=n)
        xT = integrate_to_equilibrium(f, x0, T=T)
        diff = xT - x_star
        dist = np.linalg.norm(diff)
        dists.append(dist)
        if close_enough(xT, x_star, abs_tol=abs_tol, rel_tol=rel_tol):
            count += 1
        if verbose and k % 100 == 0:
            print(f"      basin sample {k:4d}/{K} → running = {count / (k + 1):.3f}")
    frac = count / K
    if collect_dist:
        return frac, np.array(dists)
    return frac, None


# ============================================================
# 3. build model-specific params from the common core
# ============================================================
def build_all_params_from_common(common):
    S = common["S"]
    R = common["R"]
    C = common["C"]
    D = common["D"]
    kappa = common["kappa"]
    tau = common["tau"]
    m = common["m"]
    w = common["w"]
    rng = common["rng"]

    # 1) Classical MacArthur
    params_classical = dict(
        tau=2.0,
        m=m,
        w=w,
        c=C,
        r=np.full(R, 2.0),
        K=np.full(R, 5.0),
    )

    # 2) MacArthur + Crossfeeding
    params_mcrm = dict(
        varIdx={'species': np.arange(S), 'resources': np.arange(S, S + R)},
        C=C,
        D=D,
        B=np.zeros(R),
        T=np.zeros(S),
        alpha=kappa,
        tau=tau,
        death_rate=m,
        mu=np.ones(S),
        W=np.ones((R, R)),
        production_scale=0.2,    # swept param for Hopf
        resource_decay=0.1,
    )

    # 3) MacArthur + Leakage (MiCRM-style)
    params_micrm = dict(
        num_resources=R,
        C=C,
        D=D,
        leakage=0.1,             # swept param for Hopf
        rho=kappa,
        tau=tau,
        w=np.ones(S),
        m=m,
        g=1.0
    )

    # 4) MacArthur + Adaptive Uptake (SLOWED & STRONGLY DAMPED)
    params_adaptive = dict(
        S=S, R=R,
        v=np.ones(R),
        K=np.full(R, 2.0),           # slightly lower carrying
        d=m,
        s=np.full(R, 1.5),           # a bit lower supply → calmer R
        mu=np.full(R, 0.1),
        lam=np.full(S, 0.02),        # << slow adaptation
        E_star=np.full(S, 0.8),
        A_damp=0.12                  # << strong damping
    )

    return (
        params_classical,
        params_mcrm,
        params_micrm,
        params_adaptive,
    )


# ============================================================
# 4. Hopf-style sweep (1D)
# ============================================================
def hopf_sweep(model_name, base_params, f_builder, x0,
               pname=None, n_points=10, span=0.3, T=200.0):
    # pick param to sweep
    if pname is None:
        if "Crossfeeding" in model_name or "MCRM" in model_name:
            pname = "production_scale"
        elif "Leakage" in model_name or "MiCRM" in model_name:
            pname = "leakage"
        elif "Adaptive" in model_name:
            pname = "lam"
        else:
            pname = "r" if "r" in base_params else "tau"

    base_val = base_params[pname]
    if np.isscalar(base_val):
        p0 = float(base_val)
    else:
        p0 = float(np.array(base_val).flat[0])

    plist = np.linspace((1 - span) * p0, (1 + span) * p0, n_points)
    max_reals = []

    for pval in plist:
        test_params = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                       for k, v in base_params.items()}
        if np.isscalar(base_val):
            test_params[pname] = float(pval)
        else:
            arr = np.array(base_val, dtype=float)
            arr[...] = pval
            test_params[pname] = arr

        f = f_builder(test_params)
        x_eq = integrate_to_equilibrium(f, x0, T=T)
        J = finite_diff_jacobian(f, x_eq)
        lam = eigvals(J)
        max_re = float(np.max(np.real(lam)))
        max_reals.append(max_re)

    hopf_ok = 1.0 if max(max_reals) < 0.0 else 0.0
    return hopf_ok, plist, np.array(max_reals)


# ============================================================
# 5. model analysis wrapper
# ============================================================
def analyze_model(
    name,
    f_builder,
    base_params,
    x0,
    Omega_low,
    Omega_high,
    K_basin=1000,
    T=400.0,
    abs_tol=1e-2,
    rel_tol=5e-2,
    verbose=True
):
    print(f"\n🔹 Analyzing {name} ...")

    f = f_builder(base_params)
    x_star = integrate_to_equilibrium(f, x0, T=T)
    print(f"   → steady-state L2-norm: {np.linalg.norm(x_star):.4f}")

    # 1) local
    local_raw, max_real = local_stability_score(f, x_star)
    print(f"   → local: max Re(λ) = {max_real:.5f}, local_raw = {local_raw:.5f}")

    # 2) Hopf
    hopf_ok, _, _ = hopf_sweep(name, base_params, f_builder, x0)
    print(f"   → hopf-safe = {hopf_ok}")

    # if not locally stable → basin = 0 (but keep hopf)
    if local_raw <= 0.0:
        print("   → not locally stable → basin = 0")
        return {
            "Model": name,
            "eq_norm": float(np.linalg.norm(x_star)),
            "max_real_eig": max_real,
            "local_score": 0.0,
            "basin_fraction": 0.0,
            "hopf_safe": hopf_ok,
            "stability_score": 0.2 * hopf_ok,
            "basin_dists": None
        }

    # 3) basin
    print(f"   → estimating basin, K = {K_basin} ...")
    basin, dists = basin_fraction(
        f,
        x_star,
        Omega_low,
        Omega_high,
        K=K_basin,
        T=T,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        verbose=verbose,
        collect_dist=True
    )
    print(f"   → basin fraction = {basin:.4f}")

    # 4) normalize local
    local_score = local_raw / (1.0 + local_raw)

    # 5) composite
    stability_score = 0.4 * local_score + 0.4 * basin + 0.2 * hopf_ok
    print(f"   → composite stability score = {stability_score:.5f}")

    return {
        "Model": name,
        "eq_norm": float(np.linalg.norm(x_star)),
        "max_real_eig": max_real,
        "local_score": local_score,
        "basin_fraction": basin,
        "hopf_safe": hopf_ok,
        "stability_score": stability_score,
        "basin_dists": dists
    }


# ============================================================
# 6. main
# ============================================================
def main():
    # shared
    common = build_common_params(seed=1, S=5, R=5)
    (params_classical,
     params_mcrm,
     params_micrm,
     params_adaptive) = build_all_params_from_common(common)

    S = common["S"]; R = common["R"]

    # instantiate for internal dynamics
    models = CommunityModels(
        classical_params=params_classical,
        mcrm_params=params_mcrm,
        micrm_params=params_micrm,
        adaptive_params=params_adaptive
    )

    # ICs + boxes
    x0_core = np.concatenate([common["N0"], common["R0"]])
    Omega_low_core = np.concatenate([np.zeros(S), np.zeros(R)])
    Omega_high_core = np.concatenate([np.full(S, 1.0), np.full(R, 20.0)])

    x0_RN = np.concatenate([common["R0"], common["N0"]])
    Omega_low_RN = np.concatenate([np.zeros(R), np.zeros(S)])
    Omega_high_RN = np.concatenate([np.full(R, 80.0), np.full(S, 1.0)])

    x0_adaptive = np.concatenate([common["N0"], common["R0"], np.full(S * R, 0.1)])
    Omega_low_adaptive = np.zeros_like(x0_adaptive)
    Omega_high_adaptive = np.concatenate([
        np.full(S, 1.0),
        np.full(R, 20.0),
        np.full(S * R, 1.0)
    ])

    # RHS builders
    def f_builder_classical(params):
        def f(t, x):
            N = x[:S]
            Rv = x[S:]
            tau = params["tau"]
            m = params["m"]
            w = params["w"]
            c = params["c"]
            r = params["r"]
            K = params["K"]

            growth_input = (c * w.reshape(1, -1)) @ Rv
            dN = (N / tau) * (growth_input - m)

            consumption = (N @ c) * Rv
            regeneration = (r / K) * (K - Rv) * Rv
            dR = regeneration - consumption
            return np.concatenate([dN, dR])
        return f

    def f_builder_mcrm(params):
        def f(t, x):
            p = params
            v = p['varIdx']
            C = p['C']; D = p['D']
            alpha = p['alpha']; tau = p['tau']
            death_rate = p['death_rate']
            mu = p['mu']
            W = p['W']
            prod_scale = p.get('production_scale', 0.2)
            res_decay = p.get('resource_decay', 0.1)

            N = x[v['species']]
            R = x[v['resources']]
            dx = np.zeros_like(x)

            growth_term = C @ (W @ R) - p['T']
            dx[v['species']] = N * mu * growth_term - death_rate * N

            consumption = (C * R).T @ N
            production = prod_scale * (D @ consumption)
            dx[v['resources']] = (alpha - R) / tau - consumption + production - res_decay * R
            return dx
        return f

    def f_builder_micrm(params):
        def f(t, x):
            models.params_micrm = params
            return models._micrm_dynamics(t, x)
        return f

    def f_builder_adaptive(params):
        def f(t, x):
            models.params_adaptive = params
            dx = models.adaptive_crm_dynamics(t, x)
            A_damp = params.get("A_damp", 0.0)
            if A_damp > 0:
                S_ = params["S"]
                R_ = params["R"]
                A_slice = slice(S_ + R_, S_ + R_ + S_ * R_)
                dx[A_slice] -= A_damp * x[A_slice]
            return dx
        return f

    # run analyses
    results = []
    basin_logs = []

    # 1) Classical
    res1 = analyze_model(
        "Classical MacArthur",
        f_builder_classical,
        params_classical,
        x0_core,
        Omega_low_core,
        Omega_high_core,
        K_basin=1000,
        T=400.0,
        verbose=True
    )
    results.append(res1); basin_logs.append(("Classical MacArthur", res1["basin_dists"]))

    # 2) Crossfeeding
    res2 = analyze_model(
        "MacArthur + Crossfeeding (MCRM)",
        f_builder_mcrm,
        params_mcrm,
        x0_core,
        Omega_low_core,
        np.concatenate([np.full(S, 1.0), np.full(R, 80.0)]),
        K_basin=1000,
        T=400.0,
        verbose=True
    )
    results.append(res2); basin_logs.append(("MacArthur + Crossfeeding (MCRM)", res2["basin_dists"]))

    # 3) Leakage
    res3 = analyze_model(
        "MacArthur + Leakage (MiCRM)",
        f_builder_micrm,
        params_micrm,
        x0_RN,
        Omega_low_RN,
        Omega_high_RN,
        K_basin=1000,
        T=400.0,
        verbose=True
    )
    results.append(res3); basin_logs.append(("MacArthur + Leakage (MiCRM)", res3["basin_dists"]))

    # 4) Adaptive
    res4 = analyze_model(
        "MacArthur + Adaptive Uptake",
        f_builder_adaptive,
        params_adaptive,
        x0_adaptive,
        Omega_low_adaptive,
        Omega_high_adaptive,
        K_basin=800,          # a bit lower for high-dim system
        T=500.0,              # longer integration
        abs_tol=2e-2,
        rel_tol=0.1,
        verbose=True
    )
    results.append(res4); basin_logs.append(("MacArthur + Adaptive Uptake", res4["basin_dists"]))

    # build dataframe
    df = pd.DataFrame(results)

    # add stable_flag (for plotting)
    def is_stable_row(row):
        return (row["local_score"] > 0.0) and (row["basin_fraction"] > 0.0)
    df["stable_flag"] = df.apply(is_stable_row, axis=1)

    # save stability results
    df.to_csv("stability_results.csv", index=False)
    print("\n✅ Saved to stability_results.csv")
    print(df)

    # --------------------------------------------------------
    # ALSO SAVE SIMULATION PARAMETERS
    # --------------------------------------------------------
    sim_params = []

    sim_params.append({
        "Model": "Classical MacArthur",
        "S": common["S"],
        "R": common["R"],
        "tau": 2.0,
        "supply": 2.0,
        "death": 0.18,
        "special": "—"
    })
    sim_params.append({
        "Model": "MacArthur + Crossfeeding (MCRM)",
        "S": common["S"],
        "R": common["R"],
        "tau": 2.0,
        "supply": 2.0,
        "death": 0.18,
        "special": "production_scale=0.2, resource_decay=0.1"
    })
    sim_params.append({
        "Model": "MacArthur + Leakage (MiCRM)",
        "S": common["S"],
        "R": common["R"],
        "tau": 2.0,
        "supply": 2.0,
        "death": 0.18,
        "special": "leakage=0.1"
    })
    sim_params.append({
        "Model": "MacArthur + Adaptive Uptake",
        "S": common["S"],
        "R": common["R"],
        "tau": 2.0,
        "supply": 1.5,
        "death": 0.18,
        "special": "lam=0.02, A_damp=0.12, E*=0.8"
    })

    df_params = pd.DataFrame(sim_params)
    df_params.to_csv("stability_params.csv", index=False)
    print("✅ Saved to stability_params.csv")

    # ========================================================
    #  PLOTS
    # ========================================================
    # 1) composite bar (red = not stable)
    plt.figure(figsize=(9, 4))
    colors = ["tab:blue" if flag else "tab:red" for flag in df["stable_flag"]]
    plt.bar(df["Model"], df["stability_score"], color=colors)
    plt.ylabel("Composite Stability Score")
    plt.xticks(rotation=20, ha="right")
    plt.title("Stability Across CRM Variants (local + basin + Hopf)")
    plt.tight_layout()
    plt.savefig("stability_results.png", dpi=300)
    print("✅ Saved plot to stability_results.png")

    # 2) local vs basin scatter
    plt.figure(figsize=(5, 4))
    plt.scatter(df["local_score"], df["basin_fraction"])
    for _, row in df.iterrows():
        plt.text(row["local_score"] * 1.01,
                 row["basin_fraction"] * 1.01,
                 row["Model"],
                 fontsize=7)
    plt.xlabel("Local stability (σ / (1+σ))")
    plt.ylabel("Basin fraction")
    plt.title("Local vs Global Stability")
    plt.tight_layout()
    plt.savefig("stability_local_vs_basin.png", dpi=300)
    print("✅ Saved plot to stability_local_vs_basin.png")

    # 3) Hopf-safe plot
    plt.figure(figsize=(6, 2.5))
    colors = ["green" if h == 1 else "red" for h in df["hopf_safe"]]
    plt.barh(df["Model"], [1] * len(df), color=colors)
    plt.xlim(0, 1)
    plt.xticks([])
    plt.title("Hopf sweep outcome (green = stable in neighborhood)")
    plt.tight_layout()
    plt.savefig("stability_hopf_safe.png", dpi=300)
    print("✅ Saved plot to stability_hopf_safe.png")

    # 4) eigenvalue sweeps
    plt.figure(figsize=(8, 5))
    sweep_specs = [
        ("Classical MacArthur", params_classical, f_builder_classical, x0_core),
        ("MacArthur + Crossfeeding (MCRM)", params_mcrm, f_builder_mcrm, x0_core),
        ("MacArthur + Leakage (MiCRM)", params_micrm, f_builder_micrm, x0_RN),
        ("MacArthur + Adaptive Uptake", params_adaptive, f_builder_adaptive, x0_adaptive),
    ]
    for name, pbase, fb, x0_ in sweep_specs:
        _, pvals, maxre = hopf_sweep(name, pbase, fb, x0_, n_points=15, span=0.3, T=200.0)
        plt.plot(pvals, maxre, marker="o", label=name)
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.ylabel("max Re(λ)")
    plt.xlabel("bifurcation parameter (model-specific)")
    plt.title("Eigenvalue flow across parameter neighborhood")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("stability_eig_sweeps.png", dpi=300)
    print("✅ Saved plot to stability_eig_sweeps.png")

    # 5) radar chart
    metrics = ["local_score", "basin_fraction", "hopf_safe"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(6, 6))
    for _, row in df.iterrows():
        vals = [row[m] for m in metrics]
        vals += vals[:1]
        plt.polar(angles, vals, label=row["Model"])
    plt.xticks(angles[:-1], metrics)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.25, 1.0), fontsize=7)
    plt.title("Stability components per model")
    plt.savefig("stability_radar.png", dpi=300)
    print("✅ Saved plot to stability_radar.png")

    # 6) basin distance histograms
    fig, axes = plt.subplots(1, len(basin_logs), figsize=(14, 3), sharey=True)
    for ax, (name, dists) in zip(axes, basin_logs):
        if dists is not None:
            ax.hist(np.log10(dists + 1e-12), bins=30)
        ax.set_title(name, fontsize=7)
        ax.set_xlabel("log10(dist)")
    axes[0].set_ylabel("count")
    fig.suptitle("Log-distance to equilibrium from random ICs")
    plt.tight_layout()
    plt.savefig("stability_basin_hist.png", dpi=300)
    print("✅ Saved plot to stability_basin_hist.png")

    # 7) local vs global bar
    plt.figure(figsize=(8, 5))
    width = 0.35
    x = np.arange(len(df))
    plt.bar(x - width / 2, df["local_score"], width, label="Local (Lnorm)")
    plt.bar(x + width / 2, df["basin_fraction"], width, label="Basin Fraction")
    plt.xticks(x, df["Model"], rotation=20, ha="right")
    plt.ylabel("Stability Component Value")
    plt.title("Local vs Global Stability Components per Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("stability_local_vs_global_bar.png", dpi=300)
    print("✅ Saved plot to stability_local_vs_global_bar.png")


if __name__ == "__main__":
    main()