"""
simulate_coexistence_named_params_fixed.py

- 4 models, each sampled under its own parameter distribution
- We measure: richness, Shannon
- We SAVE: model, richness, shannon, and the actual sampled parameters
- We PLOT:
    * richness boxplot
    * shannon boxplot
    * per-model parameter histograms (only for columns that exist / are non-NaN)
    * mean bar (richness + shannon)

Fixes vs previous:
- histograms no longer fail when a column is all NaN for a given model
- uses explicit parameter names per model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------ global config ------------------
N_SPECIES = 5
N_RES = 5
N_SAMPLES_PER_MODEL = 70
T_FINAL = 260.0
RICH_EPS = 1e-4
rng = np.random.default_rng(1234)


# ------------------ helpers ------------------
def integrate(f, x0, t_final=T_FINAL):
    sol = solve_ivp(f, (0.0, t_final), x0, method="BDF", atol=1e-8, rtol=1e-6)
    return sol.y[:, -1]


def richness(N, eps=RICH_EPS):
    return int(np.sum(N > eps))


def shannon(N, eps=RICH_EPS):
    mask = N > eps
    if not np.any(mask):
        return 0.0
    N_ = N[mask]
    p = N_ / np.sum(N_)
    return float(-np.sum(p * np.log(p + 1e-12)))


# ------------------ model RHS ------------------
def rhs_classical(C, w, s, tau, m):
    S, R = C.shape

    def f(t, x):
        N = x[:S]
        Rv = x[S:]
        grow = (C * w.reshape(1, -1)) @ Rv
        dN = (N / tau) * (grow - m)
        cons = (N @ C) * Rv
        dR = (s - Rv) / tau - cons
        return np.concatenate([dN, dR])

    return f


def rhs_mcrm(C, w, s, tau, m, D, production_scale=0.25, resource_decay=0.08):
    S, R = C.shape

    def f(t, x):
        N = x[:S]
        Rv = x[S:]
        grow = (C * w.reshape(1, -1)) @ Rv
        dN = N * (grow - m)
        cons = (C * Rv.reshape(1, -1)).T @ N
        prod = production_scale * (D @ cons)
        dR = (s - Rv) / tau - cons + prod - resource_decay * Rv
        return np.concatenate([dN, dR])

    return f


def rhs_micrm(C, w, s, tau, m, D, leakage=0.25):
    S, R = C.shape

    def f(t, x):
        N = x[:S]
        Rv = x[S:]
        # only (1 - leakage) counts for growth
        grow = (C * w.reshape(1, -1) * (1 - leakage)) @ Rv
        dN = N * (grow - m)
        cons = (C * Rv.reshape(1, -1)).T @ N
        leaked = leakage * cons
        byp = D @ leaked
        dR = (s - Rv) / tau - cons + byp
        return np.concatenate([dN, dR])

    return f


def rhs_adaptive(C, w, s, tau, m, adaptation_rate=0.04, adaptation_damping=0.04):
    S, R = C.shape

    def f(t, x):
        N = x[:S]
        Rv = x[S:S + R]
        A = x[S + R:].reshape(S, R)

        target = 0.8 * np.max(w * Rv) + 0.05  # crude energy target
        grow = np.sum(A * (w * Rv), axis=1)
        dN = N * (grow - m)

        cons = np.sum(A * N.reshape(-1, 1) * Rv.reshape(1, -1), axis=0)
        dR = (s - Rv) / tau - cons

        dA = adaptation_rate * A * (w * Rv.reshape(1, -1) - target) - adaptation_damping * A

        return np.concatenate([dN, dR, dA.ravel()])

    return f


# ------------------ main simulation ------------------
def main():
    S, R = N_SPECIES, N_RES

    # base trait matrices
    C_base = rng.uniform(0.04, 0.14, size=(S, R))
    D_base = rng.uniform(0.0, 0.3, size=(R, R))
    np.fill_diagonal(D_base, 0.0)
    D_base = D_base / (D_base.sum(axis=0, keepdims=True) + 1e-12)
    w = np.ones(R)

    records = []

    # 1) Classical MacArthur
    for _ in range(N_SAMPLES_PER_MODEL):
        s = rng.uniform(1.6, 3.6, size=R)
        tau = rng.uniform(1.2, 2.8)
        m = rng.uniform(0.09, 0.19, size=S)

        N0 = rng.uniform(0.05, 0.15, size=S)
        R0 = s * 0.75

        f = rhs_classical(C_base, w, s, tau, m)
        xeq = integrate(f, np.concatenate([N0, R0]))
        N_eq = xeq[:S]

        records.append(dict(
            model="Classical MacArthur",
            richness=richness(N_eq),
            shannon=shannon(N_eq),
            resource_turnover=tau,
            mean_supply=float(np.mean(s)),
        ))

    # 2) MCRM (Crossfeeding)
    for _ in range(N_SAMPLES_PER_MODEL):
        s = rng.uniform(1.2, 2.2, size=R)
        tau = 2.0
        m = rng.uniform(0.10, 0.20, size=S)
        production_scale = rng.uniform(0.12, 0.50)
        resource_decay = rng.uniform(0.03, 0.14)

        N0 = rng.uniform(0.05, 0.15, size=S)
        R0 = s * 0.85

        f = rhs_mcrm(C_base, w, s, tau, m, D_base,
                     production_scale=production_scale,
                     resource_decay=resource_decay)
        xeq = integrate(f, np.concatenate([N0, R0]))
        N_eq = xeq[:S]

        records.append(dict(
            model="MacArthur + Crossfeeding (MCRM)",
            richness=richness(N_eq),
            shannon=shannon(N_eq),
            production_scale=production_scale,
            resource_decay=resource_decay,
        ))

    # 3) MiCRM (Leakage)
    for _ in range(N_SAMPLES_PER_MODEL):
        s = rng.uniform(1.5, 2.9, size=R)
        tau = rng.uniform(1.4, 2.6)
        m = rng.uniform(0.10, 0.22, size=S)
        leakage = rng.uniform(0.05, 0.50)

        N0 = rng.uniform(0.05, 0.15, size=S)
        R0 = s * 0.8

        f = rhs_micrm(C_base, w, s, tau, m, D_base, leakage=leakage)
        xeq = integrate(f, np.concatenate([N0, R0]))
        N_eq = xeq[:S]

        records.append(dict(
            model="MacArthur + Leakage (MiCRM)",
            richness=richness(N_eq),
            shannon=shannon(N_eq),
            leakage=leakage,
            resource_turnover=tau,
        ))

    # 4) Adaptive Uptake
    for _ in range(N_SAMPLES_PER_MODEL):
        s = rng.uniform(1.6, 3.0, size=R)
        tau = 2.0
        m = rng.uniform(0.08, 0.16, size=S)
        adaptation_rate = rng.uniform(0.02, 0.07)
        adaptation_damping = rng.uniform(0.025, 0.09)

        N0 = rng.uniform(0.05, 0.15, size=S)
        R0 = s * 0.85
        A0 = rng.uniform(0.15, 0.45, size=(S, R))

        f = rhs_adaptive(C_base, w, s, tau, m,
                         adaptation_rate=adaptation_rate,
                         adaptation_damping=adaptation_damping)
        xeq = integrate(f, np.concatenate([N0, R0, A0.ravel()]), t_final=340.0)
        N_eq = xeq[:S]

        records.append(dict(
            model="MacArthur + Adaptive Uptake",
            richness=richness(N_eq),
            shannon=shannon(N_eq),
            adaptation_rate=adaptation_rate,
            adaptation_damping=adaptation_damping,
        ))

    # ---------------- analysis & plotting ----------------
    df = pd.DataFrame(records)
    df.to_csv("coexistence_named_params.csv", index=False)
    print(df.groupby("model")[["richness", "shannon"]].mean())

    models = df["model"].unique()

    # boxplots (fix deprecation: use tick_labels)
    plt.figure(figsize=(6.5, 4))
    plt.boxplot([df[df.model == m]["richness"] for m in models],
                tick_labels=models)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Richness (# spp > 1e-4)")
    plt.title("Richness per model")
    plt.tight_layout()
    plt.savefig("coexist_named_richness_box.png", dpi=300)

    plt.figure(figsize=(6.5, 4))
    plt.boxplot([df[df.model == m]["shannon"] for m in models],
                tick_labels=models)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Shannon index")
    plt.title("Shannon per model")
    plt.tight_layout()
    plt.savefig("coexist_named_shannon_box.png", dpi=300)

    # parameter histograms: ONLY plot non-NaN columns
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(models):
        sub = df[df.model == m]
        plt.subplot(2, 2, i + 1)
        # columns that are not metrics and have at least one non-NaN
        cols = []
        for c in sub.columns:
            if c in ("model", "richness", "shannon"):
                continue
            if sub[c].notna().any():
                cols.append(c)
        for c in cols:
            plt.hist(sub[c].dropna(), bins=10, alpha=0.6, label=c)
        plt.title(m, fontsize=8)
        if cols:
            plt.legend(fontsize=6)
        plt.xlabel("sampled value")
        plt.ylabel("count")
    plt.suptitle("Distributions of sampled parameters per model", y=1.02)
    plt.tight_layout()
    plt.savefig("coexist_named_param_hists.png", dpi=300)

    # mean bar
    means = df.groupby("model")[["richness", "shannon"]].mean().reset_index()
    x = np.arange(len(means))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, means["richness"], width=0.4, label="Mean richness")
    plt.bar(x + 0.2, means["shannon"], width=0.4, label="Mean Shannon")
    plt.xticks(x, means["model"], rotation=20, ha="right")
    plt.ylabel("Value")
    plt.title("Average coexistence metrics per model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("coexist_named_mean_bar.png", dpi=300)

    print("✅ done: figures + CSV written.")

if __name__ == "__main__":
    main()