#!/usr/bin/env python3
# quick_benchmark_5models.py
#
# One-file benchmarking for FIVE models:
#   - mcrm (MacArthur-like with resource matrix W, thresholds T)
#   - micrm (MiCRM: leakage, byproducts, separate w,m,g)
#   - gibbs (min-rate growth with cross-feeding tensors)
#   - adaptive (Picciani–Mori style adaptive allocations)
#   - classical (MacArthur classical with resource modes)
#
# Data: single CSV with columns:
#   Eliminated, Time, BH,CA,BU,PC,BO,BV,BT,EL,FP,CH,DP,ER
#
# Example:
#   python quick_benchmark_5models.py \
#     --data_csv "/Users/edwin/Documents/Ventrulli_dataset.csv" \
#     --models all \
#     --fit --train_group "NONE" \
#     --reset_resources \
#     --save_csv "results_metrics.csv" \
#     --save_preds "results_predictions.csv" \
#     --save_plots_dir "plots"
#
# You can pass per-model params with --params_json, containing a dict keyed by model name.
# Any missing model key will fall back to built-in defaults.

import argparse, json, os, random
from typing import Dict, Tuple, Optional, List, Callable
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------- Species & data --------------------------
SPECIES = ['BH','CA','BU','PC','BO','BV','BT','EL','FP','CH','DP','ER']

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Eliminated" not in df.columns:
        raise ValueError("CSV must have 'Eliminated'.")
    tcol = next((c for c in df.columns if c.lower() in ("time","t","hours","hour")), None)
    if not tcol: raise ValueError("CSV must have a time column (Time/t/hours/hour).")
    if tcol != "Time": df = df.rename(columns={tcol:"Time"})
    for sp in SPECIES:
        if sp not in df.columns: df[sp] = 0.0
    X = df[SPECIES].to_numpy(float)
    rs = X.sum(axis=1, keepdims=True); rs[rs==0] = 1.0
    df[SPECIES] = X/rs
    return df.sort_values(["Eliminated","Time"]).reset_index(drop=True)

def split_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {str(k): g[["Time"]+SPECIES].reset_index(drop=True)
            for k,g in df.groupby("Eliminated")}

# -------------------------- Metrics (3) --------------------------
def metric_pearson(P: np.ndarray, O: np.ndarray) -> float:
    vals = []
    for j in range(P.shape[1]):
        if np.std(O[:,j])<1e-12 or np.std(P[:,j])<1e-12: continue
        vals.append(pearsonr(P[:,j], O[:,j])[0])
    return float(np.nanmean(vals)) if vals else float("nan")

def metric_rmse(P: np.ndarray, O: np.ndarray) -> float:
    return float(np.sqrt(np.mean((P-O)**2)))

def _clr(V: np.ndarray, eps=1e-9) -> np.ndarray:
    V = np.clip(V, eps, None)
    g = np.exp(np.mean(np.log(V), axis=1, keepdims=True))
    return np.log(V/g)

def metric_aitchison(P: np.ndarray, O: np.ndarray) -> float:
    Zp, Zo = _clr(P), _clr(O)
    return float(np.sqrt(np.mean((Zp-Zo)**2)))

# -------------------------- Utils --------------------------
def to_rel(N: np.ndarray) -> np.ndarray:
    s = N.sum(axis=1, keepdims=True) + 1e-12
    return N/s

def nearest_indices(t_grid: np.ndarray, t_samples: np.ndarray) -> np.ndarray:
    return np.array([np.argmin(np.abs(t_grid - ti)) for ti in t_samples], dtype=int)

def seed_everything(seed: int = 0):
    random.seed(seed); np.random.seed(seed)

def simulate_with_transfers(rhs: Callable, y0: np.ndarray, params: dict, t_samples: np.ndarray,
                            transfer_every=24.0, dilution=20.0,
                            reset_resources=False, R_reset=None,
                            atol=1e-8, rtol=1e-6):
    t0, tf = float(t_samples[0]), float(t_samples[-1])
    boundaries = np.arange(t0+transfer_every, tf+1e-9, transfer_every)
    t_all=[t0]; y_all=[y0.copy()]; y=y0.copy(); last=t0
    S = int(params.get("S", len(SPECIES)))
    for b in list(boundaries)+[tf]:
        sol = solve_ivp(lambda t,z: rhs(t,z,params), [last,b], y, method="BDF",
                        atol=atol, rtol=rtol, dense_output=False)
        for k in range(1, sol.y.shape[1]):
            t_all.append(sol.t[k]); y_all.append(sol.y[:,k].copy())
        y = sol.y[:,-1].copy()
        if b < tf:
            y[:S] = y[:S] / float(dilution)
            if reset_resources:
                if R_reset is None: raise ValueError("reset_resources=True but no R_reset.")
                y[S:S+len(R_reset)] = np.asarray(R_reset, float)
            t_all.append(b); y_all.append(y.copy())
        last=b
    return np.array(t_all), np.vstack(y_all)

# -------------------------- Default Params per model --------------------------
DEFAULTS = {
    "mcrm": {
        "varIdx":{"species": list(range(12)), "resources":[12,13,14]},
        "C": [[0.1,0.1,0.1]]*12,
        "D": [[0,0,0],[0,0,0],[0,0,0]],
        "B": 0.0,
        "T": [0.2]*12,
        "alpha": [1,1,1],
        "tau": [10,10,10],
        "death_rate": [0.01]*12,
        "mu": 1.0,
        "W": [[1,0,0],[0,1,0],[0,0,1]],
        "R0": [1,1,1],
        "_fit_enable": ["C","T","alpha","tau","death","mu"]
    },
    "micrm": {
        "num_resources": 3,
        "C": [[0.1,0.1,0.1]]*12,  # uptake per species x resource
        "D": [[0,0,0],[0,0,0],[0,0,0]],
        "leakage": 0.2,           # scalar or len-R
        "rho": [1,1,1],           # inflow
        "tau": [10,10,10],        # decay times
        "w": [1.0]*12,            # biomass yield weights
        "m": [0.2]*12,            # maintenance
        "g": 1.0,                 # global factor
        "R0": [1,1,1],
        "_fit_enable": ["C","m","rho","leakage"]  # simple set
    },
    "gibbs": {
        "num_resources": 3,    # R
        "C": [[0.1]*12]*3,     # (R,S) which resource contributes to each species
        "epsilon": [[1.0]*12]*3,
        "P": [[0,0,0],[0,0,0],[0,0,0]],
        "Pt": [[0,0,0],[0,0,0],[0,0,0]],
        "rho": [1,1,1],
        "theta": 0.2,
        "eta": 0.1,
        "R0": [1,1,1],
        "_fit_enable": ["C","epsilon","rho","theta","eta"]
    },
    "adaptive": {
        "S": 12, "R": 3,
        "v": [1.0,1.0,1.0],
        "K": [1.0,1.0,1.0],
        "d": [0.2]*12,
        "s": [1.0,1.0,1.0],
        "mu": [0.1,0.1,0.1],
        "lam": [0.5]*12,
        "E_star": [1.0]*12,
        "C0": [1.0,1.0,1.0],
        "A0": [[1/3,1/3,1/3]]*12,
        "_fit_enable": ["v","K","d","s","mu","lam"]
    },
    "classical": {
        "tau": 1.0,
        "m": [0.5]*12,           # (S,)
        "w": [1.0]*1,            # (R,)
        "c": [[0.6]*1]*12,       # (S,R)
        "r": [1.0]*1,            # (R,)
        "K": [1.0]*1,            # or 'kappa'
        "resource_mode": "external",  # logistic|external|tilman
        "R0": [1.0],
        "_fit_enable": ["c","m","r","K"]  # scalar scales per group
    }
}

# -------------------------- Model RHS --------------------------
def rhs_mcrm(t, x, p):
    v=p['varIdx']
    C=np.asarray(p['C'],float); D=np.asarray(p['D'],float)
    T=np.asarray(p['T'],float); alpha=np.asarray(p['alpha'],float)
    tau=np.asarray(p['tau'],float); death=np.asarray(p['death_rate'],float)
    mu=float(p['mu']); W=np.asarray(p['W'],float); B=p.get('B',0.0)
    N=x[v['species']]; R=x[v['resources']]
    dx=np.zeros_like(x)
    growth = C @ (W @ R) - T
    dx[v['species']] = N*mu*growth - death*N
    consumption = (C * R).T @ N
    release = D @ consumption
    bleed = (B * np.sum(death*N)) if np.isscalar(B) else (np.asarray(B)*np.sum(death*N))
    dx[v['resources']] = (alpha - R)/tau - consumption + release + bleed
    return dx

def rhs_micrm(t, x, p):
    Rn=int(p['num_resources']); R=x[:Rn]; N=x[Rn:]
    C=np.asarray(p['C'],float)           # (S,R) NOTE: we transpose usage below
    D=np.asarray(p['D'],float)
    l=p['leakage']; leak = (np.full(Rn,l) if np.isscalar(l) else np.asarray(l,float))
    rho=np.asarray(p['rho'],float); tau=np.asarray(p['tau'],float)
    w=np.asarray(p['w'],float); m=np.asarray(p['m'],float); g=float(p['g'])
    # uptake per species-resource: C * R
    uptake = C * R[None,:]                 # (S,R)
    net = np.sum(uptake*(1.0 - leak)[None,:], axis=1)   # (S,)
    dN = g * N * (w * net - m)
    # resource loss & release
    consumption = np.sum(C * (N[:,None]*R[None,:]), axis=0)
    release_terms = C * (N[:,None]*R[None,:])
    scaled_leak = (w[:,None] * leak[None,:]) * release_terms
    total_release = np.sum(scaled_leak, axis=0)
    release = D @ total_release
    dR = rho - R/tau - consumption + release
    return np.concatenate([dR, dN])

def rhs_gibbs(t, y, p):
    S = int(p['num_resources'])
    R = y[:S]; N = y[S:]
    C = np.asarray(p['C'],float)          # (R,S)
    epsilon = np.asarray(p['epsilon'],float) # (R,S)
    P = np.asarray(p['P'],float)          # (R,R)
    Pt = np.asarray(p['Pt'],float)        # (R,R)
    rho = np.asarray(p['rho'],float)      # (R,)
    theta = float(p['theta']); eta=float(p['eta'])
    # min-rate growth
    g = np.zeros_like(N)
    for k in range(len(N)):
        rates = []
        for j in range(S):
            if C[j, k] != 0:
                rates.append(epsilon[j, k]*C[j, k]*R[j])
        g[k] = min(rates) if len(rates)>0 else 0.0
    dR = np.zeros(S)
    for i in range(S):
        term1 = rho[i]
        term2 = R[i] * np.sum(C[i, :] * N)
        term3 = 0.0
        for j in range(S):
            inner = 0.0
            for k in range(len(N)):
                inner += (C[j,k]*R[j] - (g[k]/(epsilon[j,k] if epsilon[j,k]!=0 else 1.0))) * N[k]
            term3 += P[i, j] * inner
        term4 = 0.0
        for j in range(S):
            inner = 0.0
            for k in range(len(N)):
                denom = (epsilon[j,k] if epsilon[j,k]!=0 else 1.0)
                inner += (g[k]/denom) * N[k]
            term4 += theta * Pt[i, j] * inner
        dR[i] = term1 - term2 + term3 + term4
    dN = N * ((1 - theta) * g - eta)
    return np.concatenate([dR, dN])

def rhs_adaptive(t, x, p):
    S, R = int(p['S']), int(p['R'])
    N = x[:S]; C = x[S:S+R]; A = x[S+R:].reshape(S, R)
    v = np.asarray(p['v'],float); K=np.asarray(p['K'],float)
    d = np.asarray(p['d'],float); s=np.asarray(p['s'],float)
    mu = np.asarray(p['mu'],float); lam=np.asarray(p['lam'],float)
    E_star = np.asarray(p['E_star'],float)
    # Monod
    r = C / (K + C + 1e-12)
    growth = (A * (v[None,:] * r[None,:])).sum(axis=1)
    dN = N * (growth - d)
    cons = (N[:,None] * A * r[None,:]).sum(axis=0)
    # optional B tensor omitted for simplicity
    dC = s - cons - mu * C
    budget = A.sum(axis=1)
    active = (budget >= E_star).astype(float)
    penalty = active * (budget / np.maximum(E_star, 1e-12)) * growth
    dA = A * (lam[:,None] * (v[None,:] * r[None,:]) - penalty[:,None])
    return np.concatenate([dN, dC, dA.reshape(-1)])

def rhs_classical(t, y, p):
    tau=float(p["tau"]); m=np.asarray(p["m"],float)
    w=np.asarray(p["w"],float); c=np.asarray(p["c"],float)
    r=np.asarray(p["r"],float); mode=p.get("resource_mode","external")
    if "K" in p: Kk=np.asarray(p["K"],float)
    else:        Kk=np.asarray(p["kappa"],float)
    S,R = c.shape
    N=y[:S]; Rv=y[S:]
    growth_input = (c * w.reshape(1,-1)) @ Rv if R>0 else 0.0
    dN = (N/tau) * (growth_input - m)
    if R>0:
        if mode in ("logistic","external"):
            consumption = (N @ c) * (Rv if mode!="tilman" else 1.0)
        else:
            consumption = (N @ c)
        regen = (r/Kk) * (Kk - Rv) * Rv if mode=="logistic" else r*(Kk - Rv)
        dR = regen - consumption
        return np.concatenate([dN, dR])
    return dN

# -------------------------- Adapter builders & fit hooks --------------------------
def build_adapter(model: str, first_row: pd.Series, B0: float, params: dict):
    """Return: rhs, y0, P, extract_fn, theta_pack, theta_unpack, enabled_groups, state_info(optional)"""
    if model == "mcrm":
        p = params.copy(); v=p['varIdx']
        # state len
        def _end(idx):
            if hasattr(idx,"stop"): return idx.stop
            idx=np.asarray(idx); return int(idx.max()+1)
        nstate=max(_end(v['species']), _end(v['resources']))
        x0=np.zeros(nstate,float)
        x0[v['species']] = first_row[SPECIES].to_numpy(float)*B0
        R0 = np.asarray(p.get('R0', np.ones(_end(v['resources'])-(v['resources'].start if hasattr(v['resources'],'start') else 0))), float)
        x0[v['resources']] = R0
        P={"S":len(SPECIES)}
        def rhs(t,x,_): return rhs_mcrm(t,x,p)
        def extract(Y): return Y[:, v['species']]
        base = {
            "C":np.asarray(p["C"],float).copy(),
            "T":np.asarray(p["T"],float).copy(),
            "alpha":np.asarray(p["alpha"],float).copy(),
            "tau":np.asarray(p["tau"],float).copy(),
            "death":np.asarray(p["death_rate"],float).copy(),
            "mu":float(p["mu"]),
        }
        enabled = p.get("_fit_enable", ["C","T","alpha","tau","death","mu"])
        order = [g for g in ["C","T","alpha","tau","death","mu"] if g in enabled]
        def tpack(): return np.ones(len(order), float)
        def tunpack(th):
            th=np.asarray(th,float); assert th.size==len(order)
            p["C"]=base["C"].copy(); p["T"]=base["T"].copy()
            p["alpha"]=base["alpha"].copy(); p["tau"]=base["tau"].copy()
            p["death_rate"]=base["death"].copy(); p["mu"]=float(base["mu"])
            for k,g in enumerate(order):
                s=float(th[k])
                if   g=="C": p["C"]*=s
                elif g=="T": p["T"]*=s
                elif g=="alpha": p["alpha"]*=s
                elif g=="tau": p["tau"]*=s
                elif g=="death": p["death_rate"]*=s
                elif g=="mu": p["mu"]*=s
        return rhs, x0, P, extract, tpack, tunpack, order, None

    if model == "micrm":
        p = params.copy()
        Rn=int(p['num_resources'])
        R0=np.asarray(p.get('R0', np.ones(Rn)), float)
        N0 = first_row[SPECIES].to_numpy(float)*B0
        x0 = np.concatenate([R0, N0])
        P={"S":len(SPECIES)}
        def rhs(t,x,_): return rhs_micrm(t,x,p)
        def extract(Y): return Y[:, Rn:Rn+len(SPECIES)]
        base={
            "C":np.asarray(p["C"],float).copy(),
            "m":np.asarray(p["m"],float).copy(),
            "rho":np.asarray(p["rho"],float).copy(),
            "leakage": (p["leakage"] if np.isscalar(p["leakage"]) else np.asarray(p["leakage"],float).copy())
        }
        enabled = p.get("_fit_enable", ["C","m","rho","leakage"])
        order = [g for g in ["C","m","rho","leakage"] if g in enabled]
        def tpack(): return np.ones(len(order), float)
        def tunpack(th):
            th=np.asarray(th,float); assert th.size==len(order)
            p["C"]=base["C"].copy(); p["m"]=base["m"].copy(); p["rho"]=base["rho"].copy()
            p["leakage"]= (base["leakage"] if np.isscalar(base["leakage"]) else base["leakage"].copy())
            for k,g in enumerate(order):
                s=float(th[k])
                if   g=="C": p["C"]*=s
                elif g=="m": p["m"]*=s
                elif g=="rho": p["rho"]*=s
                elif g=="leakage":
                    if np.isscalar(p["leakage"]): p["leakage"]=p["leakage"]*s
                    else: p["leakage"]=np.asarray(p["leakage"])*s
        return rhs, x0, P, extract, tpack, tunpack, order, None

    if model == "gibbs":
        p = params.copy()
        S = int(p['num_resources'])
        R0=np.asarray(p.get('R0', np.ones(S)), float)
        N0 = first_row[SPECIES].to_numpy(float)*B0
        y0 = np.concatenate([R0, N0])
        P={"S":len(SPECIES)}
        def rhs(t,y,_): return rhs_gibbs(t,y,p)
        def extract(Y): return Y[:, S:S+len(SPECIES)]
        base={
            "C":np.asarray(p["C"],float).copy(),
            "epsilon":np.asarray(p["epsilon"],float).copy(),
            "rho":np.asarray(p["rho"],float).copy(),
            "theta":float(p["theta"]),
            "eta":float(p["eta"])
        }
        enabled = p.get("_fit_enable", ["C","epsilon","rho","theta","eta"])
        order = [g for g in ["C","epsilon","rho","theta","eta"] if g in enabled]
        def tpack(): return np.ones(len(order), float)
        def tunpack(th):
            th=np.asarray(th,float); assert th.size==len(order)
            p["C"]=base["C"].copy(); p["epsilon"]=base["epsilon"].copy()
            p["rho"]=base["rho"].copy(); p["theta"]=float(base["theta"]); p["eta"]=float(base["eta"])
            for k,g in enumerate(order):
                s=float(th[k])
                if   g=="C": p["C"]*=s
                elif g=="epsilon": p["epsilon"]*=s
                elif g=="rho": p["rho"]*=s
                elif g=="theta": p["theta"]*=s
                elif g=="eta": p["eta"]*=s
        return rhs, y0, P, extract, tpack, tunpack, order, None

    if model == "adaptive":
        p = params.copy()
        S,R=int(p["S"]), int(p["R"])
        N0 = np.zeros(S); N0[:len(SPECIES)] = first_row[SPECIES].to_numpy(float)*B0
        C0 = np.asarray(p.get("C0", np.ones(R)), float)
        A0 = np.asarray(p.get("A0", np.ones((S,R))/R), float).reshape(-1)
        x0 = np.concatenate([N0, C0, A0])
        P={"S":S}
        def rhs(t,x,_): return rhs_adaptive(t,x,p)
        def extract(Y): return Y[:, :S]
        base = {
            "v": np.asarray(p["v"],float).copy(),
            "K": np.asarray(p["K"],float).copy(),
            "d": np.asarray(p["d"],float).copy(),
            "s": np.asarray(p["s"],float).copy(),
            "mu": np.asarray(p["mu"],float).copy(),
            "lam": np.asarray(p["lam"],float).copy(),
        }
        enabled = p.get("_fit_enable", ["v","K","d","s","mu","lam"])
        order = [g for g in ["v","K","d","s","mu","lam"] if g in enabled]
        def tpack(): return np.ones(len(order), float)
        def tunpack(th):
            th=np.asarray(th,float); assert th.size==len(order)
            p["v"]=base["v"].copy(); p["K"]=base["K"].copy()
            p["d"]=base["d"].copy(); p["s"]=base["s"].copy()
            p["mu"]=base["mu"].copy(); p["lam"]=base["lam"].copy()
            for k,g in enumerate(order):
                s=float(th[k])
                p[g] = np.asarray(p[g],float)*s
        return rhs, x0, P, extract, tpack, tunpack, order, None

    if model == "classical":
        p = params.copy()
        c=np.asarray(p["c"],float); S,R=c.shape
        N0 = np.zeros(S); N0[:len(SPECIES)] = first_row[SPECIES].to_numpy(float)*B0
        R0 = np.asarray(p.get("R0", np.ones(R)), float)
        y0 = np.concatenate([N0, R0])
        P={"S":S}
        def rhs(t,y,_): return rhs_classical(t,y,p)
        def extract(Y): return Y[:, :S]
        base={
            "c":np.asarray(p["c"],float).copy(),
            "m":np.asarray(p["m"],float).copy(),
            "r":np.asarray(p["r"],float).copy(),
            "K":np.asarray(p["K"],float).copy() if "K" in p else np.asarray(p["kappa"],float).copy()
        }
        enabled = p.get("_fit_enable", ["c","m","r","K"])
        order = [g for g in ["c","m","r","K"] if g in enabled]
        def tpack(): return np.ones(len(order), float)
        def tunpack(th):
            th=np.asarray(th,float); assert th.size==len(order)
            for key in base: p[key]=base[key].copy()
            for k,g in enumerate(order):
                s=float(th[k]); p[g] = np.asarray(p[g],float)*s
        return rhs, y0, P, extract, tpack, tunpack, order, None

    raise ValueError(f"Unknown model: {model}")

# -------------------------- Fitting --------------------------
def fit_DE(rhs, x0, P, df_train, extract, theta_pack, theta_unpack, order: List[str],
           transfer_h=24.0, dilution=20.0,
           reset_resources=False, R_reset=None,
           bounds_low=0.1, bounds_high=3.0, reg_lambda=0.0,
           maxiter=60, popsize=15, seed=0):
    if len(order)==0:
        # nothing to fit
        t=df_train["Time"].to_numpy(float)
        Vobs=df_train[SPECIES].to_numpy(float)
        t_grid,Y = simulate_with_transfers(rhs, x0, P, t,
                                           transfer_every=transfer_h, dilution=dilution,
                                           reset_resources=reset_resources, R_reset=R_reset)
        Vpred = to_rel(extract(Y))[nearest_indices(t_grid, t), :]
        return np.array([]), metric_rmse(Vpred, Vobs)

    seed_everything(seed)
    t = df_train["Time"].to_numpy(float)
    Vobs = df_train[SPECIES].to_numpy(float)

    def objective(th):
        theta_unpack(th)
        t_grid, Y = simulate_with_transfers(rhs, x0, P, t,
                                            transfer_every=transfer_h, dilution=dilution,
                                            reset_resources=reset_resources, R_reset=R_reset)
        Vpred = to_rel(extract(Y))[nearest_indices(t_grid, t), :]
        loss = metric_rmse(Vpred, Vobs)
        if reg_lambda>0:
            loss += reg_lambda * float(np.sum((np.asarray(th)-1.0)**2))
        return loss

    bounds = [(bounds_low, bounds_high)] * len(order)
    res = differential_evolution(objective, bounds=bounds, maxiter=maxiter,
                                 popsize=popsize, seed=seed, polish=True)
    best = res.x; best_loss = float(res.fun)
    theta_unpack(best)  # lock in best
    return best, best_loss

# -------------------------- Plot (legend outside) --------------------------
def save_stacked_bar_pred_vs_obs(outdir: str, community: str, model: str,
                                 times: np.ndarray,
                                 Vobs: np.ndarray, Vpred: np.ndarray):
    os.makedirs(outdir, exist_ok=True)
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 5 + 0.12*len(times)))
    bottom_obs = np.zeros(len(times))
    bottom_pred = np.zeros(len(times))
    x_obs  = np.arange(len(times)) - width/2
    x_pred = np.arange(len(times)) + width/2
    for j, sp in enumerate(SPECIES):
        ax.bar(x_obs,  Vobs[:, j], width, bottom=bottom_obs,  edgecolor='none', label=sp)
        ax.bar(x_pred, Vpred[:, j], width, bottom=bottom_pred, edgecolor='none')
        bottom_obs  += Vobs[:, j]
        bottom_pred += Vpred[:, j]
    ax.set_xticks(np.arange(len(times)))
    ax.set_xticklabels([f"{t:g}h" for t in times], rotation=0)
    ax.set_ylabel("Relative abundance")
    ax.set_title(f"{community} — {model.upper()}: Observed (left) vs Predicted (right)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    # Place legend outside (right side), prevent overlap
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, frameon=False)
    fig.tight_layout(rect=[0,0,0.85,1])  # leave room for legend on the right
    out_path = os.path.join(outdir, f"{community}__{model}_stacked_bars.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True, type=str)
    ap.add_argument("--models", type=str, default="mcrm",
                    help="Comma-separated from {mcrm,micrm,gibbs,adaptive,classical} or 'all'")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--train_group", type=str, default="NONE")
    ap.add_argument("--b0", type=float, default=1.0)
    ap.add_argument("--transfer_hours", type=float, default=24.0)
    ap.add_argument("--dilution", type=float, default=20.0)
    ap.add_argument("--reset_resources", action="store_true")
    ap.add_argument("--params_json", type=str, default="",
                    help="JSON of per-model dict, e.g. {'mcrm': {...}, 'micrm': {...}}")
    ap.add_argument("--fit_bounds_low", type=float, default=0.1)
    ap.add_argument("--fit_bounds_high", type=float, default=3.0)
    ap.add_argument("--fit_reg_lambda", type=float, default=0.0)
    ap.add_argument("--fit_maxiter", type=int, default=60)
    ap.add_argument("--fit_popsize", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_csv", type=str, default=None)
    ap.add_argument("--save_preds", type=str, default=None)
    ap.add_argument("--save_plots_dir", type=str, default=None)
    args = ap.parse_args()

    seed_everything(args.seed)

    df = load_csv(args.data_csv)
    groups = split_groups(df)
    if args.train_group not in groups:
        raise RuntimeError(f"train_group '{args.train_group}' not found. Available: {list(groups)}")

    # Resolve models
    all_names = ["mcrm","micrm","gibbs","adaptive","classical"]
    models = all_names if args.models.strip().lower()=="all" \
             else [m.strip().lower() for m in args.models.split(",")]
    for m in models:
        if m not in all_names: raise ValueError(f"Unknown model '{m}'")

    # Load per-model params (start from defaults)
    user = json.loads(args.params_json) if args.params_json else {}
    params_by_model = {}
    for m in models:
        base = json.loads(json.dumps(DEFAULTS[m]))  # deep-ish copy via JSON
        if m in user:
            # shallow-merge keys from user[m]
            for k,v in user[m].items():
                base[k] = v
        params_by_model[m] = base

    metrics_rows = []
    preds_rows = []

    # Loop models independently (fit per model if requested)
    for m in models:
        print(f"\n=== Model: {m.upper()} ===")
        params = params_by_model[m]
        # Build adapter on the training group's first row
        train_df = groups[args.train_group]
        rhs, x0, P, extract, tpack, tunpack, order, _ = build_adapter(m, train_df.iloc[0], args.b0, params)

        best_theta = None
        if args.fit:
            best_theta, best_loss = fit_DE(
                rhs, x0, P, train_df, extract, tpack, tunpack, order,
                transfer_h=args.transfer_hours, dilution=args.dilution,
                reset_resources=args.reset_resources, R_reset=params.get("R0", None),
                bounds_low=args.fit_bounds_low, bounds_high=args.fit_bounds_high,
                reg_lambda=args.fit_reg_lambda,
                maxiter=args.fit_maxiter, popsize=args.fit_popsize, seed=args.seed
            )
            print(f"[{m}] fit groups={order}  theta={best_theta}  rmse={best_loss:.6f}")

        # Evaluate all communities for this model
        for label, g in groups.items():
            rhs_g, x0_g, P_g, extract_g, tpack_g, tunpack_g, order_g, _ = build_adapter(
                m, g.iloc[0], args.b0, params
            )
            if args.fit and best_theta is not None and len(order_g)>0:
                tunpack_g(best_theta)
            t = g["Time"].to_numpy(float)
            Vobs = g[SPECIES].to_numpy(float)
            t_grid, Y = simulate_with_transfers(
                rhs_g, x0_g, P_g, t,
                transfer_every=args.transfer_hours, dilution=args.dilution,
                reset_resources=args.reset_resources, R_reset=params.get("R0", None)
            )
            Vpred = to_rel(extract_g(Y))[nearest_indices(t_grid, t), :]
            # metrics (3)
            row = {
                "model": m,
                "community": label,
                "pearson_r": metric_pearson(Vpred, Vobs),
                "rmse": metric_rmse(Vpred, Vobs),
                "aitchison": metric_aitchison(Vpred, Vobs)
            }
            metrics_rows.append(row)
            # preds
            if args.save_preds:
                for i, ti in enumerate(t):
                    rec = {"model": m, "community": label, "Time": float(ti)}
                    for j, sp in enumerate(SPECIES):
                        rec[f"pred_{sp}"] = float(Vpred[i, j])
                        rec[f"obs_{sp}"]  = float(Vobs[i, j])
                    preds_rows.append(rec)
            # plots
            if args.save_plots_dir:
                save_stacked_bar_pred_vs_obs(args.save_plots_dir, label, m, t, Vobs, Vpred)

    # Save outputs
    out = pd.DataFrame(metrics_rows).sort_values(["model","community"]).reset_index(drop=True)
    print("\n----- METRICS -----")
    print(out.to_string(index=False))
    if args.save_csv: out.to_csv(args.save_csv, index=False)
    if args.save_preds and preds_rows:
        pd.DataFrame(preds_rows).to_csv(args.save_preds, index=False)

if __name__ == "__main__":
    main()