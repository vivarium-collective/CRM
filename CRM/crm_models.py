import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import entropy
from numpy.linalg import eigvals
from typing import Dict, Tuple, Callable
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# CommunityModels: MCRM, Gibbs, MiCRM, Adaptive, Classical
# =========================================================
class CommunityModels:
    def __init__(self,
                 mcrm_params=None,
                 gibbs_params=None,
                 micrm_params=None,
                 adaptive_params=None,
                 classical_params=None):
        """
        Provide zero or more parameter dicts. Any provided model will be run by run_all().
        For the classical MacArthur CRM, pass classical_params.
        """
        self.params_mcrm = mcrm_params
        self.params_gibbs = gibbs_params
        self.params_micrm = micrm_params
        self.params_adaptive = adaptive_params
        self.params_classical = classical_params
        self.results = {}

    # --------------------- MCRM ---------------------
    def _mcrm_dynamics(self, t, x):
        p = self.params_mcrm
        v = p['varIdx']
        C, D = p['C'], p['D']
        B, T, alpha, tau = p['B'], p['T'], p['alpha'], p['tau']
        death_rate = p['death_rate']
        mu = p['mu']
        W = p['W']

        N = x[v['species']]
        R = x[v['resources']]
        dx = np.zeros_like(x)

        # shared/species W handling
        if p.get('W_mode', 'shared') == 'species':
            growth_term = np.sum(C * W * R, axis=1) - T
        else:
            growth_term = C @ (W @ R) - T

        dx[v['species']] = N * mu * growth_term - death_rate * N
        consumption = (C * R).T @ N
        production = D @ consumption
        dx[v['resources']] = (alpha - R) / tau - consumption + production + B * (death_rate @ N)
        return dx

    # --------------------- Gibbs ---------------------
    def _gibbs_dynamics(self, t, y):
        p = self.params_gibbs
        S = p['num_resources']
        R = y[:S]
        N = y[S:]
        C, epsilon, P, Pt = p['C'], p['epsilon'], p['P'], p['Pt']
        rho, theta, eta = p['rho'], p['theta'], p['eta']

        g = np.array([
            min([epsilon[j, k] * C[j, k] * R[j] for j in range(S) if C[j, k] != 0])
            if any(C[:, k] != 0) else 0.0 for k in range(len(N))
        ])

        dR = np.zeros(S)
        for i in range(S):
            term1 = rho[i]
            term2 = R[i] * np.sum(C[i, :] * N)
            term3 = sum(P[i, j] * sum((C[j, k] * R[j] - g[k] / epsilon[j, k]) * N[k]
                        for k in range(len(N))) for j in range(S))
            term4 = theta * sum(Pt[i, j] * sum((g[k] / epsilon[j, k]) * N[k]
                        for k in range(len(N))) for j in range(S))
            dR[i] = term1 - term2 + term3 + term4

        dN = N * ((1 - theta) * g - eta)
        return np.concatenate([dR, dN])

    # --------------------- MiCRM ---------------------
    def _micrm_dynamics(self, t, x):
        p = self.params_micrm
        R = x[:p['num_resources']]   # R_α
        N = x[p['num_resources']:]   # N_i

        C = p['C']                   # c_{iα}
        D = p['D']                   # D_{αβ}
        l = p['leakage']             # ℓ_α (scalar or array)
        rho = p['rho']               # κ_α
        tau = p['tau']               # τ_α
        w = p['w']                   # w_i
        m = p['m']                   # m_i
        g = p['g']                   # scalar

        leak = np.array(l) if isinstance(l, (list, np.ndarray)) else np.full(R.shape, l)

        resource_uptake = C * R                      # (species, resources)
        one_minus_leak = (1 - leak)
        net_uptake = np.sum(resource_uptake * one_minus_leak, axis=1)
        dN = g * N * (w * net_uptake - m)

        consumption = np.sum(C * (N[:, None] * R[None, :]), axis=0)

        release_terms = C * (N[:, None] * R[None, :])
        scaled_leak = (w[:, None] * leak[None, :]) * release_terms
        total_release = scaled_leak.sum(axis=0)
        release = D @ total_release

        dR = rho - R / tau - consumption + release
        return np.concatenate([dR, dN])

    # --------------------- Adaptive CRM (Picciani–Mori) ---------------------
    def adaptive_crm_dynamics(self, t, x):
        """
        State: [N (S,), C (R,), A (S*R,)]
        params_adaptive keys:
            S,R; v(R,), K(R,), d(S,), s(R,), mu(R,), lam(S,), E_star(S,),
            N0(S,), C0(R,), A0(S,R), optional B(R,S,R), nonneg_clip(bool)
        """
        p = self.params_adaptive
        S, R = p['S'], p['R']

        N = x[:S]
        C = x[S:S+R]
        A = x[S+R:].reshape(S, R)

        v = np.asarray(p['v'], float)
        K = np.asarray(p['K'], float)
        d = np.asarray(p['d'], float)
        s = np.asarray(p['s'], float)
        mu = np.asarray(p['mu'], float)
        lam = np.asarray(p['lam'], float)
        E_star = np.asarray(p['E_star'], float)
        B = np.asarray(p.get('B', np.zeros((R, S, R))), float)
        nonneg = bool(p.get('nonneg_clip', True))

        r = C / (K + C + 1e-12)                         # (R,)

        growth = (A * (v[None, :] * r[None, :])).sum(axis=1)   # (S,)
        dN = N * (growth - d)

        cons = (N[:, None] * A * r[None, :]).sum(axis=0)       # (R,)
        prod = np.tensordot(B, (N[:, None] * A * r[None, :]), axes=([1, 2], [0, 1]))
        dC = s - cons + prod - mu * C

        budget = A.sum(axis=1)                                  # (S,)
        gsum = growth
        active = (budget >= E_star).astype(float)
        penalty = active * (budget / np.maximum(E_star, 1e-12)) * gsum
        dA = A * (lam[:, None] * (v[None, :] * r[None, :]) - penalty[:, None])

        dx = np.concatenate([dN, dC, dA.reshape(-1)])

        if nonneg:
            s_end = S
            r_end = S + R
            dx[:S] = np.where((N <= 0) & (dN < 0), 0.0, dx[:S])
            dx[s_end:r_end] = np.where((C <= 0) & (dC < 0), 0.0, dx[s_end:r_end])
            flatA = A.reshape(-1); flatdA = dA.reshape(-1)
            dx[r_end:] = np.where((flatA <= 0) & (flatdA < 0), 0.0, flatdA)

        return dx

    # --------------------- Classical MacArthur CRM (and variants) ---------------------
    def classical_crm_dynamics(self, params, num_species=None, num_resources=None,
                               timesteps=10000, dt=0.01,
                               initial_N=None, initial_R=None,
                               resource_mode='logistic'):
        """
        Classical MacArthur CRM + two variants (no plotting, no scenario logic).

        params keys:
          tau : float
          m   : (S,)   maintenance/death rates
          w   : (R,)   resource values
          c   : (S,R)  uptake coefficients
          r   : (R,)   resource regeneration/supply rate
          K   : (R,)   carrying capacities (logistic/external)
          OR
          kappa : (R,) target supply levels (alias for K)

        resource_mode:
          'logistic' : logistic resource growth; consumption ∝ R
          'external' : supply towards K;    consumption ∝ R
          'tilman'   : supply towards K;    consumption independent of R
        """
        if resource_mode not in ('logistic', 'external', 'tilman'):
            raise ValueError("resource_mode must be 'logistic', 'external', or 'tilman'.")

        tau = float(params["tau"])
        m   = np.asarray(params["m"], float)      # (S,)
        w   = np.asarray(params["w"], float)      # (R,)
        c   = np.asarray(params["c"], float)      # (S,R)
        r   = np.asarray(params["r"], float)      # (R,)
        K_or_kappa = np.asarray(params["K"] if "K" in params else params["kappa"], float)  # (R,)

        S, R = c.shape
        if num_species is None:   num_species = S
        if num_resources is None: num_resources = R

        N = np.array(initial_N, float) if initial_N is not None else np.full(num_species, 0.1, float)
        Rv = np.array(initial_R, float) if initial_R is not None else np.full(num_resources, 5.0, float)

        T = int(timesteps)
        t = np.arange(T, dtype=float) * float(dt)
        N_traj = np.zeros((T, num_species), float); N_traj[0] = N
        R_traj = np.zeros((T, num_resources), float); R_traj[0] = Rv

        def dN_dt(N_vec, R_vec):
            growth_input = (c * w.reshape(1, -1)) @ R_vec   # (S,)
            return (N_vec / tau) * (growth_input - m)

        def dR_dt(R_vec, N_vec):
            if resource_mode in ('logistic', 'external'):
                consumption = (N_vec @ c) * R_vec           # (R,)
            else:  # 'tilman'
                consumption = (N_vec @ c)                   # (R,)
            if resource_mode == 'logistic':
                regeneration = (r / K_or_kappa) * (K_or_kappa - R_vec) * R_vec
            else:  # 'external' or 'tilman'
                regeneration = r * (K_or_kappa - R_vec)
            return regeneration - consumption

        for k in range(1, T):
            N = np.clip(N + dN_dt(N, Rv) * dt, 0.0, None)
            Rv = np.clip(Rv + dR_dt(Rv, N) * dt, 0.0, None)
            N_traj[k] = N
            R_traj[k] = Rv

        self.results['classical_crm'] = {
            't': t,
            'N': N_traj,
            'R': R_traj,
            'mode': resource_mode,
            'params': {'tau': tau, 'm': m, 'w': w, 'c': c, 'r': r, 'K_or_kappa': K_or_kappa}
        }
        return t, N_traj, R_traj

    # --------------------- Runner ---------------------
    def run_all(self, time_step=600, num_points=1000):
        """
        Runs any provided models. For classical CRM, set self.params_classical
        and this will run a short Euler sim with its own timestep grid.
        """
        t_eval = np.linspace(0, time_step, num_points)
        self.results = {}

        # MCRM
        if self.params_mcrm:
            x0 = self.params_mcrm['x0']
            sol = solve_ivp(lambda t, x: self._mcrm_dynamics(t, x), [0, time_step], x0, t_eval=t_eval)
            self.results['mcrm'] = {'t': sol.t, 'X': sol.y}

        # Gibbs
        if self.params_gibbs:
            x0 = np.concatenate([self.params_gibbs['R0'], self.params_gibbs['N0']])
            sol = solve_ivp(lambda t, x: self._gibbs_dynamics(t, x), [0, time_step], x0, t_eval=t_eval)
            self.results['gibbs'] = {'t': sol.t, 'X': sol.y}

        # MiCRM
        if self.params_micrm:
            x0 = np.concatenate([self.params_micrm['R0'], self.params_micrm['N0']])
            sol = solve_ivp(lambda t, x: self._micrm_dynamics(t, x), [0, time_step], x0, t_eval=t_eval)
            self.results['micrm'] = {'t': sol.t, 'X': sol.y}

        # Adaptive Picciani–Mori
        if self.params_adaptive:
            S, R = self.params_adaptive['S'], self.params_adaptive['R']
            x0 = np.concatenate([
                np.asarray(self.params_adaptive['N0'], float),
                np.asarray(self.params_adaptive['C0'], float),
                np.asarray(self.params_adaptive['A0'], float).reshape(S * R)
            ])
            sol = solve_ivp(lambda t, x: self.adaptive_crm_dynamics(t, x), [0, time_step], x0, t_eval=t_eval)
            self.results['adaptive'] = {'t': sol.t, 'X': sol.y}

        # Classical MacArthur CRM (Euler; its own config)
        if self.params_classical:
            cp = self.params_classical
            _ = self.classical_crm_dynamics(params=cp,
                                            num_species=None,
                                            num_resources=None,
                                            timesteps=cp.get('timesteps', 50000),
                                            dt=cp.get('dt', 0.01),
                                            initial_N=cp.get('initial_N', None),
                                            initial_R=cp.get('initial_R', None),
                                            resource_mode=cp.get('resource_mode', 'logistic'))

        return self.results



class CRMBenchmarkEngine:
    """
    Benchmark wrapper for a CommunityModels instance that may include:
      - mcrm
      - gibbs
      - micrm
      - adaptive
      - classical_crm

    Performance knobs:
      - compute_stability: compute Jacobian/eigs at the final state (off by default)
      - stability_models: limit stability to a subset of models
      - compute_extra: compute Bray–Curtis & PCA on species-only (can be heavy)
      - metrics_stride: subsample time for extras to speed up (default: 5)

    Usage:
        engine = CRMBenchmarkEngine(model, t_max=1000, n_points=300,
                                    compute_extra=True,
                                    compute_stability=False,
                                    stability_models=('mcrm', 'classical_crm'),
                                    metrics_stride=10)
        df = engine.run()
        print(df)
    """

    def __init__(self,
                 community_model_instance,
                 t_max: float = 1000,
                 n_points: int = 1000,
                 compute_extra: bool = True,
                 compute_stability: bool = False,
                 stability_models: Tuple[str, ...] = ('mcrm',),
                 metrics_stride: int = 5):
        self.community_model = community_model_instance
        self.t_max = float(t_max)
        self.n_points = int(n_points)
        self.compute_extra = bool(compute_extra)
        self.compute_stability = bool(compute_stability)
        self.stability_models = set(stability_models or ())
        self.metrics_stride = max(1, int(metrics_stride))
        self.results = []

    # ----------------------- metrics helpers -----------------------
    @staticmethod
    def alpha_diversity(abundances, threshold=1e-4) -> Dict[str, float]:
        a = np.array(abundances, dtype=float)
        a = a[a > threshold]
        if a.size == 0 or a.sum() == 0:
            return {"Richness": 0, "Shannon": 0, "Simpson": 0,
                    "InvSimpson": 0, "Evenness": 0}
        p = a / a.sum()
        richness = int(p.size)
        shannon = float(entropy(p))
        simpson = float((p ** 2).sum())
        inv_simpson = float(1.0 / simpson) if simpson > 0 else 0.0
        evenness = float(shannon / np.log(richness)) if richness > 1 else 0.0
        return {"Richness": richness, "Shannon": shannon, "Simpson": simpson,
                "InvSimpson": inv_simpson, "Evenness": evenness}

    @staticmethod
    def feasibility_check(species_traj: np.ndarray,
                          resources_traj: np.ndarray,
                          threshold: float = 1e-6) -> Tuple[bool, bool]:
        final_species = species_traj[-1]
        final_resources = resources_traj[-1]
        feasible_species = bool(np.all(final_species >= threshold))
        feasible_resources = bool(np.all(final_resources >= threshold))
        return feasible_species, feasible_resources

    @staticmethod
    def compute_jacobian(f: Callable[[float, np.ndarray], np.ndarray],
                         x_eq: np.ndarray,
                         eps: float = 1e-6) -> np.ndarray:
        x_eq = np.asarray(x_eq, float)
        n = x_eq.size
        J = np.zeros((n, n), float)
        for i in range(n):
            x1 = x_eq.copy(); x2 = x_eq.copy()
            x1[i] -= eps / 2.0
            x2[i] += eps / 2.0
            f1 = np.asarray(f(0.0, x1), float)
            f2 = np.asarray(f(0.0, x2), float)
            J[:, i] = (f2 - f1) / eps
        return J

    @staticmethod
    def stability_check(f: Callable[[float, np.ndarray], np.ndarray],
                        x_eq: np.ndarray) -> Tuple[bool, np.ndarray]:
        try:
            J = CRMBenchmarkEngine.compute_jacobian(f, x_eq)
            eigenvalues = eigvals(J)
            is_stable = bool(np.all(np.real(eigenvalues) < 0))
        except Exception:
            eigenvalues = np.array([])
            is_stable = False
        return is_stable, eigenvalues

    @staticmethod
    def additional_metrics(species_traj: np.ndarray,
                           threshold: float = 1e-6) -> Dict[str, float]:
        """
        Compute extras on SPECIES ONLY (time x S), not the whole state.
        Bray–Curtis computed on compositional trajectories; PCA on the same.
        """
        from scipy.spatial.distance import pdist
        from sklearn.decomposition import PCA

        X = np.asarray(species_traj, float)  # (T,S)
        species_means = np.mean(X, axis=0)
        species_vars = np.var(X, axis=0)

        # Bray–Curtis on composition
        row_sums = X.sum(axis=1, keepdims=True) + threshold
        normed = X / row_sums
        try:
            bc = pdist(normed, metric='braycurtis')
            bc_mean = float(np.mean(bc))
            bc_var = float(np.var(bc))
        except Exception:
            bc_mean = 0.0
            bc_var = 0.0

        # PCA
        try:
            pca = PCA(n_components=2).fit(normed)
            var_explained = pca.explained_variance_ratio_
            pca1, pca2 = float(var_explained[0]), float(var_explained[1])
        except Exception:
            pca1, pca2 = 0.0, 0.0

        return {
            "MeanAbundance": float(np.mean(species_means)),
            "VarAbundance": float(np.mean(species_vars)),
            "BrayCurtisMean": bc_mean,
            "BrayCurtisVar": bc_var,
            "PCA1": pca1,
            "PCA2": pca2
        }

    # ----------------------- model-specific extractors -----------------------
    def _extract_mcrm(self, out):
        X = out['X'].T                 # (T, S+R)
        S = self.community_model.params_mcrm['num_species']
        species = X[:, :S]
        resources = X[:, S:]
        f = lambda t, x: self.community_model._mcrm_dynamics(t, x)  # RHS expects [N,R]
        xeq = X[-1]
        return species, resources, xeq, f

    def _extract_gibbs(self, out):
        X = out['X'].T                 # state = [R; N]
        Rn = self.community_model.params_gibbs['R0'].shape[0]
        Nn = self.community_model.params_gibbs['N0'].shape[0]
        resources = X[:, :Rn]
        species = X[:, Rn:Rn+Nn]
        f = lambda t, x: self.community_model._gibbs_dynamics(t, x)  # RHS expects [R;N]
        xeq = X[-1]                    # already [R;N]
        return species, resources, xeq, f

    def _extract_micrm(self, out):
        X = out['X'].T                 # state = [R; N]
        Rn = self.community_model.params_micrm['R0'].shape[0]
        Nn = self.community_model.params_micrm['N0'].shape[0]
        resources = X[:, :Rn]
        species = X[:, Rn:Rn+Nn]
        f = lambda t, x: self.community_model._micrm_dynamics(t, x)  # RHS expects [R;N]
        xeq = X[-1]
        return species, resources, xeq, f

    def _extract_adaptive(self, out):
        # state = [N (S), C (R), A (S*R)]
        X = out['X'].T
        S = self.community_model.params_adaptive['S']
        R = self.community_model.params_adaptive['R']
        species = X[:, :S]
        resources = X[:, S:S+R]
        f = lambda t, x: self.community_model.adaptive_crm_dynamics(t, x)  # RHS expects [N,C,A]
        xeq = X[-1]
        return species, resources, xeq, f

    def _extract_classical(self, out):
        # classical_crm stored {'N': (T,S), 'R': (T,R)} separately
        Ntraj = out['N']               # (T, S)
        Rtraj = out['R']               # (T, R)
        species = Ntraj
        resources = Rtraj
        params = out['params']
        mode = out.get('mode', 'logistic')
        S = Ntraj.shape[1]
        Rn = Rtraj.shape[1]

        # Inline RHS to enable stability/Jacobian
        tau = float(params['tau'])
        m = np.asarray(params['m'], float)
        w = np.asarray(params['w'], float)        # (R,)
        c = np.asarray(params['c'], float)        # (S,R)
        r = np.asarray(params['r'], float)        # (R,)
        K_or_kappa = np.asarray(params['K_or_kappa'], float)

        def rhs(t, x):
            N = x[:S]
            Rv = x[S:]
            growth_input = (c * w.reshape(1, -1)) @ Rv      # (S,)
            dN = (N / tau) * (growth_input - m)
            cons = (N @ c) * Rv if mode in ('logistic', 'external') else (N @ c)
            regen = (r / K_or_kappa) * (K_or_kappa - Rv) * Rv if mode == 'logistic' else r * (K_or_kappa - Rv)
            dR = regen - cons
            dN = np.where((N <= 0) & (dN < 0), 0.0, dN)
            dR = np.where((Rv <= 0) & (dR < 0), 0.0, dR)
            return np.concatenate([dN, dR])

        xeq = np.concatenate([Ntraj[-1], Rtraj[-1]])
        return species, resources, xeq, rhs

    # ----------------------- main runner -----------------------
    def run(self) -> pd.DataFrame:
        """
        Calls community_model.run_all with the configured (t_max, n_points),
        then benchmarks each available model output.
        """
        results_dict = self.community_model.run_all(time_step=self.t_max, num_points=self.n_points)
        self.results = []

        # Dispatch table for extractors
        dispatch = {
            'mcrm': self._extract_mcrm,
            'gibbs': self._extract_gibbs,
            'micrm': self._extract_micrm,
            'adaptive': self._extract_adaptive,
            'classical_crm': self._extract_classical,
        }

        for name, out in results_dict.items():
            if name not in dispatch:
                # Unknown block (ignore)
                continue

            # Extract species/resources, final state, and RHS
            species, resources, xeq, f_dyn = dispatch[name](out)

            # Core metrics
            alpha = self.alpha_diversity(species[-1])
            feas_s, feas_r = self.feasibility_check(species, resources)

            # Optional stability (Jacobian + eigs)
            stable, eigs = (False, np.array([]))
            if self.compute_stability and (name in self.stability_models):
                stable, eigs = self.stability_check(f_dyn, xeq)

            result = {
                "Model": name,
                "FeasibleSpecies": feas_s,
                "FeasibleResources": feas_r,
                "Stable": stable,
                "MaxEigenvalue": float(np.max(np.real(eigs))) if eigs.size > 0 else np.nan,
                **alpha
            }

            # Optional extra metrics (subsample to speed up)
            if self.compute_extra:
                species_sub = species[::self.metrics_stride]  # time subsampling
                extras = self.additional_metrics(species_sub)
                result.update(extras)

            self.results.append(result)

        return pd.DataFrame(self.results)


def plot_community_results(results, num_species, num_resources, save_dir="community_plots"):
    """
    Plot species and resource trajectories from CommunityModels results.
    Handles: mcrm, gibbs, micrm, adaptive, classical_crm
    Saves plots instead of just showing them.
    """
    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    for model_name, res in results.items():
        print(f"Plotting and saving {model_name} ...")
        t = res['t']

        if model_name == 'classical_crm':
            N_traj, R_traj = res['N'], res['R']

        elif model_name == 'adaptive':
            X = res['X']
            N_traj = X[:num_species, :].T
            R_traj = X[num_species:num_species+num_resources, :].T

        else:
            X = res['X']
            if model_name == 'gibbs':
                R_traj = X[:num_resources, :].T
                N_traj = X[num_resources:, :].T
            elif model_name == 'micrm':
                R_traj = X[:num_resources, :].T
                N_traj = X[num_resources:, :].T
            else:  # mcrm
                N_traj = X[:num_species, :].T
                R_traj = X[num_species:num_species+num_resources, :].T

        # --- Plot species and resources ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for i in range(N_traj.shape[1]):
            axes[0].plot(t, N_traj[:, i], label=f"Species {i+1}")
        axes[0].set_title(f"{model_name} - Species")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Abundance")
        axes[0].legend()

        for j in range(R_traj.shape[1]):
            axes[1].plot(t, R_traj[:, j], label=f"Resource {j+1}")
        axes[1].set_title(f"{model_name} - Resources")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Concentration")
        axes[1].legend()

        plt.tight_layout()

        # Save figure
        file_path = os.path.join(save_dir, f"{model_name}_trajectories.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close to avoid display overlap

        print(f"Saved: {file_path}")

def plot_species_multimodel(results, num_species, save_dir="community_plots"):
    """
    Create one multi-panel figure with species trajectories for all models.
    Each subplot corresponds to a model.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)

    if len(results) == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        t = res['t']

        if model_name == 'classical_crm':
            N_traj = res['N']
        elif model_name == 'adaptive':
            X = res['X']
            N_traj = X[:num_species, :].T
        else:  # mcrm, gibbs, micrm
            X = res['X']
            if model_name in ['gibbs', 'micrm']:
                N_traj = X[num_species:, :].T
            else:  # mcrm
                N_traj = X[:num_species, :].T

        for i in range(N_traj.shape[1]):
            ax.plot(t, N_traj[:, i], label=f"Sp {i + 1}")
        ax.set_title(f"{model_name} - Species")
        ax.set_xlabel("Time")
        if ax == axes[0]:
            ax.set_ylabel("Abundance")
        ax.legend(fontsize=8)

    plt.tight_layout()
    file_path = os.path.join(save_dir, "all_models_species.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved species multi-panel: {file_path}")

def plot_resources_multimodel(results, num_resources, save_dir="community_plots"):
    """
    Create one multi-panel figure with resource trajectories for all models.
    Each subplot corresponds to a model.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)

    if len(results) == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        t = res['t']

        if model_name == 'classical_crm':
            R_traj = res['R']
        elif model_name == 'adaptive':
            X = res['X']
            R_traj = X[num_species:num_species + num_resources, :].T
        else:  # mcrm, gibbs, micrm
            X = res['X']
            if model_name in ['gibbs', 'micrm']:
                R_traj = X[:num_resources, :].T
            else:  # mcrm
                R_traj = X[num_species:num_species + num_resources, :].T

        for j in range(R_traj.shape[1]):
            ax.plot(t, R_traj[:, j], label=f"Res {j + 1}")
        ax.set_title(f"{model_name} - Resources")
        ax.set_xlabel("Time")
        if ax == axes[0]:
            ax.set_ylabel("Concentration")
        ax.legend(fontsize=8)

    plt.tight_layout()
    file_path = os.path.join(save_dir, "all_models_resources.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved resources multi-panel: {file_path}")

def plot_benchmark_summary(df, save_prefix=None):
    """
    Make summary plots from CRMBenchmarkEngine results DataFrame.
    Required columns: Model, MaxEigenvalue, Stable, Richness, Shannon, Evenness,
                      BrayCurtisMean, FeasibleSpecies, FeasibleResources.

    Parameters
    ----------
    df : pandas.DataFrame
    save_prefix : str | None
        If given, saves each figure as f"{save_prefix}_<name>.png"
    """
    # Ensure model order is consistent on all plots
    models = list(df["Model"])
    x = np.arange(len(models))

    # 1) Max eigenvalue (stability margin)
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x, df["MaxEigenvalue"].fillna(np.nan), width=0.6)
    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right")
    ax1.set_ylabel("Max eigenvalue (Re)")
    ax1.set_title("Stability margin per model")
    ax1.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if save_prefix:
        fig1.savefig(f"{save_prefix}_maxeig.png", dpi=300, bbox_inches="tight")

    # 2) Diversity metrics: Richness, Shannon, Evenness
    metrics = ["Richness", "Shannon", "Evenness"]
    fig2, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, m in zip(axes, metrics):
        ax.bar(x, df[m], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(m)
        ax.grid(alpha=0.3, axis="y")
    fig2.suptitle("Diversity metrics", y=1.02)
    plt.tight_layout()
    if save_prefix:
        fig2.savefig(f"{save_prefix}_diversity.png", dpi=300, bbox_inches="tight")

    # 3) Bray–Curtis mean across time
    if "BrayCurtisMean" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.bar(x, df["BrayCurtisMean"], width=0.6)
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=20, ha="right")
        ax3.set_ylabel("Bray–Curtis (mean)")
        ax3.set_title("Community compositional variation")
        ax3.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if save_prefix:
            fig3.savefig(f"{save_prefix}_braycurtis.png", dpi=300, bbox_inches="tight")

    # 4) Feasibility strip (species/resources)
    feas_mat = np.vstack([df["FeasibleSpecies"].astype(float).to_numpy(),
                          df["FeasibleResources"].astype(float).to_numpy()])
    fig4, ax4 = plt.subplots(figsize=(7, 2.2))
    im = ax4.imshow(feas_mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["Species feasible", "Resources feasible"])
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=20, ha="right")
    ax4.set_title("Feasibility")
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_prefix:
        fig4.savefig(f"{save_prefix}_feasibility.png", dpi=300, bbox_inches="tight")

    # 5) Stability vs diversity scatter
    fig5, ax5 = plt.subplots(figsize=(6.5, 4.2))
    y = df["MaxEigenvalue"].to_numpy()
    s = df["Stable"].astype(bool).to_numpy()
    shannon = df["Shannon"].to_numpy()
    # scatter (Stable vs Unstable markers)
    ax5.scatter(shannon[s], y[s], marker="o", label="Stable")
    ax5.scatter(shannon[~s], y[~s], marker="x", label="Unstable")
    ax5.axhline(0, linestyle="--", linewidth=1)
    ax5.set_xlabel("Shannon diversity")
    ax5.set_ylabel("Max eigenvalue (Re)")
    ax5.set_title("Stability vs diversity")
    ax5.legend(frameon=False)
    ax5.grid(alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        fig5.savefig(f"{save_prefix}_stability_scatter.png", dpi=300, bbox_inches="tight")


def radar_plot_from_results(df, save_dir="community_plots", filename="radar_plot.png"):
    """
    Compute scores (Coexistence, Adaptation, Stability, Tractability) from results dataframe
    and save radar plot.

    Expected df columns: ["Model", "Stable", "MaxEigenvalue",
                          "Richness", "Evenness", "VarAbundance",
                          "BrayCurtisMean", "PCA1"]
    """

    os.makedirs(save_dir, exist_ok=True)

    # Normalizer
    scaler = MinMaxScaler()

    # ---- Coexistence: Richness + Evenness
    coexistence = scaler.fit_transform(df[["Richness"]])[:,0]*0.5 + \
                  scaler.fit_transform(df[["Evenness"]])[:,0]*0.5

    # ---- Adaptation: VarAbundance high + BrayCurtis low
    adapt = scaler.fit_transform(df[["VarAbundance"]])[:,0]*0.5 + \
            (1 - scaler.fit_transform(df[["BrayCurtisMean"]])[:,0])*0.5

    # ---- Stability: Stable flag + eigenvalue (negative better)
    eig_norm = scaler.fit_transform(df[["MaxEigenvalue"]])[:,0]
    stability = np.where(df["Stable"], 1 - eig_norm, 0.2)

    # ---- Tractability: PCA1 high, VarAbundance low
    tractability = scaler.fit_transform(df[["PCA1"]])[:,0]*0.7 + \
                   (1 - scaler.fit_transform(df[["VarAbundance"]])[:,0])*0.3

    # Collect scores
    scores = pd.DataFrame({
        "Model": df["Model"],
        "Coexistence": coexistence,
        "Adaptation": adapt,
        "Stability": stability,
        "Tractability": tractability
    })

    # ========== Radar Plot ==========
    categories = ["Coexistence", "Adaptation", "Stability", "Tractability"]
    N = len(categories)

    # Angles for radar
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

    for _, row in scores.iterrows():
        values = row[categories].tolist()
        values += values[:1]  # close
        ax.plot(angles, values, label=row["Model"], linewidth=2)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_yticklabels([])  # hide radial labels
    ax.set_title("CRM Model Comparison", size=14, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    # Save
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Radar plot saved: {file_path}")
    return scores


if __name__=="__main__":

    # -------------------- SET RANDOM SEED --------------------
    np.random.seed(42)

    # -------------------- CONFIG --------------------
    num_species = 5
    num_resources = 5

    # -------------------- SHARED INITIAL CONDITIONS --------------------
    N0 = np.ones(num_species) * 0.5
    R0 = np.ones(num_resources) * 0.5

    # -------------------- SHARED PARAMETERS --------------------
    C = np.array([
        [1, 0, 0.5, 0, 0],
        [0, 1, 0, 0.5, 0],
        [0.5, 0, 1, 0, 0.5],
        [0, 0.5, 0, 1, 0],
        [0, 0, 0.5, 0, 1]
    ])
    D = np.array([
        [0, 0.2, 0, 0, 0.1],
        [0.1, 0, 0.3, 0, 0],
        [0, 0.1, 0, 0.4, 0.1],
        [0.2, 0, 0.1, 0, 0.2],
        [0.1, 0.1, 0, 0.3, 0]
    ])
    death_vals = np.random.uniform(0.05, 0.1, size=num_species)
    tau = np.ones(num_resources)
    alpha = np.ones(num_resources)
    B = np.ones(num_resources) * 0.1
    rho = np.ones(num_resources) * 0.2
    leakage = 0.2
    g = 1.0
    w = np.ones(num_species)

    # =============================
    # Helper to coerce vectors
    # =============================
    def _as_resource_vec(x, R, name):
        """
        Cast x to a length-R 1D float vector.
        Accepts scalar, (R,), or other shapes (fallback = mean broadcast).
        """
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return np.full(R, float(arr))
        if arr.shape == (R,):
            return arr
        # fallback: broadcast mean (change to raise if you prefer strictness)
        return np.full(R, float(arr.mean()))


    # =============================
    # Parameter construction
    # (expects you already defined:
    # num_species, num_resources, C, D, B, alpha, tau, death_vals,
    # rho, leakage, w, g, N0, R0)
    # =============================

    # -------------------- MCRM PARAMETERS --------------------
    mcrm_params = {
        'num_species': num_species,
        'num_resources': num_resources,
        'C': C,
        'D': D,
        'W': np.eye(num_resources),
        'B': B,
        'alpha': alpha,
        'tau': tau,
        'death_rate': death_vals,
        'mu': 1.0,
        'T': 0.5,
        'x0': np.concatenate([N0, R0]),
        'varIdx': {
            'species': list(range(num_species)),
            'resources': list(range(num_species, num_species + num_resources))
        },
        'W_mode': 'shared'
    }

    # -------------------- GIBBS PARAMETERS --------------------
    P = (D > 0).astype(float)
    P /= P.sum(axis=1, keepdims=True)
    Pt = P.T
    gibbs_params = {
        'num_resources': num_resources,
        'C': C.T,
        'epsilon': np.ones_like(C).T,
        'P': P,
        'Pt': Pt,
        'rho': rho,
        'theta': 0.3,
        'eta': death_vals,  # same as death_rate
        'R0': R0,
        'N0': N0
    }

    # -------------------- MICRM PARAMETERS --------------------
    micrm_params = {
        'num_resources': num_resources,
        'C': C,
        'D': D,
        'm': death_vals,  # same as MCRM death_rate / Gibbs eta
        'leakage': leakage,
        'rho': rho,
        'tau': tau,
        'w': w,
        'g': g,
        'R0': R0,
        'N0': N0
    }

    # -------------------- ADAPTIVE (Picciani–Mori) PARAMETERS --------------------
    S, R = num_species, num_resources
    v_adapt = w if 'w' in globals() else np.ones(R)  # (R,)
    K_adapt = np.full(R, 0.5)  # (R,) Monod K (tune as needed)
    s_adapt = rho  # (R,) resource inflow
    mu_adapt = np.zeros(R)  # (R,) 0 for batch; set >0 for decay
    lam_adapt = np.full(S, 0.3)  # (S,) adaptation speed
    E_star = np.full(S, 1.0)  # (S,) budget per species

    A0_raw = np.clip(C, 0.0, None).astype(float)  # (S,R)
    row_sums = A0_raw.sum(axis=1, keepdims=True) + 1e-12
    A0 = A0_raw / row_sums * E_star[:, None]  # (S,R)

    adaptive_params = {
        'S': S, 'R': R,
        'v': v_adapt,  # (R,)
        'K': K_adapt,  # (R,)
        'd': death_vals,  # (S,)
        's': s_adapt,  # (R,)
        'mu': mu_adapt,  # (R,)
        'lam': lam_adapt,  # (S,)
        'E_star': E_star,  # (S,)
        'N0': N0,  # (S,)
        'C0': R0,  # (R,)
        'A0': A0,  # (S,R)
        'B': np.zeros((R, S, R)),  # optional cross-feeding; fill if needed
        'nonneg_clip': True,
    }

    # -------------------- CLASSICAL MacArthur CRM PARAMETERS --------------------
    # Robustly coerce tau and alpha to length-R vectors
    tau_vec = _as_resource_vec(tau, R, "tau")
    alpha_vec = _as_resource_vec(alpha, R, "alpha")

    eps = 1e-12
    r_classical = 1.0 / np.maximum(tau_vec, eps)  # (R,)
    K_classical = alpha_vec  # (R,)

    classical_params = {
        "tau": float(np.asarray(tau_vec).mean()),
        "m": death_vals,  # (S,)
        "w": w,  # (R,)
        "c": C,  # (S,R)
        "r": r_classical,  # (R,)
        "K": K_classical,  # (R,)
        "timesteps": 60000,
        "dt": 0.01,
        "initial_N": N0,
        "initial_R": R0,
        "resource_mode": "external",  # 'logistic', 'external', or 'tilman'
    }

    # -------------------- BUILD & RUN --------------------
    model = CommunityModels(
        mcrm_params=mcrm_params,
        gibbs_params=gibbs_params,
        micrm_params=micrm_params,
        adaptive_params=adaptive_params,
        classical_params=classical_params
    )

    results = model.run_all()

    # Access:
    # results['mcrm']           -> {'t', 'X'}    # state = [N (S), R (R)] as provided in x0
    # results['gibbs']          -> {'t', 'X'}    # state = [R; N]
    # results['micrm']          -> {'t', 'X'}    # state = [R; N]
    # results['adaptive']       -> {'t', 'X'}    # state = [N (S), C (R), A (S*R)]
    # results['classical_crm']  -> {'t', 'N', 'R', 'mode', 'params'}

    plot_community_results(results=results, num_species=num_species, num_resources=num_resources, save_dir="community_plots")
    plot_species_multimodel(results, num_species, save_dir="community_plots")
    plot_resources_multimodel(results, num_resources, save_dir="community_plots")

    engine = CRMBenchmarkEngine(
        model,
        t_max=1000,
        n_points=500,
        compute_extra=True,
        compute_stability=True,  # turn stability ON
        stability_models=('mcrm', 'gibbs', 'micrm', 'adaptive', 'classical_crm'),  # include all
        metrics_stride=10
    )

    df = engine.run()
    plot_benchmark_summary(df, save_prefix="benchmark")

    scores = radar_plot_from_results(df, save_dir="community_plots", filename="models_radar.png")
    print(scores)


