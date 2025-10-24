import numpy as np
from scipy.integrate import solve_ivp

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