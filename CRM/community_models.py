import numpy as np
from scipy.integrate import solve_ivp

class CommunityModels:
    def __init__(self, mcrm_params=None, gibbs_params=None, micrm_params=None):
        self.params_mcrm = mcrm_params
        self.params_gibbs = gibbs_params
        self.params_micrm = micrm_params
        self.results = {}

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

        growth = np.sum(C * W * R, axis=1) - T if p.get('W_mode', 'shared') == 'species' else C @ (W @ R) - T
        dx[v['species']] = N * mu * growth - death_rate * N
        consumption = (C * R).T @ N
        production = D @ consumption
        dx[v['resources']] = (alpha - R) / tau - consumption + production + B * (death_rate @ N)
        return dx

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

    def _micrm_dynamics(self, t, x):
        p = self.params_micrm
        R = x[:p['num_resources']]  # R_α
        N = x[p['num_resources']:]  # N_i

        # Parameters
        C = p['C']                 # c_{iα}
        D = p['D']                 # D_{αβ}
        l = p['leakage']           # ℓ_α (scalar or array)
        rho = p['rho']             # κ_α
        tau = p['tau']             # τ_α
        w = p['w']                 # w_i
        m = p['m']                 # m_i
        g = p['g']                 # scalar

        # Ensure leakage is in array form
        leak = np.array(l) if isinstance(l, (list, np.ndarray)) else np.full(R.shape, l)

        # Compute uptake and growth
        resource_uptake = C * R  # shape: (species, resources)
        one_minus_leak = (1 - leak)
        net_uptake = np.sum(resource_uptake * one_minus_leak, axis=1)  # shape: (species,)
        dN = g * N * (w * net_uptake - m)  # shape: (species,)

        # Consumption term
        consumption = np.sum(C * (N[:, None] * R[None, :]), axis=0)  # shape: (resources,)

        # Leakage release term
        release_terms = C * (N[:, None] * R[None, :])  # shape: (species, resources)
        scaled_leak = (w[:, None] * leak[None, :]) * release_terms  # shape: (species, resources)
        total_release = scaled_leak.sum(axis=0)  # shape: (resources,)
        release = D @ total_release  # shape: (resources,)

        dR = rho - R / tau - consumption + release
        return np.concatenate([dR, dN])

    def run_all(self, time_step=10000, num_points=1000):
        t_eval = np.linspace(0, time_step, num_points)
        self.results = {}

        if self.params_mcrm:
            x0 = self.params_mcrm['x0']
            sol = solve_ivp(lambda t, x: self._mcrm_dynamics(t, x), [0, time_step], x0, t_eval=t_eval)
            self.results['mcrm'] = {'t': sol.t, 'X': sol.y}

        if self.params_gibbs:
            x0 = np.concatenate([self.params_gibbs['R0'], self.params_gibbs['N0']])
            sol = solve_ivp(lambda t, x: self._gibbs_dynamics(t, x), [0, time_step], x0, t_eval=t_eval)
            self.results['gibbs'] = {'t': sol.t, 'X': sol.y}

        if self.params_micrm:
            x0 = np.concatenate([self.params_micrm['R0'], self.params_micrm['N0']])
            sol = solve_ivp(lambda t, x: self._micrm_dynamics(t, x), [0, time_step], x0, t_eval=t_eval)
            self.results['micrm'] = {'t': sol.t, 'X': sol.y}

        return self.results