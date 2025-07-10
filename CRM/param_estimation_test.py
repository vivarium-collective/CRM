import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, dual_annealing
from pyswarm import pso
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

class AdaptiveMetabolicEstimator:
    def __init__(self, t, observed_data, initial_conditions):
        self.t = t
        self.observed_data = observed_data
        self.initial_conditions = initial_conditions

    def adaptive_strategy_model(self, y, t, v_gal, v_eth, K_gal, K_eth, Y, Q, delta, d):
        n, c_gal, c_eth, a_gal, a_eth = y
        r_gal = c_gal / (K_gal + c_gal + 1e-10)
        r_eth = c_eth / (K_eth + c_eth + 1e-10)
        dn_dt = n * (v_gal * a_gal * r_gal + v_eth * a_eth * r_eth - delta)
        dc_gal_dt = -n * a_gal * r_gal
        dc_eth_dt = -n * a_eth * r_eth + Y * n * a_gal * r_gal
        total_uptake = a_gal + a_eth
        theta = 1 if total_uptake >= Q else 0
        penalty_term = theta * (total_uptake / Q) * (v_gal * r_gal * a_gal + v_eth * r_eth * a_eth)
        da_gal_dt = a_gal * d * delta * (v_gal * r_gal - penalty_term)
        da_eth_dt = a_eth * d * delta * (v_eth * r_eth - penalty_term)
        return [dn_dt, dc_gal_dt, dc_eth_dt, da_gal_dt, da_eth_dt]


    def simulate(self, params):
        v_gal, v_eth, K_gal, K_eth, Y, Q, delta, d = params
        sol = odeint(self.adaptive_strategy_model, self.initial_conditions, self.t,
                     args=(v_gal, v_eth, K_gal, K_eth, Y, Q, delta, d))
        return sol[:, 0]  # Return population

    def objective(self, params):
        pred = self.simulate(params)
        return np.mean((pred - self.observed_data) ** 2)

    def fit_pso(self, bounds):
        lb, ub = zip(*bounds)
        best_params, best_error = pso(self.objective, lb, ub, swarmsize=30, maxiter=100)
        return best_params, best_error

    def fit_de(self, bounds):
        result = differential_evolution(self.objective, bounds)
        return result.x, result.fun

    def fit_annealing(self, bounds):
        result = dual_annealing(self.objective, bounds)
        return result.x, result.fun

    def fit_ga(self, bounds):
        varbound = np.array(bounds)
        algorithm_param = {'max_num_iteration': 100, 'population_size': 30, 'mutation_probability': 0.1,
                           'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3,
                           'crossover_type': 'uniform', 'max_iteration_without_improv': None}

        model = ga(function=self.objective,
                   dimension=len(bounds),
                   variable_type='real',
                   variable_boundaries=varbound,
                   algorithm_parameters=algorithm_param)
        model.run()
        return model.output_dict['variable'], model.output_dict['function']

    class CRMParameterEstimator:
        def __init__(self, t_obs, N_obs, resource_mode='logistic'):
            self.t_obs = t_obs
            self.N_obs = N_obs
            self.resource_mode = resource_mode

        def simulate_crm(self, params, initial_N, initial_R, dt=0.01):
            num_species = len(initial_N)
            num_resources = len(initial_R)
            timesteps = len(self.t_obs)
            tau, m, w1, w2, c11, c12, c21, c22, r1, r2, K1, K2 = params
            tau = np.array([tau] * num_species)
            m = np.array([m] * num_species)
            w = np.array([w1, w2])
            c = np.array([[c11, c12], [c21, c22]])
            r = np.array([r1, r2])
            K_or_kappa = np.array([K1, K2])

            N = np.array(initial_N)
            R = np.array(initial_R)
            N_traj = np.zeros((timesteps, num_species))
            R_traj = np.zeros((timesteps, num_resources))
            N_traj[0] = N
            R_traj[0] = R

            def dN_dt(N, R):
                growth_input = np.dot(c * w.reshape(1, -1), R)
                return (N / tau) * (growth_input - m)

            def dR_dt(R, N):
                if self.resource_mode == 'logistic':
                    regeneration = (r / K_or_kappa) * (K_or_kappa - R) * R
                    consumption = np.dot(N, c) * R
                elif self.resource_mode == 'external':
                    regeneration = r * (K_or_kappa - R)
                    consumption = np.dot(N, c) * R
                elif self.resource_mode == 'tilman':
                    regeneration = r * (K_or_kappa - R)
                    consumption = np.dot(N, c)
                return regeneration - consumption

            for t in range(1, timesteps):
                N += dN_dt(N, R) * dt
                R += dR_dt(R, N) * dt
                N = np.clip(N, 0, None)
                R = np.clip(R, 0, None)
                N_traj[t] = N
                R_traj[t] = R

            return N_traj, R_traj

        def objective_function(self, params, initial_N, initial_R):
            N_traj, _ = self.simulate_crm(params, initial_N, initial_R)
            N_model = N_traj[:, 0]
            return np.mean((self.N_obs - N_model) ** 2)

        def fit_de(self, bounds, initial_N, initial_R):
            result = differential_evolution(
                lambda params: self.objective_function(params, initial_N, initial_R),
                bounds, seed=42)
            return result

        def fit_pso(self, lb, ub, initial_N, initial_R):
            result = pso(
                lambda params: self.objective_function(params, initial_N, initial_R),
                lb, ub, swarmsize=30, maxiter=50)
            return result

        def fit_ga(self, varbound, initial_N, initial_R):
            def f(params):
                return self.objective_function(params, initial_N, initial_R)

            algorithm_param = {'max_num_iteration': 50,
                               'population_size': 30,
                               'mutation_probability': 0.1,
                               'elit_ratio': 0.01,
                               'crossover_probability': 0.5,
                               'parents_portion': 0.3,
                               'crossover_type': 'uniform',
                               'max_iteration_without_improv': None}
            model = ga(function=f,
                       dimension=12,
                       variable_type='real',
                       variable_boundaries=varbound,
                       algorithm_parameters=algorithm_param)
            model.run()
            return model.output_dict

        def plot_fit(self, best_params, initial_N, initial_R):
            N_traj, _ = self.simulate_crm(best_params, initial_N, initial_R)
            N_model = N_traj[:, 0]
            plt.plot(self.t_obs, self.N_obs, 'o', label='Observed')
            plt.plot(self.t_obs, N_model, '-', label='Fitted')
            plt.xlabel("Time")
            plt.ylabel("Biomass")
            plt.legend()
            plt.title("Fit of CRM Model")
            plt.show()