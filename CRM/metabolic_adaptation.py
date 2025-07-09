import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model
from cobra import Model


# dFBA timestep function
def dfba_timestep(
    model: Model,
    initial_conditions: dict,
    kinetic_params: dict,
    substrate_to_reaction_map: dict,
    biomass_name_map: tuple,
    dt: float,
):
    updated_state = initial_conditions.copy()

    # Set substrate uptake bounds using Michaelis-Menten kinetics
    for substrate_id, reaction_id in substrate_to_reaction_map.items():
        Km, Vmax = kinetic_params[substrate_id]
        S = initial_conditions[substrate_id]
        flux = Vmax * S / (Km + S + 1e-8)  # avoid divide-by-zero
        model.reactions.get_by_id(reaction_id).lower_bound = -flux  # uptake is negative

    # Solve FBA
    solution = model.optimize()

    # Get current biomass and biomass flux (growth rate)
    biomass_id, biomass_rxn = biomass_name_map
    biomass_flux = solution.fluxes[biomass_rxn]
    current_biomass = updated_state[biomass_id]

    # Update biomass using: dX = µ * X * dt
    new_biomass = current_biomass + (biomass_flux * current_biomass * dt)
    updated_state[biomass_id] = new_biomass

    # Update substrates using: dS = flux * biomass * dt
    for substrate_id, reaction_id in substrate_to_reaction_map.items():
        flux = solution.fluxes[reaction_id]
        substrate_conc = updated_state[substrate_id]
        updated_state[substrate_id] = substrate_conc + (flux * current_biomass * dt)

    return updated_state


# Perform full dFBA simulation
def perform_dfba(
    model: Model,
    initial_conditions: dict,
    kinetic_params: dict,
    substrate_to_reaction_map: dict,
    biomass_name_map: tuple,
    dt: float,
    total_steps: int,
):
    results = {key: [value] for key, value in initial_conditions.items()}
    current_state = initial_conditions.copy()

    for _ in range(total_steps):
        current_state = dfba_timestep(
            model,
            current_state,
            kinetic_params,
            substrate_to_reaction_map,
            biomass_name_map,
            dt,
        )
        for key in current_state:
            results[key].append(current_state[key])

    return results


class AdaptiveMetabolicSimulator:
    def __init__(self, params, initial_conditions, t, mode='single'):
        """
        mode: 'single', 'multi', or 'crossfeeding'
        """
        self.params = params
        self.initial_conditions = initial_conditions
        self.t = t
        self.mode = mode

    def _single_species_model(self, y, t):
        v_gal = self.params['v_gal']
        v_eth = self.params['v_eth']
        K_gal = self.params['K_gal']
        K_eth = self.params['K_eth']
        Y = self.params['Y']
        Q = self.params['Q']
        delta = self.params['delta']
        d = self.params['d']

        n, c_gal, c_eth, a_gal, a_eth = y
        r_gal = c_gal / (K_gal + c_gal)
        r_eth = c_eth / (K_eth + c_eth)

        dn_dt = n * (v_gal * a_gal * r_gal + v_eth * a_eth * r_eth - delta)
        dc_gal_dt = -n * a_gal * r_gal
        dc_eth_dt = -n * a_eth * r_eth + Y * n * a_gal * r_gal

        total_uptake = a_gal + a_eth
        theta = 1 if total_uptake >= Q else 0
        penalty = theta * (total_uptake / Q) * (v_gal * r_gal * a_gal + v_eth * r_eth * a_eth)

        da_gal_dt = a_gal * d * delta * (v_gal * r_gal - penalty)
        da_eth_dt = a_eth * d * delta * (v_eth * r_eth - penalty)

        return [dn_dt, dc_gal_dt, dc_eth_dt, da_gal_dt, da_eth_dt]

    def _multi_species_model(self, y, t):
        v, K, Y, Q, delta, d = [self.params[k] for k in ['v', 'K', 'Y', 'Q', 'delta', 'd']]
        S, R = v.shape

        n = y[:S]
        c = y[S:S+R]
        a = y[S+R:].reshape((S, R))

        r = c / (K + c)
        growth = np.sum(v * a * r, axis=1)
        dn_dt = n * (growth - delta)

        dc_dt = -np.sum(n[:, None] * a * r, axis=0)
        if R >= 2:
            dc_dt[1] += np.sum(Y * n * a[:, 0] * r[:, 0])

        total_uptake = np.sum(a, axis=1)
        theta = (total_uptake >= Q).astype(float)
        penalty = theta[:, None] * (total_uptake / Q)[:, None] * np.sum(v * r * a, axis=1, keepdims=True)
        da_dt = a * d[:, None] * delta[:, None] * (v * r - penalty)

        return np.concatenate([dn_dt, dc_dt, da_dt.flatten()])

    def _crossfeeding_model(self, y, t):
        v, K, Y, Q, delta, d = [self.params[k] for k in ['v', 'K', 'Y', 'Q', 'delta', 'd']]
        S, R = v.shape

        n = y[:S]
        c = y[S:S+R]
        a = y[S+R:].reshape((S, R))

        r = np.zeros_like(a)
        for i in range(S):
            for j in range(R):
                if K[i, j] > 0:
                    r[i, j] = c[j] / (K[i, j] + c[j])

        growth = np.sum(v * a * r, axis=1)
        dn_dt = n * (growth - delta)

        dc_dt = np.zeros(R)
        dc_dt[0] = -np.sum(n[0] * a[0, 0] * r[0, 0])
        dc_dt[1] = -np.sum(n[1] * a[1, 1] * r[1, 1])
        dc_dt[1] += np.sum(Y[0] * n[0] * a[0, 0] * r[0, 0])

        total_uptake = np.sum(a, axis=1)
        theta = (total_uptake >= Q).astype(float)
        penalty = theta[:, None] * (total_uptake / Q)[:, None] * np.sum(v * r * a, axis=1, keepdims=True)
        da_dt = a * d[:, None] * delta[:, None] * (v * r - penalty)

        return np.concatenate([dn_dt, dc_dt, da_dt.flatten()])

    def run(self, plot=False):
        if self.mode == 'single':
            sol = odeint(self._single_species_model, self.initial_conditions, self.t)
        elif self.mode == 'multi':
            sol = odeint(self._multi_species_model, self.initial_conditions, self.t)
        elif self.mode == 'crossfeeding':
            sol = odeint(self._crossfeeding_model, self.initial_conditions, self.t)
        else:
            raise ValueError("Invalid mode. Choose 'single', 'multi', or 'crossfeeding'.")

        self.solution = sol

        if plot:
            self.plot()

        return self.t, sol

    def plot(self):
        sol = self.solution

        if self.mode == 'single':
            n, c_gal, c_eth, a_gal, a_eth = sol.T

            plt.figure(figsize=(10, 8))
            plt.subplot(3, 1, 1)
            plt.plot(self.t, n, color='black', label='Population')
            plt.yscale('log')
            plt.ylabel('Cells/mL')
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.plot(self.t, c_gal, label='Galactose', color='blue')
            plt.plot(self.t, c_eth, label='Ethanol', color='red')
            plt.ylabel('Resource (g/mL)')
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(self.t, a_gal, label='Galactose Strategy', linestyle='--', color='blue')
            plt.plot(self.t, a_eth, label='Ethanol Strategy', linestyle='--', color='red')
            plt.xlabel('Time (h)')
            plt.ylabel('Strategy')
            plt.legend()

            plt.tight_layout()
            plt.suptitle("Adaptive Metabolic Strategy Simulation", y=1.02)
            plt.show()

        else:
            S = self.params['v'].shape[0]
            R = self.params['v'].shape[1]

            n = self.solution[:, :S].T
            c = self.solution[:, S:S+R].T
            a = self.solution[:, S+R:].reshape((-1, S, R)).transpose((1, 2, 0))

            plt.figure(figsize=(8, 6))
            for i in range(S):
                plt.plot(self.t, n[i], label=f"Species {i+1}")
            plt.xlabel("Time (h)")
            plt.ylabel("Biomass (cells/mL)")
            plt.legend()
            plt.title("Species Growth")
            plt.show()

            plt.figure(figsize=(8, 6))
            for j in range(R):
                plt.plot(self.t, c[j], label=f"Resource {j+1}")
            plt.xlabel("Time (h)")
            plt.ylabel("Concentration (g/mL)")
            plt.legend()
            plt.title("Resource Dynamics")
            plt.show()

            plt.figure(figsize=(8, 6))
            for i in range(S):
                for j in range(R):
                    plt.plot(self.t, a[i, j], label=f"S{i+1} R{j+1}", linestyle="--" if j else "-")
            plt.xlabel("Time (h)")
            plt.ylabel("Metabolic Strategy")
            plt.legend()
            plt.title("Adaptive Metabolic Strategies")
            plt.show()