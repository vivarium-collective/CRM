from process_bigraph import Process
from process_bigraph.emitter import emitter_from_wires
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class adaptive_crm(Process):
    config_schema = {
        "species_number": "integer",
        "resource_number": "integer",
        "adaptation_rate": "map[float]",  # d_sigma
        "death_rate": "map[float]",  # delta_sigma
        "yield_coefficient": "map[map[float]]",  # species_name -> {resource_name: Y_ij}
        "resource_value": "map[float]",  # v_i
        "monod_constant": "map[map[float]]",  # K_ij
        "metabolic_capacity": "map[float]",  # Q_sigma
    }

    def initialize(self, config):
        self.species_number = config["species_number"]
        self.resource_number = config["resource_number"]

        self.adaptation_rate = config["adaptation_rate"]
        self.death_rate = config["death_rate"]
        self.yield_coefficient = config["yield_coefficient"]
        self.resource_value = config["resource_value"]
        self.monod_constant = config["monod_constant"]
        self.metabolic_capacity = config["metabolic_capacity"]

        self.species_names = list(self.death_rate.keys())
        self.resource_names = list(self.resource_value.keys())

        self.d_array = np.array([self.adaptation_rate[s] for s in self.species_names])
        self.delta_array = np.array([self.death_rate[s] for s in self.species_names])
        self.Q_array = np.array([self.metabolic_capacity[s] for s in self.species_names])
        self.v_array = np.array([self.resource_value[r] for r in self.resource_names])

        self.K_matrix = np.array([
            [self.monod_constant[s][r] for r in self.resource_names]
            for s in self.species_names
        ])

        self.Y_matrix = np.array([
            [self.yield_coefficient[s][r] for r in self.resource_names]
            for s in self.species_names
        ])

    def inputs(self):
        return {
            "species": "map[float]",
            "concentrations": "map[float]",
            "strategies": "map[map[float]]",
        }

    def outputs(self):
        return {
            "species_delta": "map[float]",
            "concentrations_delta": "map[float]",
            "strategies_delta": "map[map[float]]",
        }

    def adaptive_ode(self, t, y):
        S, R = self.species_number, self.resource_number

        n = y[:S]
        c = y[S:S+R]
        a = y[S+R:].reshape((S, R))

        r = c / (self.K_matrix + c)
        growth = np.sum(self.v_array * a * r, axis=1)

        # Population dynamics
        dn_dt = n * (growth - self.delta_array)

        # Resource dynamics (no external inflow or decay)
        dc_dt = -np.sum(n[:, None] * a * r, axis=0)
        if R >= 2:
            dc_dt[1] += np.sum(self.Y_matrix[:, 1] * n * a[:, 0] * r[:, 0])

        # Strategy dynamics
        total_uptake = np.sum(a, axis=1)
        theta = (total_uptake >= self.Q_array).astype(float)
        penalty = theta[:, None] * (total_uptake / self.Q_array)[:, None] * \
                  np.sum(self.v_array * r * a, axis=1, keepdims=True)
        da_dt = a * self.d_array[:, None] * self.delta_array[:, None] * \
                (self.v_array * r - penalty)

        return np.concatenate([dn_dt, dc_dt, da_dt.flatten()])

    def update(self, state, interval):
        n0 = np.array([state["species"][s] for s in self.species_names])
        c0 = np.array([state["concentrations"][r] for r in self.resource_names])
        a0 = np.array([
            [state["strategies"][s][r] for r in self.resource_names]
            for s in self.species_names
        ])
        y0 = np.concatenate([n0, c0, a0.flatten()])

        sol = solve_ivp(self.adaptive_ode, [0, interval], y0, t_eval=[interval], method="RK45")
        y1 = np.maximum(sol.y[:, -1], 0)

        n1 = y1[:self.species_number]
        c1 = y1[self.species_number:self.species_number + self.resource_number]
        a1 = y1[self.species_number + self.resource_number:].reshape((self.species_number, self.resource_number))

        delta_species = dict(zip(self.species_names, n1 - n0))
        delta_conc = dict(zip(self.resource_names, c1 - c0))
        delta_strategies = {
            self.species_names[i]: {
                self.resource_names[j]: a1[i, j] - a0[i, j]
                for j in range(self.resource_number)
            }
            for i in range(self.species_number)
        }

        return {
            "species_delta": delta_species,
            "concentrations_delta": delta_conc,
            "strategies_delta": delta_strategies
        }


def get_adaptive_crm_emitter(state_keys):
    """
    Returns a standard emitter step spec for Adaptive CRM simulations.
    Only includes relevant state keys if present.

    Parameters:
        state_keys (list): list of keys in the adaptive CRM state,
            e.g., ['species', 'concentrations', 'strategies', 'global_time']

    Returns:
        dict: emitter step spec usable in a Composite
    """
    POSSIBLE_KEYS = {"species", "concentrations", "strategies", "global_time"}
    included_keys = [key for key in POSSIBLE_KEYS if key in state_keys]

    emitter_spec = {key: [key] for key in included_keys}
    return emitter_from_wires(emitter_spec)


def plot_adaptive_crm_simulation(results):
    """
    Automatically plots species, resource, and strategy dynamics from Adaptive CRM simulation results.

    Parameters:
        results (list): output of `gather_emitter_results(sim)[('emitter',)]`
    """
    if not results:
        raise ValueError("No results to plot")

    # Step 1: Identify entity names
    first_result = results[0]
    species_names = list(first_result.get("species", {}).keys())
    resource_names = list(first_result.get("concentrations", {}).keys())
    has_strategies = "strategies" in first_result
    timepoints = [r.get("global_time", i) for i, r in enumerate(results)]

    # Step 2: Build matrices
    biomass = np.array([[r["species"][s] for s in species_names] for r in results])
    resources = np.array([[r["concentrations"][res] for res in resource_names] for r in results])
    times = np.array(timepoints)

    if has_strategies:
        strategies = {
            s: np.array([[r["strategies"][s][res] for res in resource_names] for r in results])
            for s in species_names
        }

    # Step 3: Plot
    n_rows = 3 if has_strategies else 2
    fig, axs = plt.subplots(n_rows, 1, figsize=(8, 7), sharex=True)

    # Biomass
    for i, s in enumerate(species_names):
        axs[0].plot(times, biomass[:, i], label=s)
    axs[0].set_ylabel("Biomass")
    axs[0].set_title("Species Dynamics")
    axs[0].legend()

    # Resource
    for i, r in enumerate(resource_names):
        axs[1].plot(times, resources[:, i], label=r)
    axs[1].set_ylabel("Concentration")
    axs[1].set_title("Resource Dynamics")
    axs[1].legend()

    # Strategy (optional)
    if has_strategies:
        for i, s in enumerate(species_names):
            for j, rname in enumerate(resource_names):
                axs[2].plot(times, strategies[s][:, j], label=f"{s}-{rname}")
        axs[2].set_ylabel("Strategy (a)")
        axs[2].set_xlabel("Time (h)")
        axs[2].set_title("Adaptive Strategy Dynamics")
        axs[2].legend()

    plt.tight_layout()
    plt.show()
