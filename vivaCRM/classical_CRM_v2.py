from process_bigraph import Process
from process_bigraph.emitter import emitter_from_wires
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class CRM(Process):
    """
    Consumer-Resource Model (CRM) Process using process_bigraph.

    Simulates species-resource interactions with different resource dynamics (modes):
    - 'logistic': logistic growth of resources
    - 'external': externally supplied resources (linear inflow)
    - 'tilman': externally supplied with constant uptake rate
    """

    config_schema = {
        "species_number": "integer",
        "resource_number": "integer",
        "tau": "map[float]",
        "maintenance": "map[float]",                # species_name -> m_sigma
        "resource_value": "map[float]",             # resource_name -> w_i
        "resource_uptake_rate": "map[map[float]]",  # species_name -> {resource_name: c_sigma_i}
        "carrying_capacity": "map[float]",          # resource_name -> K_i
        "uptake_rate": "map[float]",                # resource_name -> r_i
        "resource_mode": "string",                  # 'logistic', 'external', or 'tilman'
    }

    def initialize(self, config):
        self.species_number = config["species_number"]
        self.resource_number = config["resource_number"]
        self.tau = config["tau"]
        self.maintenance = config["maintenance"]
        self.resource_value = config["resource_value"]
        self.resource_uptake_rate = config["resource_uptake_rate"]
        self.carrying_capacity = config["carrying_capacity"]
        self.uptake_rate = config["uptake_rate"]
        self.mode = config["resource_mode"]

        # Create ordered lists of species and resources for consistent mapping
        self.species_names = list(self.maintenance.keys())
        self.resource_names = list(self.resource_value.keys())

        # Convert to arrays for fast math
        self.m = np.array([self.maintenance[s] for s in self.species_names])
        self.w = np.array([self.resource_value[r] for r in self.resource_names])
        self.K = np.array([self.carrying_capacity[r] for r in self.resource_names])
        self.r = np.array([self.uptake_rate[r] for r in self.resource_names])
        self.C = np.array([[self.resource_uptake_rate[s][r] for r in self.resource_names] for s in self.species_names])
        self.tau_array = np.array([self.tau[s] for s in self.species_names])

    def inputs(self):
        return {
            "species": "map[float]",
            "concentrations": "map[float]",
        }

    def outputs(self):
        return {
            "species_delta": "map[float]",
            "concentrations_delta": "map[float]",
        }

    def crm_ode(self, t, y):
        N = y[:self.species_number]   # species abundances
        R = y[self.species_number:]   # resource concentrations

        # Species growth
        growth_input = np.dot(self.C * self.w, R)       # shape: (species,)
        dNdt = (N / self.tau_array) * (growth_input - self.m)

        # Resource dynamics
        if self.mode == "logistic":
            regeneration = (self.r / self.K) * (self.K - R) * R
            consumption = np.dot(N, self.C) * R
        elif self.mode == "external":
            regeneration = self.r * (self.K - R)
            consumption = np.dot(N, self.C) * R
        elif self.mode == "tilman":
            regeneration = self.r * (self.K - R)
            consumption = np.dot(N, self.C)
        else:
            raise ValueError(f"Invalid resource_mode: {self.mode}")

        dRdt = regeneration - consumption
        return np.concatenate([dNdt, dRdt])

    def update(self, state, interval):
        N0 = np.array([state["species"][s] for s in self.species_names])
        R0 = np.array([state["concentrations"][r] for r in self.resource_names])
        y0 = np.concatenate([N0, R0])

        sol = solve_ivp(self.crm_ode, [0, interval], y0, method="RK45", t_eval=[interval])
        y_final = np.maximum(sol.y[:, -1], 0)

        Nf = y_final[:self.species_number]
        Rf = y_final[self.species_number:]

        species_delta = dict(zip(self.species_names, Nf - N0))
        concentrations_delta = dict(zip(self.resource_names, Rf - R0))

        return {
            "species_delta": species_delta,
            "concentrations_delta": concentrations_delta,
        }


def get_crm_emitter(state_keys):
    """
    Returns a standard emitter step spec for CRM simulations.
    Only includes relevant CRM state keys if present.

    Parameters:
        state_keys (list): list of keys in the CRM state, e.g., ['species', 'concentrations']

    Returns:
        dict: emitter step spec usable in a Composite
    """
    POSSIBLE_KEYS = {"species", "concentrations", "global_time"}
    included_keys = [key for key in POSSIBLE_KEYS if key in state_keys]

    # emitter_from_wires takes a dict: {output_key: [input_key]}
    emitter_spec = {key: [key] for key in included_keys}
    return emitter_from_wires(emitter_spec)


def plot_crm_simulation(results):
    """
    Automatically plots species and resource dynamics from CRM simulation results.

    Parameters:
        results (list): output of `gather_emitter_results(sim)[('emitter',)]`
    """
    if not results:
        raise ValueError("No results to plot")

    # Step 1: Identify species and resource names from first entry
    first_result = results[0]
    species_names = list(first_result.get("species", {}).keys())
    resource_names = list(first_result.get("concentrations", {}).keys())
    timepoints = [r["global_time"] for r in results]

    # Step 2: Build biomass and resource matrices
    biomass = np.array([
        [r["species"][s] for s in species_names]
        for r in results
    ])
    resources = np.array([
        [r["concentrations"][res] for res in resource_names]
        for r in results
    ])
    times = np.array(timepoints)

    # Step 3: Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot species dynamics
    for i, s in enumerate(species_names):
        axs[0].plot(times, biomass[:, i], label=s)
    axs[0].set_ylabel("Biomass")
    axs[0].legend()
    axs[0].set_title("Consumer Resource Model: Biomass Dynamics")

    # Plot resource dynamics
    for i, res in enumerate(resource_names):
        axs[1].plot(times, resources[:, i], label=res)
    axs[1].set_ylabel("Resource Concentration")
    axs[1].set_xlabel("Time (h)")
    axs[1].legend()
    axs[1].set_title("Resource Dynamics")

    plt.tight_layout()
    plt.show()


