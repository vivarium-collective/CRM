import matplotlib.pyplot as plt
import numpy as np


# Scenario generator function for trait-based CRM simulations
def generate_crm_scenario(scenario_name="High Resource Availability", num_species=5, num_resources=3):
    """Generate a parameter dictionary for a given CRM scenario."""
    if scenario_name == "High Resource Availability":
        return {
            "tau": np.random.uniform(1, 3, num_species),
            "m": np.random.uniform(0.1, 0.5, num_species),
            "w": np.random.uniform(0.5, 1.5, num_resources),
            "c": np.random.rand(num_species, num_resources),
            "K": np.full(num_resources, 20.0),
            "r": np.random.uniform(1.0, 2.0, num_resources)
        }

    elif scenario_name == "Low Resource Availability":
        return {
            "tau": np.random.uniform(1, 3, num_species),
            "m": np.random.uniform(0.1, 0.5, num_species),
            "w": np.random.uniform(0.5, 1.5, num_resources),
            "c": np.random.rand(num_species, num_resources),
            "K": np.full(num_resources, 5.0),
            "r": np.random.uniform(0.1, 0.5, num_resources)
        }

    elif scenario_name == "Niche Specialization":
        return {
            "tau": np.random.uniform(1, 3, num_species),
            "m": np.random.uniform(0.1, 0.5, num_species),
            "w": np.ones(num_resources),
            "c": np.eye(num_species, num_resources),
            "K": np.full(num_resources, 10.0),
            "r": np.random.uniform(0.5, 1.5, num_resources)
        }

    elif scenario_name == "Strong Generalist Competition":
        return {
            "tau": np.random.uniform(1, 3, num_species),
            "m": np.random.uniform(0.1, 0.5, num_species),
            "w": np.ones(num_resources),
            "c": np.ones((num_species, num_resources)),
            "K": np.full(num_resources, 10.0),
            "r": np.random.uniform(0.5, 1.5, num_resources)
        }

    elif scenario_name == "Different Timescales":
        return {
            "tau": np.concatenate([np.full(num_species // 2, 0.5), np.full(num_species - num_species // 2, 2.5)]),
            "m": np.random.uniform(0.1, 0.5, num_species),
            "w": np.random.uniform(0.5, 1.5, num_resources),
            "c": np.random.rand(num_species, num_resources),
            "K": np.full(num_resources, 10.0),
            "r": np.random.uniform(0.5, 1.5, num_resources)
        }

    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}")


def simulate_crm(params=None, scenario_name=None, num_species=5, num_resources=3,
                 timesteps=10000, dt=0.01, initial_N=None, initial_R=None,
                 plot=True, resource_mode=None):
    """
    Simulate a Consumer-Resource Model (CRM) with optional trait-based scenario generation
    and selectable resource dynamics.

    Parameters:
    - params: dict with keys (ignored if scenario_name is provided)
    - scenario_name: str, name of a predefined scenario
    - resource_mode: 'logistic' (default), 'external', or 'tilman'
    - Returns: N_traj, R_traj (trajectories)
    """

    # Default behavior
    if resource_mode is None:
        resource_mode = 'logistic'

    # Scenario generation
    if scenario_name is not None:
        print(f"Generating parameters for scenario: {scenario_name}")
        params = generate_crm_scenario(scenario_name, num_species, num_resources)
    elif params is None:
        raise ValueError("You must provide either 'params' or 'scenario_name'.")

    # Print mode info
    if resource_mode == 'logistic':
        print("Simulating the classical MacArthur CRM with logistic resource growth...")
    elif resource_mode == 'external':
        print("Simulating CRM with externally supplied resources (mass-action depletion)...")
    elif resource_mode == 'tilman':
        print("Simulating Tilman's CRM with externally supplied resources and constant-rate consumption...")
    else:
        raise ValueError("Invalid resource_mode. Choose 'logistic', 'external', or 'tilman'.")

    # Unpack parameters
    tau, m, w, c = params["tau"], params["m"], params["w"], params["c"]
    r = params["r"]
    K_or_kappa = params["K"] if "K" in params else params["kappa"]

    # Initial conditions
    N = np.array(initial_N) if initial_N is not None else np.full(num_species, 0.1)
    R = np.array(initial_R) if initial_R is not None else np.full(num_resources, 5.0)

    N_traj = np.zeros((timesteps, num_species))
    R_traj = np.zeros((timesteps, num_resources))
    N_traj[0] = N
    R_traj[0] = R

    # dN/dt
    def dN_dt(N, R):
        growth_input = np.dot(c * w.reshape(1, -1), R)
        return (N / tau) * (growth_input - m)

    # dR/dt
    def dR_dt(R, N):
        if resource_mode == 'logistic':
            regeneration = (r / K_or_kappa) * (K_or_kappa - R) * R
            consumption = np.dot(N, c) * R
        elif resource_mode == 'external':
            regeneration = r * (K_or_kappa - R)
            consumption = np.dot(N, c) * R
        elif resource_mode == 'tilman':
            regeneration = r * (K_or_kappa - R)
            consumption = np.dot(N, c)
        return regeneration - consumption

    # Simulation loop
    for t in range(1, timesteps):
        N += dN_dt(N, R) * dt
        R += dR_dt(R, N) * dt
        N = np.clip(N, 0, None)
        R = np.clip(R, 0, None)
        N_traj[t] = N
        R_traj[t] = R

    # Plotting
    if plot:
        time = np.arange(timesteps) * dt
        plt.figure(figsize=(10, 5))
        for i in range(num_species):
            plt.plot(time, N_traj[:, i], label=f'Species {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Population Dynamics')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        for j in range(num_resources):
            plt.plot(time, R_traj[:, j], label=f'Resource {j+1}')
        plt.xlabel('Time')
        plt.ylabel('Resource Concentration')
        plt.title('Resource Dynamics')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return N_traj, R_traj


# Plotting function for CRM simulation results
def plot_crm_dynamics(N_traj, R_traj, title_suffix=""):
    """
    Plot population and resource dynamics for CRM simulation.

    Parameters:
    - N_traj: 2D array of species populations over time
    - R_traj: 2D array of resource concentrations over time
    - title_suffix: str to append to plot titles for scenario context
    """
    timesteps = N_traj.shape[0]

    # Plot Population Dynamics
    plt.figure(figsize=(12, 5))
    for i in range(N_traj.shape[1]):
        plt.plot(range(timesteps), N_traj[:, i], label=f"Species {i+1}")
    plt.title(f"Population Dynamics {title_suffix}")
    plt.xlabel("Time step")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Plot Resource Dynamics
    plt.figure(figsize=(12, 5))
    for j in range(R_traj.shape[1]):
        plt.plot(range(timesteps), R_traj[:, j], label=f"Resource {j+1}")
    plt.title(f"Resource Dynamics {title_suffix}")
    plt.xlabel("Time step")
    plt.ylabel("Resource Concentration")
    plt.legend()
    plt.grid(True)
    plt.show()
