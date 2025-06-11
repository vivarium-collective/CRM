from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Define the adaptive metabolic strategy model
def adaptive_strategy_model(y, t, v_gal, v_eth, K_gal, K_eth, Y, Q, delta, d, E_star):

    n, c_gal, c_eth, a_gal, a_eth = y

    # Resource uptake rates using Monod function
    r_gal = c_gal / (K_gal + c_gal)
    r_eth = c_eth / (K_eth + c_eth)

    # Population growth
    dn_dt = n * (v_gal * a_gal * r_gal + v_eth * a_eth * r_eth - delta)

    # Resource consumption
    dc_gal_dt = -n * a_gal * r_gal
    dc_eth_dt = -n * a_eth * r_eth + Y * n * a_gal * r_gal  # Ethanol production from galactose

    # Total metabolic uptake
    total_uptake = a_gal + a_eth

    # Heaviside function for constraint enforcement
    theta = 1 if total_uptake >= E_star else 0

    # Penalty term for exceeding metabolic capacity
    penalty_term = theta * (total_uptake / E_star) * (v_gal * r_gal * a_gal + v_eth * r_eth * a_eth)

    # Adaptive metabolic strategy updates
    da_gal_dt = a_gal * d * delta * (v_gal * r_gal - penalty_term)
    da_eth_dt = a_eth * d * delta * (v_eth * r_eth - penalty_term)

    return [dn_dt, dc_gal_dt, dc_eth_dt, da_gal_dt, da_eth_dt]


# model parameters (from Table A, Figure 1A)
v_gal = 1.20e10  # cells/g of resource
v_eth = 1.25e10  # cells/g of resource
K_gal = 1.e-3  # g of resource/mL
K_eth = 9.67e-3  # g of resource/mL
Y = 0.53  # g ethanol/g galactose
Q = 2.18e-5  # g of resource/cell
delta = 2.15e-6  # death rate (1/h)
alpha_gal_0 = 0.7e-11  # Initial galactose metabolic strategy (g of resource/(cell·h))
alpha_eth_0 = 3.75e-12  # Initial ethanol metabolic strategy (g of resource/(cell·h))
d = 4.20e-6  # Adaptation velocity (nondimensional)

# Initial conditions including metabolic strategies
n0 = 1.0e6 # Initial population (cells/mL)
c_gal_0 = 5.0e-3  # Initial galactose concentration (g/mL)
c_eth_0 = 0.0  # Initial ethanol concentration (g/mL)
a_gal_0 = alpha_gal_0  # Initial galactose metabolic strategy
a_eth_0 = alpha_eth_0  # Initial ethanol metabolic strategy
y0 = [n0, c_gal_0, c_eth_0, a_gal_0, a_eth_0]

# Time range for integration
t = np.linspace(0, 71, 1000)  # Simulate for 70 hours

# Parameters from Table A
E_star = Q  # Maximum total resource uptake capacity

# Integrate the ODE system with adaptive metabolic strategies
sol = odeint(adaptive_strategy_model, y0, t, args=(v_gal, v_eth, K_gal, K_eth, Y, Q, delta, d, E_star))
n, c_gal, c_eth, a_gal, a_eth = sol.T

# Plot results to visualize adaptive metabolic strategies
plt.figure(figsize=(8, 8))

# Population Growth
plt.subplot(3, 1, 1)
plt.plot(t, n, label='Population (cells/mL)', color='black')
plt.ylabel('Population (cells/mL)')
plt.yscale('log')
plt.legend()

# Resource Dynamics
plt.subplot(3, 1, 2)
plt.plot(t, c_gal, label='Galactose (g/mL)', color='blue')
plt.plot(t, c_eth, label='Ethanol (g/mL)', color='red')
plt.ylabel('Resource Concentration (g/mL)')
plt.legend()

# Adaptive Metabolic Strategies
plt.subplot(3, 1, 3)
plt.plot(t, a_gal, label='Galactose Strategy', color='blue', linestyle='dashed')
plt.plot(t, a_eth, label='Ethanol Strategy', color='red', linestyle='dashed')
plt.xlabel('Time (h)')
plt.ylabel('Metabolic Strategy (g of resource/(cell·h))')
plt.legend()

plt.suptitle('Reproduction of Figure 1A: Adaptive Metabolic Strategy Model')
plt.tight_layout()
plt.show()

# Load the yeast growth data
file_path = '/Users/edwin/Downloads/yeast_growth_data.csv'  # Replace with your file path
yeast_data = pd.read_csv(file_path)
yeast_data_cleaned = yeast_data[["Time", "MeanDensity"]].iloc[1:]

# Extract time and density data
time_data = yeast_data_cleaned["Time"].values
density_data = yeast_data_cleaned["MeanDensity"].values

# Convert data to numeric for modeling
yeast_data_cleaned["Time"] = pd.to_numeric(yeast_data_cleaned["Time"], errors="coerce")
yeast_data_cleaned["MeanDensity"] = pd.to_numeric(yeast_data_cleaned["MeanDensity"], errors="coerce")
yeast_data_cleaned = yeast_data_cleaned.dropna()

# Extract relevant columns
time = yeast_data['Time']
mean_density = yeast_data['MeanDensity']
std_dev = yeast_data['Standard deviation']

# Plot experimental data and MCMC simulation
plt.figure(figsize=(8, 8))
# Plot the mean density as a line
plt.plot(time, mean_density, label="Experimental Data", color='blue', linewidth=2)
plt.plot(t, n, label='Simulation', color='black')
plt.ylabel('Population (cells/mL)')
plt.xlabel("Time (hours)")
plt.legend()
plt.show()



