import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, weibull_min
import pandas as pd
import random

# Define parameters
num_simulations = 1000
time_steps = 48

#Electricity load
ele_file = "electricity_load_profile_sample.xlsm"
ele_start_row = 1
ele_end_row = 2
ele_start_column = 1
ele_end_column =  24
ele_data = pd.read_excel(ele_file, header = None, nrows = ele_end_row - ele_start_row + 1, usecols = range(ele_start_column, ele_end_column + 1),
                                                                                                           skiprows = range(0, ele_start_row))
ele = ele_data.values.flatten()
load = np.ones(time_steps)

for t in range(time_steps):
        load[t] = (15+2*random.uniform(0,1) * ele[t])

load_mean = np.mean(load)  # Mean electricity load in kW
load_std_dev = np.std(load)  # Standard deviation of electricity load
solar_alpha = 2.5  # Shape parameter for beta distribution of solar energy
solar_beta = 4.0   # Shape parameter for beta distribution of solar energy
wind_shape = 2.0   # Shape parameter for Weibull distribution of wind energy
wind_scale = 10.0  # Scale parameter for Weibull distribution of wind energy
battery_capacity = 1  # Battery capacity in kWh
battery_efficiency = 0.85  # Efficiency of the battery system

# Generate random samples for uncertainties
loads = np.random.normal(load_mean, load_std_dev, (time_steps, num_simulations))
solar_energy = beta.rvs(solar_alpha, solar_beta, size=(time_steps, num_simulations))
wind_energy = weibull_min.rvs(wind_shape, scale=wind_scale, size=(time_steps, num_simulations))
hydropower_energy = np.random.uniform(20, 25, (time_steps, num_simulations))  # Example range for hydropower
biogas_energy = np.random.triangular(1, 1.5, 2, (time_steps, num_simulations))  # Example triangular distribution for biogas

#print ("solar values:",solar_energy)

# Define function for loss of power supply probability (LPSP)
def calculate_lpsp(energy_supply, energy_demand):
    return np.sum(energy_supply < energy_demand) / len(energy_supply)

# Perform Monte Carlo simulation for each time step
lpsps = np.zeros(time_steps)
for t in range(time_steps):
    energy_demand = loads[t]
    energy_supply = (0.2 * 0.63 * 26 * solar_energy[t]/1000) + (0.5 * 1.18 * 3.142 * 1.9 * 0.4 * wind_energy[t]**3 /4000) + \
        (6 * hydropower_energy[t] * 0.7 * 50 * 9.8/ 3600) + (5.6 * biogas_energy[t]/50) + \
                    (5 * battery_capacity * battery_efficiency)
    print("Energy demand:",energy_demand)
    print("Energy supply:",energy_supply)
    lpsp = calculate_lpsp(energy_supply, energy_demand)
    #print("LPSP:",lpsp)
    lpsps[t] = lpsp.mean()

# Aggregate LPSP values over the 48-hour period
total_lpsp = np.mean(lpsps)

# Plot LPSP over time
plt.plot(range(1, time_steps + 1), lpsps)
plt.title('Loss of Power Supply Probability (LPSP) Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('LPSP')
plt.grid(True)
plt.show()

# Print total LPSP
print("Total LPSP over 48 hours:", total_lpsp)
