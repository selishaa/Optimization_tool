import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, weibull_min
import pandas as pd
import random
from scipy.special import gamma

# Define parameters
num_simulations = 1
time_steps = 48

No_data = 48
# PV generation:
Voc_stc = 29.61
Isc_stc = 4.11
Vmax = 23.17
Imax = 4.49
Pmax = 104
NCOT = 43
kv = -0.13
ki = 0.00468
#npv = np.random.randint(60, 150)
Ns = 1
Np = 1
Tc = np.zeros(No_data)
Voc = np.zeros(No_data)
Isc = np.zeros(No_data)
P_pv = np.zeros(No_data)
i_pv = np.zeros(No_data)
# Import solar radiation
G_file = "stillwaterdata_sample.xlsx"     # excel file's name 
G_start_row = 1
G_end_row = 48
G_data = pd.read_excel(G_file, header = None, nrows= G_end_row - G_start_row +1, usecols= [6], skiprows= range(0, G_start_row))
G = G_data.values.flatten()
# Import temperature 
T_file = "stillwaterdata_sample.xlsx"
T_start_row = 1
T_end_row = 48
T_data = pd.read_excel(T_file, header = None, nrows = T_end_row - T_start_row + 1, usecols= [4], skiprows = range(0, T_start_row))
Ta= T_data.values.flatten()


# Wind generation 
v_file = "stillwaterdata_sample.xlsx"
v_start_row = 1
v_end_row = 48
v_data = pd.read_excel(v_file, header = None, nrows = v_end_row - v_start_row +1, usecols= [5], skiprows= range(0, v_start_row))
v = v_data.values.flatten()

vin = 2.5
vo = 25
vr = 12
Pr = 600
p_wind = np.zeros(No_data)
wind =np.zeros(No_data)

# Hydro power generation
hydro_gen = np.zeros(No_data)
hydro_ele = np.zeros(No_data)
Hd = 50
n_hydro = 0.7
g = 9.8

# Biogas generation
biogas_gen = np.zeros(No_data)
n_bio = 0.7
bio_elec = np.zeros(No_data)

# Battery 
ele_file = "electricity_load_profile_sample.xlsm"
ele_start_row = 1
ele_end_row = 2
ele_start_column = 1
ele_end_column =  24
ele_data = pd.read_excel(ele_file, header = None, nrows = ele_end_row - ele_start_row + 1, usecols = range(ele_start_column, ele_end_column + 1),
                                                                                                           skiprows = range(0, ele_start_row))
ele = ele_data.values.flatten()
load = np.ones(No_data)
p_bat_total = np.zeros(No_data)
p_bat = np.zeros(No_data)

rated_capacity = 1 # kAh
voltage = 24 #V
p_bat_factor = rated_capacity * voltage # kwh
n_inverter = 0.9

load_mean = np.mean(load)  # Mean electricity load in kW
load_std_dev = np.std(load)  # Standard deviation of electricity load

# Define the mean and standard deviation of solar energy generation
solar_mean = np.mean(G)/1000  # Mean solar energy generation (scaled between 0 and 1)
solar_std_dev = np.std(G)/1000  # Standard deviation of solar energy generation (scaled between 0 and 1)
#scale_factor = 500  # Scale factor to adjust the mean of solar energy (in MW)
print ("Solar mean:", solar_mean)
print("std:", solar_std_dev)

# Calculate alpha and beta parameters for the beta distribution
solar_alpha = (solar_mean ** 2 * (1 - solar_mean) / (solar_std_dev ** 2) - solar_mean) * ((1 - solar_mean) / solar_mean) ** 2
solar_beta= solar_alpha * ((1 - solar_mean) / solar_mean)

print("Alpha:",solar_alpha)
print("Beta:",solar_beta)

# Define mean and standard deviation of wind energy
wind_mean = np.mean(v)  # Mean of wind energy (in m/s)
wind_std_dev = np.std(v)  # Standard deviation of wind energy (in m/s)

# Calculate shape parameter (c) and scale factor (k) based on mean and standard deviation
wind_shape = (wind_std_dev / wind_mean) ** (-1.086)
wind_scale = wind_mean / gamma(1 + 1 / wind_shape)

battery_capacity = 1  # Battery capacity in kWh
battery_efficiency = 0.85  # Efficiency of the battery system

# Loss of power supply
LPS = np.zeros(No_data)

x_1= 26
x_2=1
x_3=6
x_4=1

# Generate random samples for uncertainties
loads = np.random.normal(load_mean, load_std_dev, (time_steps, num_simulations))
solar_energy = beta.rvs(solar_alpha, solar_beta, size=(time_steps, num_simulations))
wind_energy = weibull_min.rvs(wind_shape, scale=wind_scale, size=(time_steps, num_simulations))
hydropower_energy = np.random.uniform(20, 25, (time_steps, num_simulations))  # Example range for hydropower
biogas_energy = np.random.triangular(1, 1.5, 2, (time_steps, num_simulations))  # Example triangular distribution for biogas

# Define function for loss of power supply probability (LPSP)
def calculate_lpsp(energy_supply, energy_demand):
    return np.sum(energy_supply < energy_demand) / len(energy_supply)

# Perform Monte Carlo simulation for each time step
lpsps = np.zeros(time_steps)
for t in range(time_steps):
    #Solar Energy
    Tc[t] = Ta[t] + ((NCOT - 20) / 800) * solar_energy[t]
    Voc[t] = Voc_stc - (kv * Tc[t])
    Isc[t] = (Isc_stc + (ki * (Tc[t] - 25))) * solar_energy[t] 
    i_pv[t] = Ns * Np * Voc[t] * Isc[t] * Pmax / (Vmax * Imax)
    P_pv[t] = x_1 * i_pv[t]
    #Wind Energy
    if vin <= wind_energy[t] <= vr:
            wind[t] = Pr * (wind_energy[t]-vin)/(vr-vin)
            p_wind[t] = x_2 * wind[t]/1000
    elif vr <= wind_energy[t] <= vo:
            wind[t] = Pr
            p_wind[t] = x_2 * wind[t]/1000
    else:
            wind[t] = 0
            p_wind[t] = 0
    #Hydropower
    hydro_gen[t] = x_3* n_hydro * 1 * Hd * g *hydropower_energy[t]/3600
    #Biogas
    biogas_gen[t] = x_4 * 5.6 * biogas_energy[t]/50
    #LPSP
    energy_demand = loads[t]
    energy_supply = P_pv[t] + p_wind[t] + hydro_gen[t] + biogas_gen[t] + \
                    battery_capacity * battery_efficiency
    lpsp = calculate_lpsp(energy_supply, energy_demand)
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
