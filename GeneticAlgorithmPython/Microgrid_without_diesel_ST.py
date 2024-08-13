from deap import base
from deap import creator
from deap import tools
from pages import elitism
from pages.utils import get_graph

import random
import numpy as np 
import math
import pandas as pd 
 
import matplotlib.pyplot as plt
import seaborn as sns
import threading

DIMENSIONS = 5
BOUND_LOW, BOUND_UP = [100,1,1,1,5],[300,100,100,100,100]  

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.5
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE= 10
CROWDING_FACTOR = 20
PENALTY_VALUE = 10**10

r_dash = 0.0375    # nominal interest rate
f = 0.015          # annual inflation rate
n_proj = 25        # lifetime in years 

r = (r_dash- f)/(1+f)
CRF = r * (1+r)**n_proj / ((1+r)**n_proj - 1)

n_rep = 8 #the lifetime in year of a equiment
SFF = r/ ((1+r)**n_rep -1)

# Initialization of power rating  for components in kW 
PRATING_PV = 0.1       # Power rating for PV solar of each array 0.1 kW
PRATING_WT = 1         # Power rating for each wind turbine 1kW
PRATING_HT = 1         # Power rating for each hydro turbine 1kW
PRATING_BG = 5         # Power rating for each biogas powered gererator 5kW
PRATING_CB = 24        # Power rating for battery 1kAh, 24V
# Capital cost 
# for each electrical equipment per kWh
CC_PV = 3000 #$/kW
CC_WT = 1800 #$/kW
CC_HT = 2300 #$/kW
CC_BG = 1200 #$/kW
CC_CB = 1500 #$/kAh


# Maintainance cost 
# for electrical equipment
n_maintain = 25 # the number of years to maintain equiment
CM_PV = 65 #$/year
CM_WT = 95 
CM_HT = 15
CM_BG = 100 
CM_CB = 50

# Replacement cost
# for electrical equipment in 25 years including biogas generator (BG) and battery (CB)
CR_BG = 1000 
CR_CB = 1500

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))                # set minimize condition
creator.create("Individual", list, fitness = creator.FitnessMin)             # create a list of individual that is suitable for minimization 

def randomFloat(low, up):                      # a list of variables that are chosen randomly 
    return [random.randint (l,u) for l, u in zip(low, up)]

toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)              # set the boundaries for the variables in the list
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)      # create individuals
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)             # create population

def minimize(individual):
    x_1 = individual[0]
    x_2 = individual[1]
    x_3 = individual[2]
    x_4 = individual[3]
    x_5 = individual[4]
    InitialCostElec = CRF * (x_1 *PRATING_PV *CC_PV + x_2 * PRATING_WT * CC_WT + x_3 * PRATING_HT * CC_HT
                         + x_4 *PRATING_BG * CC_BG + x_5 * PRATING_CB * CC_CB)
    MaintainCost = (1+f)**n_maintain * (x_1 * CM_PV + x_2 * CM_WT + x_3 * CM_HT + x_4 * CM_BG + x_5 * CM_CB)
    ReplaceCost = SFF * (CR_BG + CR_CB )
    ACS = InitialCostElec + MaintainCost + ReplaceCost
    return ACS,
toolbox.register("evaluate", minimize)

# CONSTRAINT 
No_data = 3*12*24
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
G_file = "GeneticAlgorithmPython/stillwaterdata_sample.xlsx"     # excel file's name 
G_start_row = 1
G_end_row = 3*24*12
G_data = pd.read_excel(G_file, header = None, nrows= G_end_row - G_start_row +1, usecols= [6], skiprows= range(0, G_start_row))
G = G_data.values.flatten()
# Import temperature 
T_file = "GeneticAlgorithmPython/stillwaterdata_sample.xlsx"
T_start_row = 1
T_end_row = 3*24*12
T_data = pd.read_excel(T_file, header = None, nrows = T_end_row - T_start_row + 1, usecols= [4], skiprows = range(0, T_start_row))
Ta= T_data.values.flatten()


# Wind generation 
v_file = "GeneticAlgorithmPython/stillwaterdata_sample.xlsx"
v_start_row = 1
v_end_row = 3*24*12
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
ele_file = "GeneticAlgorithmPython/electricity_load_profile_sample.xlsm"
ele_start_row = 1
ele_end_row = 36
ele_start_column = 1
ele_end_column =  24
ele_data = pd.read_excel(ele_file, header = None, nrows = ele_end_row - ele_start_row + 1, usecols = range(ele_start_column, ele_end_column + 1),
                                                                                                           skiprows = range(0, ele_start_row))
ele = ele_data.values.flatten()

p_bat_total = np.zeros(No_data)
p_bat = np.zeros(No_data)

rated_capacity = 1 # kAh
voltage = 24 #V
p_bat_factor = rated_capacity * voltage # kwh
n_inverter = 0.9
# Load of electricity and cooking purposes
load_elec_cooking = np.ones(No_data)
# Load of water using purpose
load_water = np.ones(No_data)
n_pump = 0.5
Hd = 50 # Check Hd = 20 or 50
# import the water load in m3
water_file = "GeneticAlgorithmPython/water_usage_data_final_sample.xlsx"
water_start_row = 1
water_end_row = 3*12*24
water_load_data = pd.read_excel(water_file, header = None, nrows = water_end_row - water_start_row + 1, usecols = [9], 
                                skiprows = range(0, water_start_row))
Q = water_load_data.values.flatten() 
W_irrigation = 80 + 100 * random.uniform(0,1) * np.ones(No_data)
W_load = Q + W_irrigation
# Total load convert to electricity 
load = np.ones(No_data)

# Loss of power supple
LPS = np.zeros(No_data)

# Water load





# constraint function
# Constrain satisfaction
def feasible(individual):
    x_1 = individual[0]
    x_2 = individual[1]
    x_3 = individual[2]
    x_4 = individual[3]
    x_5 = individual[4]
    #PV generation:
    for t in range(No_data):
        Tc[t] = Ta[t] + ((NCOT - 20) / 800) * G[t]
        Voc[t] = Voc_stc - (kv * Tc[t])
        Isc[t] = (Isc_stc + (ki * (Tc[t] - 25))) * G[t] / 1000
        i_pv[t] = Ns * Np * Voc[t] * Isc[t] * Pmax / (Vmax * Imax)
        P_pv[t] = x_1 * i_pv[t]/1000
    # Wind generation:
    for t in range(No_data):
        if vin <= v[t] <= vr:
            wind[t] = Pr * (v[t]-vin)/(vr-vin)
            p_wind[t] = x_2 * wind[t]/1000
        elif vr <= v[t] <= vo:
            wind[t] = Pr
            p_wind[t] = x_2 * wind[t]/1000
        else:
            wind[t] = 0
            p_wind[t] = 0
    # Hydro power generation:
    for t in range(No_data):
        hydro_ele[t] = 20+(5 * random.uniform(0,1))
        hydro_gen[t] = x_3* n_hydro * 1 * Hd * g *hydro_ele[t]/3600
    # Biogas generation:
    for t in range(No_data):
        bio_elec[t] = 1+ random.uniform(0,1)
        biogas_gen[t] = x_4 * 5.6 * bio_elec[t]/50
    #load of electricity and cooking purposes (in kWh)
    for t in range(No_data):
        load_elec_cooking[t] = (15+2*random.uniform(0,1) * ele[t]) + (850/24) 
    # load of water using purpose (in kWh)
    for t in range(No_data):
        load_water[t] = W_load[t] * Hd / (n_pump*367) 
    # total load (in kWh)
    for t in range(No_data):
        load[t] = load_elec_cooking[t] + load_water[t]
    

    # Battery:
    for t in range(No_data):
        if t == 0:
            #if (p_bat_total[t] > 0).all(): #charging
            if p_bat_total[t] > 0: #charging
                p_bat_total[t] =  (p_wind[t] + P_pv[t] + hydro_gen[t] + biogas_gen[t])- load[t]
                p_bat[t] = x_5 * (p_bat_total[t]/p_bat_factor)
            else:
                p_bat[t] = 0
        else:
            #if (p_wind[t] + P_pv[t] + hydro_gen[t] + biogas_gen[t] + p_bat[t-1] > load[t]).all(): # discharging while retaining some charge 
            if p_wind[t] + P_pv[t] + hydro_gen[t] + biogas_gen[t] + p_bat[t-1] > load[t]: # discharging while retaining some charge 
                p_bat_total[t] =   (p_bat_total[t-1] + p_wind[t] + P_pv[t] + hydro_gen[t] + biogas_gen[t] - load[t]) 
                p_bat[t] = x_5 * (p_bat_total[t]/p_bat_factor)
            else:
                p_bat[t] = 0 # load unfulfill
    # Loss of power supply probability
    for t in range(No_data):
     # add more condition to the LPS in case the load is loss
       LPS[t] = (load[t]- (p_wind[t]+ P_pv[t] + hydro_gen[t] + biogas_gen[t]+ p_bat[t]))*n_inverter
       if LPS[t] <= 0:
          LPS[t] = 0
       else:
          LPS[t] = (load[t]- (p_wind[t]+ P_pv[t] + hydro_gen[t] + biogas_gen[t]+ p_bat[t]))*n_inverter
    LPSP = np.sum(LPS)/np.sum(load)
    return LPSP <= 0.1

toolbox.decorate("evaluate",tools.DeltaPenalty(feasible,PENALTY_VALUE))     # set penalty value

toolbox.register("select", tools.selTournament, tournsize = 2)              # selection 
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)   # crossover 
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/ DIMENSIONS)  # mutation

def main(request):
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen= MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])

    x_1, x_2, x_3, x_4, x_5 = best
    #PV generation:                      # define all the equations again to print them inside the main()
    #PV generation:
    for t in range(No_data):
        Tc[t] = Ta[t] + ((NCOT - 20) / 800) * G[t]
        Voc[t] = Voc_stc - (kv * Tc[t])
        Isc[t] = (Isc_stc + (ki * (Tc[t] - 25))) * G[t] / 1000
        i_pv[t] = Ns * Np * Voc[t] * Isc[t] * Pmax / (Vmax * Imax)
        P_pv[t] = x_1 * i_pv[t]/1000
        P_pv_ele = np.sum(P_pv)
    # Wind generation:
    for t in range(No_data):
        if vin <= v[t] <= vr:
            wind[t] = Pr * (v[t]-vin)/(vr-vin)
            p_wind[t] = x_2 * wind[t]/1000
            P_wind_ele = np.sum(p_wind)
        elif vr <= v[t] <= vo:
            wind[t] = Pr
            p_wind[t] = x_2 * wind[t]/1000
            P_wind_ele = np.sum(p_wind)
        else:
            wind[t] = 0
            p_wind[t] = 0
            P_wind_ele = np.sum(p_wind)
    # Hydro power generation:
    for t in range(No_data):
        hydro_ele[t] = 20+(5 * random.uniform(0,1))
        hydro_gen[t] = x_3* n_hydro * 1 * Hd * g *hydro_ele[t]/3600
        hydro_gen_ele = np.sum(hydro_gen)
    # Biogas generation:
    for t in range(No_data):
        bio_elec[t] = 1+ random.uniform(0,1)
        biogas_gen[t] = x_4 * 5.6 * bio_elec[t]/50
        biogas_gen_ele = np.sum(biogas_gen)
    #load of electricity and cooking purposes (in kWh)
    for t in range(No_data):
        load_elec_cooking[t] = (15+2*random.uniform(0,1) * ele[t]) + 850/24
        load_total = np.sum(load_elec_cooking)
    # load of water using purpose (in kWh)
    for t in range(No_data):
        load_water[t] = W_load[t] * Hd / (n_pump*367)
    # total load (in kWh)
    for t in range(No_data):
        load[t] = load_elec_cooking[t] + load_water[t]
        total_load = np.sum(load)
    
    # Battery:
    for t in range(No_data):
        if t == 0:
            #if (p_bat_total[t] > 0).all(): #charging
            if p_bat_total[t] > 0: #charging
                p_bat_total[t] =  (p_wind[t] + P_pv[t] + hydro_gen[t] + biogas_gen[t])- load[t]
                p_bat[t] = x_5 * (p_bat_total[t]/p_bat_factor)
            else:
                p_bat[t] = 0
        else:
            #if (p_wind[t] + P_pv[t] + hydro_gen[t] + biogas_gen[t] + p_bat[t-1] > load[t]).all(): # discharging while retaining some charge 
            if p_wind[t] + P_pv[t] + hydro_gen[t] + biogas_gen[t] + p_bat[t-1] > load[t]: # discharging while retaining some charge 
                p_bat_total[t] =   (p_bat_total[t-1] + p_wind[t] + P_pv[t] + hydro_gen[t] + biogas_gen[t] - load[t]) 
                p_bat[t] = x_5 * (p_bat_total[t]/p_bat_factor)
            else:
                p_bat[t] = 0 # load unfulfill
    # Loss of power supply probability
    for t in range(No_data):
     # add more condition to the LPS in case the load is loss
       LPS[t] = (load[t]- (p_wind[t]+ P_pv[t] + hydro_gen[t] + biogas_gen[t]+ p_bat[t]))*n_inverter
       if LPS[t] <= 0:
          LPS[t] = 0
       else:
          LPS[t] = (load[t]- (p_wind[t]+ P_pv[t] + hydro_gen[t] + biogas_gen[t]+ p_bat[t]))*n_inverter
    LPSP = np.sum(LPS)/np.sum(load)

    print("LPS =", LPS)
    print("LPSP =",LPSP)

    InitialCost = CRF * (x_1 *PRATING_PV *CC_PV + x_2 * PRATING_WT * CC_WT + x_3 * PRATING_HT * CC_HT
                         + x_4 *PRATING_BG * CC_BG + x_5 * PRATING_CB * CC_CB)
    MaintainCost = (1+f)**n_maintain * (x_1 * CM_PV + x_2 * CM_WT + x_3 * CM_HT + x_4 * CM_BG + x_5 * CM_CB)
    ReplaceCost = SFF * (CR_BG + CR_CB)
    ACS = InitialCost + MaintainCost + ReplaceCost    
    NPC = ACS/CRF
    print("ACS = ", ACS)
    print("Initial cost = ", InitialCost)
    print("Net present cost (NPC) = ", NPC) 

    # Lifecycle CO2 emission 
    # CO2 emission from PV solar producing electricity
    P_pv_CO2 = (P_pv_ele*85)/10**6

    #CO2 emission from wind producing electricity
    P_wind_CO2 = (P_wind_ele*26)/10**6

    #CO2 emission from hydro producing electricity
    hydro_gen_CO2 = (hydro_gen_ele*26)/10**6

    #CO2 emission from biogas producing electricity
    biogas_gen_CO2 = (biogas_gen_ele*45)/10**6

    #Total CO2 emission
    CO2_total = P_pv_CO2 + P_wind_CO2 + hydro_gen_CO2 + biogas_gen_CO2
    print("The total amount of CO2 emitted:", CO2_total)

    # Calculating HDI
    Eexcess = (P_pv_ele-load_total)+(P_wind_ele-load_total)+(hydro_gen_ele-load_total)+(biogas_gen_ele-load_total)
    HDI_total = 0.0978*math.log((total_load+min(0.3*Eexcess,0.5*total_load))/700)-0.0319
    print("HDI =", HDI_total)
    
     # Store important values in session
    request.session['InitialCost_without_diesel'] = InitialCost
    request.session['CO2_total_without_diesel'] = CO2_total
    request.session['HDI_total_without_diesel'] = HDI_total
    request.session['Eexcess_without_diesel'] = Eexcess
    request.session['ACS_without_diesel'] = ACS
    request.session['NPC_without_diesel'] = NPC



    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics
    plt.figure()
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

        # Plot
    # create a single figure with a 2x2 grid of subplots for 7 graphs
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,7))
    # Plot PV 
    axes[0,0].plot(np.arange(No_data), P_pv)
    axes[0,0].set_title('PV generation')
    axes[0,0].set_xlabel('Time (hours)')
    axes[0,0].set_ylabel('electricity generated by PV panel (kWh)')
    # Plot wind
    axes[0,1].plot(np.arange(No_data), p_wind)
    axes[0,1].set_title('wind generation')
    axes[0,1].set_xlabel('Time (hours)')
    axes[0,1].set_ylabel('electricity generates by wind turbine (kWh)')
    # plot hydro
    axes[1,0].plot(np.arange(No_data), hydro_gen)
    axes[1,0].set_title('hydro generation')
    axes[1,0].set_xlabel('Time (hours)')
    axes[1,0].set_ylabel('electricity generated by pico hydro power plants (kWh)')
    # plot biogas
    axes[1,1].plot(np.arange(No_data), biogas_gen)
    axes[1,1].set_title('biogas generation')
    axes[1,1].set_xlabel('Time (hours)')
    axes[1,1].set_ylabel('electricity generated by biogas generator (kWh)')

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Plot load 
    plt.figure()
    plt.plot(np.arange(No_data), load)
    plt.xlabel("Time (hours)")
    plt.ylabel("Load of electricity")
    plt.title("Electricity load")

    # plot LSP
    plt.figure()
    plt.plot(np.arange(No_data), LPS)
    plt.xlabel("Time (hours)")
    plt.ylabel("Loss of power")
    plt.title("LPS")
    # show all the figure
    plt.show()
    
if __name__ == "__main__":
    main()
    
    
    #Pie chart to split each sum for microgrid without diesel
    P_pv_ele = np.sum(P_pv)
    P_wind_ele = np.sum(p_wind)
    hydro_gen_ele = np.sum(hydro_gen)
    biogas_gen_ele = np.sum(biogas_gen)
    
    
    resource_sums = {
    'PV': P_pv_ele,
    'Wind': P_wind_ele,
    'Hydro': hydro_gen_ele,
    'Biogas': biogas_gen_ele,
    }
    labels = resource_sums.keys()
    sizes = resource_sums.values()
    colors = ['darkgreen', 'darkmagenta', 'darkorange', 'darkblue']  
    # Create the pie chart
    plt.figure(figsize=(8, 5))
    plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', textprops={'fontsize': 10})
    plt.title('Electricity Generated From Microgrid without diesel')
    plt.show()
    
    
def figwd1():
         # create a single figure with a 2x2 grid of subplots for 7 graphs
    P_pv_ele = 4.6
    P_wind_ele = 0.3
    hydro_gen_ele = 94.9
    biogas_gen_ele = 0.2
    
    
    resource_sums = {
    'PV': P_pv_ele,
    'Wind': P_wind_ele,
    'Hydro': hydro_gen_ele,
    'Biogas': biogas_gen_ele,
    }
    labels = resource_sums.keys()
    sizes = resource_sums.values()
    colors = ['darkgreen', 'darkmagenta', 'darkorange', 'darkblue']  
    # Create the pie chart
    plt.figure(figsize=(8, 5))
    plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', textprops={'fontsize': 10})
    plt.title('Electricity Generated From Microgrid without diesel')
    plt.show()

        # Adjust layout to prevent overlapping titles and labels
    
    graph = get_graph()
    return graph    
    
def figwd2():
         # Plot load 
    plt.figure()
    plt.plot(np.arange(No_data), load)
    plt.xlabel("Time (hours)")
    plt.ylabel("Load of electricity")
    plt.title("Electricity load")
            
    graph = get_graph()
    return graph 
    
    
def figwd3():
           # plot LSP
    plt.figure()
    plt.plot(np.arange(No_data), LPS)
    plt.xlabel("Time (hours)")
    plt.ylabel("Loss of power")
    plt.title("LPS")
        # show all the figure
    plt.show()
    graph = get_graph()
    return graph 
        
