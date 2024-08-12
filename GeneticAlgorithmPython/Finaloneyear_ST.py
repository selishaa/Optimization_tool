from deap import base
from deap import creator
from deap import tools, algorithms
from pages import elitism
from pages.utils import get_graph
import openpyxl



import random
import numpy as np 
import math
import pandas as pd 
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import HttpRequest

 
import matplotlib.pyplot as plt
import seaborn as sns
import threading


from django.shortcuts import render
from django.http import HttpRequest


# Need to add 2 days: G (irradiation), Ta(temperature), v(velocity of wind), ele (to calculus load of electricity), water domestic
DIMENTIONS = 8                         # the number of variables in fitness function
BOUND_LOW, BOUND_UP = [50,5,1,1,5,1,5,5],[300,10,10,10,15,15,15,15]  #[25,1,1,1,5,1,5,5],[300,15,10,10,15,20,15,15]    # The boundaries for all variables

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.5
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE= 10
CROWDING_FACTOR = 20
PENALTY_VALUE = 10**6

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
 
# for Water equipment 
CC_BP = 2500 #$/pump : capital cost for biogas powered water pump
CC_WP = 1000 #$/pump : capital cost for wind powered water pump
CC_PP = 6000 #$/pump : capital cost for PV solar powered water pump

# for Other equipment:
CC_BD = 10000 # capital cost for biogas digester (not in fitness function)
CC_RE = 10000 # capital cost for 5 acre-foot resevoir (not in fitness function)


# Maintainance cost 
# for electrical equipment
n_maintain = 25 # the number of years to maintain equiment
CM_PV = 65 #$/year
CM_WT = 95 
CM_HT = 15
CM_BG = 100 
CM_CB = 50
# for water equipment 
CM_BP = 100
CM_WP = 100
CM_PP = 50
# for other
CM_BD = 100
CM_RE = 80

# Replacement cost
# for electrical equipment in 25 years including biogas generator (BG) and battery (CB)
CR_BG = 1000 
CR_CB = 1500
# for water equipment in 25 years including biogas powered water pump (BP)
CR_BP =2500
# for other equipment, we do not need to replace in 25 years: no need


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))                # set minimize condition
creator.create("Individual", list, fitness = creator.FitnessMin)             # create a list of individual that is suitable for minimization 

def randomFloat(low, up):                      # a list of variables that are chosen randomly 
    return [random.randint (l,u) for l, u in zip(low,up)]


toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)              # set the boundaries for the variables in the list
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)      # create individuals
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)             # create population

def minimize(individual):     # define fitness function
    x_1 = individual[0]
    x_2 = individual[1]
    x_3 = individual[2]
    x_4 = individual[3]
    x_5 = individual[4]
    x_6 = individual[5]
    x_7 = individual[6]
    x_8 = individual[7]
    InitialCostElec = CRF * (x_1 *PRATING_PV *CC_PV + x_2 * PRATING_WT * CC_WT + x_3 * PRATING_HT * CC_HT
                         + x_4 *PRATING_BG * CC_BG + x_5 * PRATING_CB * CC_CB)
    InitialCostWater = CRF * (x_6 * CC_BP + x_7 * CC_WP + x_8 * CC_PP)
    InitialCost = InitialCostElec + InitialCostWater
    MaintainCost = (1+f)**n_maintain * (x_1 * CM_PV + x_2 * CM_WT + x_3 * CM_HT + x_4 * CM_BG + x_5 * CM_CB
                               + x_6 * CM_BP + x_7 * CM_WP + x_8 * CM_PP)
    ReplaceCost = SFF * (CR_BG + CR_CB + CR_BP)
    ACS = InitialCost + MaintainCost + ReplaceCost      #  minimize the cost = ACS
    return ACS,

toolbox.register("evaluate", minimize)

# CONSTRAINT 
No_data = 3*24*12
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
G_file ="GeneticAlgorithmPython/stillwaterdata_sample.xlsx"      # excel file's name 
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
ele_file = "GeneticAlgorithmPython/electricity_load_profile_sample.xlsx"
ele_start_row = 1
ele_end_row = 36
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

# Loss of power supple
LPS = np.zeros(No_data)

# Biogas water powered pump
Q_bio = np.zeros(No_data)
Height = 50
n_pump_engine = 0.1
#bio_elec = np.zeros(24)

# Wind water pump (taking the value of wwind velocity above)
W_wind = np.zeros(No_data)

# PV powered water pump
Q_pv = np.zeros(No_data)
n_pv = 0.7
pw = 998 # kg/m3: the density of water 
g = 9.8

#221

nP = 700  # population
hh = 120  # household
hectors = 80
cattle = 450
print("population is:", nP)
print("household is:", hh)
print("hectors is:", hectors)
print("cattle is:", cattle)










    
W_domestic_file = "GeneticAlgorithmPython/water_usage_data_final_sample.xlsx"
W_domestic_s_row = 1
W_domestic_e_row = 864
wb = openpyxl.load_workbook(W_domestic_file)
sheet = wb.active
for cell in sheet[10]:  
    if isinstance(cell.value, (int, float)): 
        cell.value = (cell.value * nP) / 700
wb.save("GeneticAlgorithmPython/water_usage_data_final_sample_updated.xlsx")


W_domestic_data = pd.read_excel("GeneticAlgorithmPython/water_usage_data_final_sample_updated.xlsx", header = None, nrows= W_domestic_e_row - W_domestic_s_row + 1, usecols= [9], skiprows= range(0, W_domestic_s_row))
W_domestic = W_domestic_data.values.flatten()
W_irrigation = (80 + 100 * random.uniform(0,1) * np.ones(No_data))/80*hectors   #/80*hectors
W_cattle = np.ones(No_data) * random.uniform(0.75,1) /24 * 0.1 * cattle
W_load = W_domestic + W_irrigation + W_cattle
W_res = np.zeros(No_data)

    # Loss of water supply probability:
LWS = np.zeros(No_data)

print("outside is running:", LWS)

# constraint function
# Constrain satisfaction
def feasible(individual):
    x_1 = individual[0]
    x_2 = individual[1]
    x_3 = individual[2]
    x_4 = individual[3]
    x_5 = individual[4]
    x_6 = individual[5]
    x_7 = individual[6]
    x_8 = individual[7]
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
    #load
    for t in range(No_data):
        load[t] = (15+2*random.uniform(0,1) * ele[t])*hh/120 
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

    # Biogas water powered pump
    for t in range(No_data):
        bio_elec[t] = 1+ random.uniform(0,1)
        Q_bio[t] = x_6 *(n_pump_engine *bio_elec[t] *5.6*367)/Height
    # Wind water pump
    for t in range(No_data):
        if v[t] >= 3 and v[t] <= 10:
            W_wind[t] = x_7 *3
        elif v[t] >= 10.1 and v[t]<= 16.9:
            W_wind[t] = x_7 * 6
        elif v[t] >=17 and v[t] <= 20:
            W_wind [t] = x_7 * 10
        else:
            W_wind [t] = 0
    # PV powered water pump
    for t in range(No_data):
        Q_pv[t] = x_8* n_pv*3600* G[t] * 0.1/ (pw*g*Height)
    # Water load
    for t in range(No_data):
        if t == 0:
            if Q_bio[t] + W_wind[t] + Q_pv[t] - W_load[t] >0:
                W_res[t] = Q_bio[t] + W_wind[t] + Q_pv[t] - W_load[t]
            else:
                W_res[t] = 0
        else:
            if W_res[t-1] + Q_bio[t] + W_wind[t] + Q_pv[t] > W_load[t]:
                W_res[t] = W_res[t-1] + Q_bio[t] + W_wind[t] + Q_pv[t] - W_load[t]
            else:
                W_res[t] =0
    # Loss of water supply probability
    for t in range(No_data):
        LWS[t] = W_load[t] - (Q_bio[t] + W_wind[t]+ Q_pv[t]+ W_res[t])
        if W_load[t] >Q_bio[t] + W_wind[t]+ Q_pv[t]+ W_res[t]:
            LWS[t] = W_load[t] - (Q_bio[t] + W_wind[t]+ Q_pv[t]+ W_res[t])
        else:
            LWS[t] = 0
        LWSP = np.sum(LWS)/np.sum(W_load)
    return LPSP <= 0.1 and LWSP <= 0.1 # and x_3 <= 3 and x_1 >25


toolbox.decorate("evaluate",tools.DeltaPenalty(feasible,PENALTY_VALUE))     # set penalty value

toolbox.register("select", tools.selTournament, tournsize = 2)              # selection 
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)   # crossover 
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/ DIMENTIONS)  # mutation

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def main(request):
    nP = request.session.get('population', 700)  # Default to 700 if not found
    hh = request.session.get('household', 120)
    hectors = request.session.get('hectors', 80)
    cattle = request.session.get('cattle', 450)

    print("The number of population is inside", nP)
    print("The number of household is inside:", hh)
    print("The number of hectors is inside:", hectors)
    print("The number of cattle is inside:", cattle)


    
   
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen= MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    #PRINT THIS
    print("-- Best Fitness = ", best.fitness.values[0])
    print("main is running",best)
    

    x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8 = best
    
    
    global bestIND
    bestIND = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]
    #PV generation:                      # define all the equations again to print them inside the main()
    for t in range(No_data):
        Tc[t] = Ta[t] + ((NCOT - 20) / 800) * G[t]
        Voc[t] = Voc_stc - (kv * Tc[t])
        Isc[t] = (Isc_stc + (ki * (Tc[t] - 25))) * G[t] / 1000
        i_pv[t] = Ns * Np * Voc[t] * Isc[t] * Pmax / (Vmax * Imax)
        P_pv[t] = x_1 * i_pv[t]/1000
        P_pv_ele = np.sum(P_pv)
        print("P_pv_ele is calculated inside the main", P_pv_ele)
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
    #load
    for t in range(No_data):
        load[t] = (15+2*random.uniform(0,1) * ele[t])
        load_total = np.sum(load)
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

    # Biogas water powered pump
    for t in range(No_data):
        bio_elec[t] = 1+ random.uniform(0,1)
        Q_bio[t] = x_6 *(n_pump_engine *bio_elec[t] *5.6*367)/Height
        Q_bio_pump = np.sum(Q_bio)
    # Wind water pump
    for t in range(No_data):
        if v[t] >= 3 and v[t] <= 10:
            W_wind[t] = x_7 *3
            W_wind_pump = np.sum(W_wind)
        elif v[t] >= 10.1 and v[t]<= 16.9:
            W_wind[t] = x_7 * 6
            W_wind_pump = np.sum(W_wind)
        elif v[t] >=17 and v[t] <= 20:
            W_wind [t] = x_7 * 10
            W_wind_pump = np.sum(W_wind)
        else:
            W_wind [t] = 0
            W_wind_pump = np.sum(W_wind)
    # PV powered water pump
    for t in range(No_data):
        Q_pv[t] = x_8* n_pv*3600* G[t] * 0.1/ (pw*g*Height)
        Q_pv_pump = np.sum(Q_pv)
    # Water load
    for t in range(No_data):
        if t == 0:
            if Q_bio[t] + W_wind[t] + Q_pv[t] - W_load[t] >0:
                W_res[t] = Q_bio[t] + W_wind[t] + Q_pv[t] - W_load[t]
                Load_W_total = np.sum(W_res)
            else:
                W_res[t] = 0
                Load_W_total = np.sum(W_res)
        else:
            if W_res[t-1] + Q_bio[t] + W_wind[t] + Q_pv[t] > W_load[t]:
                W_res[t] = W_res[t-1] + Q_bio[t] + W_wind[t] + Q_pv[t] - W_load[t]
                Load_W_total = np.sum(W_res)
            else:
                W_res[t] =0
                Load_W_total = np.sum(W_res)
    # Loss of water supply probability
    for t in range(No_data):
        LWS[t] = W_load[t] - (Q_bio[t] + W_wind[t]+ Q_pv[t]+ W_res[t])
        if W_load[t] >Q_bio[t] + W_wind[t]+ Q_pv[t]+ W_res[t-1]:
            LWS[t] = W_load[t] - (Q_bio[t] + W_wind[t]+ Q_pv[t]+ W_res[t])
        else:
            LWS[t] = 0
    # Loss of water supply probability
    for t in range(No_data):
        LWS[t] = W_load[t] - (Q_bio[t] + W_wind[t]+ Q_pv[t]+ W_res[t])
        LWSP = np.sum(LWS)/np.sum(W_load)
    
    #PRINT
    #print("LPS =", LPS)
    global LPSP_print 
    LPSP_print = LPSP
    print("LPSP =",LPSP)
    global LWSP_print
    LWSP_print = LWSP
    #print("LWS =",LWS)
    print("LWSP =",LWSP)

    InitialCostElec = CRF * (x_1 *PRATING_PV *CC_PV + x_2 * PRATING_WT * CC_WT + x_3 * PRATING_HT * CC_HT
                         + x_4 *PRATING_BG * CC_BG + x_5 * PRATING_CB * CC_CB)
    InitialCostWater = CRF * (x_6 * CC_BP + x_7 * CC_WP + x_8 * CC_PP)
    InitialCost = InitialCostElec + InitialCostWater # important
    
    MaintainCost = (1+f)**n_maintain * (x_1 * CM_PV + x_2 * CM_WT + x_3 * CM_HT + x_4 * CM_BG + x_5 * CM_CB
                               + x_6 * CM_BP + x_7 * CM_WP + x_8 * CM_PP)
    ReplaceCost = SFF * (CR_BG + CR_CB + CR_BP)
    ACS = InitialCost + MaintainCost + ReplaceCost      # important minimize the cost = ACS
    NPC = ACS/CRF # important
    
    print("ACS = ", ACS)
    print("Initial cost = ", InitialCost)
    print("Net present cost (NPC) = ", NPC) 
    
    # Lifecycle CO2 emission 
    # CO2 emission from PV solar producing electricity
    P_pv_CO2 = (P_pv_ele*85)/10**6

    # CO2 emission from wind producing electricity
    P_wind_CO2 = (P_wind_ele*26)/10**6

    # CO2 emission from hydro producing electricity
    hydro_gen_CO2 = (hydro_gen_ele*26)/10**6

    # CO2 emission from biogas producing electricity
    biogas_gen_CO2 = (biogas_gen_ele*45)/10**6

    # CO2 emission from PV solar water pump
    Q_PV_Wpump = (Q_pv_pump*50)/(.50*367)
    Q_pv_pump_CO2 = (Q_PV_Wpump*85)/10**6

    # CO2 emission from wind water pump
    W_WIND_Wpump = (W_wind_pump*50)/(.50*367)
    W_wind_pump_CO2 = (W_WIND_Wpump*26)/10**6

    # CO2 emission from biogas water pump
    Q_BIO_Wpump = (Q_bio_pump*50)/(.50*367)
    Q_bio_pump_CO2 = (Q_BIO_Wpump*45)/10**6

    # Total CO2 emission
    CO2_total = P_pv_CO2 + P_wind_CO2 + hydro_gen_CO2 + biogas_gen_CO2 + Q_pv_pump_CO2+ W_wind_pump_CO2 + Q_bio_pump_CO2
    print("The total amount of CO2 emitted:", CO2_total) #important

    # Calculating HDI
    ele_Eexcess = (P_pv_ele-load_total)+(P_wind_ele-load_total)+(hydro_gen_ele-load_total)+(biogas_gen_ele-load_total)
    W_Eexcess = (Q_pv_pump-Load_W_total)+(W_wind_pump-Load_W_total)+(Q_bio_pump-Load_W_total)
    Eexcess = ele_Eexcess + W_Eexcess
    HDI_total = 0.0978*math.log(((load_total+Load_W_total)+min(0.3*Eexcess,0.5*(load_total+Load_W_total)))/700)-0.0319
    
    # Store important values in session
    request.session['InitialCost'] = InitialCost
    request.session['CO2_total'] = CO2_total
    request.session['HDI_total'] = HDI_total
    request.session['Eexcess'] = Eexcess
    request.session['ACS'] = ACS
    request.session['NPC'] = NPC
    
    
    print("HDI = ", HDI_total) #important
    print ("Eexcess = ", Eexcess) # important
    print("Electricity Eexcess = ", ele_Eexcess)
    print("Water Eexcess = ", W_Eexcess) 
    
   


    
    
    # extract statistics:
    global minFitnessValues, meanFitnessValues
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # Plot
    # create a single figure with a 4x2 grid of subplots for 7 graphs
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14,7))
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
    axes[0,2].plot(np.arange(No_data), hydro_gen)
    axes[0,2].set_title('hydro generation')
    axes[0,2].set_xlabel('Time (hours)')
    axes[0,2].set_ylabel('electricity generated by pico hydro power plants (kWh)')
    # plot biogas
    axes[0,3].plot(np.arange(No_data), biogas_gen)
    axes[0,3].set_title('biogas generation')
    axes[0,3].set_xlabel('Time (hours)')
    axes[0,3].set_ylabel('electricity generated by biogas generator (kWh)')
    # plot biogas water powered pump
    axes[1,0].plot(np.arange(No_data), Q_bio)
    axes[1,0].set_title('Water pumped by biogas')
    axes[1,0].set_xlabel('Time (hours)')
    axes[1,0].set_ylabel('The amount of water pumped by biogas (m3)')
    # plot wind water powered pump
    axes[1,1].plot(np.arange(No_data), W_wind)
    axes[1,1].set_title('water pumped by wind turbine')
    axes[1,1].set_xlabel('Time (hours)')
    axes[1,1].set_ylabel('The amount of water pumped by wind turbine (m3)')
    # plot PV water powered pump
    axes[1,2].plot(np.arange(No_data), Q_pv)
    axes[1,2].set_title('water pumped by PV')
    axes[1,2].set_xlabel('Time (hours)')
    axes[1,2].set_ylabel('The amount of water pumped by PV solar panels (m3)')
    # Remove the last empty subplots
    fig.delaxes(axes[1, 3])
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

    # plot LWS
    plt.figure()
    plt.plot(np.arange(No_data), LWS)
    plt.xlabel("Time (hours)")
    plt.ylabel("loss of water")
    plt.title("LWS")

    # plot statistics
    plt.figure()
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # show all the figure
    #plt.show()
    
if __name__ == "__main__":
    main()

def bestYear():
    return bestIND

def LPSP():
    return LPSP_print

def LWSP():
    return LWSP_print



def return_graph2():
    # Plot load 
    plt.figure()
    plt.plot(np.arange(No_data), load)
    plt.xlabel("Time (hours)")
    plt.ylabel("Load of electricity")
    plt.title("Electricity load")
    graph = get_graph()
    return graph
    print("graph is running", graph)
    
def return_graph3():
    # plot LSP
    plt.figure()
    plt.plot(np.arange(No_data), LPS)
    plt.xlabel("Time (hours)")
    plt.ylabel("Loss of power")
    plt.title("LPS")
    graph = get_graph()
    return graph

def return_graph4():
    # plot LWS
    plt.figure()
    plt.plot(np.arange(No_data), LWS)
    plt.xlabel("Time (hours)")
    plt.ylabel("loss of water")
    plt.title("LWS")
    graph = get_graph()
    return graph

def return_graph5():
   # plot statistics
    plt.figure()
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations') 
    graph = get_graph()
    return graph

def return_graph6():
    P_pv_ele = np.sum(P_pv)
    P_wind_ele = np.sum(p_wind)
    hydro_gen_ele = np.sum(hydro_gen)
    biogas_gen_ele = np.sum(biogas_gen)
       # Assume these are calculated or fetched from your data source
    #P_pv_ele = 500  # Example value, replace with actual calculation or data fetching
   # P_wind_ele = 300
   # hydro_gen_ele = 400
   # biogas_gen_ele = 200

   
    resource_sums = {
    'PV Electricity': P_pv_ele,
    'Wind Electricity': P_wind_ele,
    'Hydro Electricity': hydro_gen_ele,
    'Biogas Electricity': biogas_gen_ele,
    }
    labels = resource_sums.keys()
    sizes = resource_sums.values()
    colors = ['yellow', 'maroon', 'beige', 'purple']  
    # Create the pie chart
    plt.figure(figsize=(8, 5))
    plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', textprops={'fontsize': 8})
    plt.title('Electricity Generated From Renewable Sources')
    plt.show()
    graph = get_graph()
    return graph

    
def return_graph7():
    Q_bio_pump = np.sum(Q_bio)
    Q_pv_pump = np.sum(Q_pv)
    W_wind_pump = np.sum(W_wind)
  
    
    resource_sums = {
    'Biogas Water Pump': Q_bio_pump,
    'PV Water Pump': Q_pv_pump,
    'Wind Water Pump': W_wind_pump,
    }
    labels = resource_sums.keys()
    sizes = resource_sums.values()
    colors = ['pink', 'cyan', 'grey']  
    # Create the pie chart
    plt.figure(figsize=(8, 5))
    plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', textprops={'fontsize': 8})
    plt.title('Water Pumped From Renewable Sources')
    plt.show()
    graph = get_graph()
    
    
    return graph


