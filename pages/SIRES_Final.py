from deap import base
from deap import creator
from deap import tools, algorithms

import random
import numpy as np 
import math
import io
import base64

import matplotlib.pyplot as plt
import seaborn as sns

DIMENTIONS = 8                         # the number of variables in fitness function
BOUND_LOW, BOUND_UP = [50,5,1,1,5,1,5,5],[300,10,10,10,15,15,15,15]          # The boundaries for all variables

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.5
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 10                 
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
    return [random.randint (l,u) for l, u in zip([low]*DIMENTIONS, [up]* DIMENTIONS)]


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
    ACS = InitialCost + MaintainCost + ReplaceCost      # minimize the cost = ACS
    #returns ACS and InitialCost (this function is called from views)
    cost = [ACS, InitialCost]
    return ACS

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
G = np.array([0, 0, 0, 0, 0, 0.02, 0.2, 0.4, 0.5, 0.7, 0.9, 0.95, 1, 0.9, 0.7, 0.6, 0.4, 0.2, 0.05, 0, 0, 0, 0, 0])
Ta = np.array([25, 24, 24, 23, 23, 22, 22, 23, 25, 27, 29, 31, 32, 32, 33, 34, 34, 33, 32, 3, 30, 28, 21, 22])

# Wind generation
v = np.array([8.1, 8.5, 9.2, 8.1, 6.9, 4.6, 4.6, 0, 8.1, 6.9, 11.5, 17.3, 16, 15, 18.4, 12.7, 13.8, 10.4, 12.7, 13.8, 12.7, 8.1, 6.9, 10.4])
vin = 2.5
vo = 25
vr = 12
Pr = 600
p_wind = np.zeros(No_data)
wind = np.zeros(No_data)

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

# Battery:
ele = np.array([0.11778, 0.11196, 0.10325, 0.10039, 0.09591, 0.5035, 0.91736, 1.2161, 1.2479, 1.275, 1.2212, 1.3021, 1.3432, 1.3977, 1.5504, 1.5775, 1.7571, 1.7895, 1.8118, 1.8303, 1.7077, 1.4831, 0.37005, 0.10145])
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
n_pump_engine = 0.5
#bio_elec = np.zeros(24)

# Wind water pump
W_wind = np.zeros(No_data)
v = np.array([8.1, 8.5, 9.2, 8.1, 6.9, 4.6, 4.6, 0, 8.1, 6.9, 11.5, 17.3, 16, 15, 18.4, 12.7, 13.8, 10.4, 12.7, 13.8, 12.7, 8.1, 6.9, 10.4])

# PV powered water pump
Q_pv = np.zeros(No_data)
n_pv = 0.7
pw = 998 # kg/m3: the density of water 
g = 9.8

# Water load
W_domestic = np.array([2.7, 1.57, 1.65, 1.35, 1.7, 1.55, 1.37, 1.37, 1.02, 0.67, 0.82, 1.17, 1.52, 1.95, 2.87, 2.9, 3.32, 2.67, 2.15, 2.35, 1.77, 2.15, 2.45, 1.87])
W_irrigation = 80 + 100 * random.uniform(0,1) * np.ones(No_data)
W_load = W_domestic + W_irrigation
W_res = np.zeros(No_data)

# Loss of water supply probability:
LWS = np.zeros(No_data)

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
        load[t] = (15+2*random.uniform(0,1) * ele[t])
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
    return LPSP <= 0.1 and LWSP <= 0.1 #and x_3 <= 3 and x_1 >25
    #return values of LPSP and LWSP (this function is called from views)
    #results = [LPSP, LWSP]
    #return results


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

def main():
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
    print("-- Best Fitness = ", best.fitness.values[0])

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
        load[t] = (15+2*random.uniform(0,1) * ele[t])
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
                W_res[t] = 00
        else:
            if W_res[t-1] + Q_bio[t] + W_wind[t] + Q_pv[t] > W_load[t]:
                W_res[t] = W_res[t-1] + Q_bio[t] + W_wind[t] + Q_pv[t] - W_load[t]
            else:
                W_res[t] =0
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
    InitialCost = InitialCostElec + InitialCostWater
    MaintainCost = (1+f)**n_maintain * (x_1 * CM_PV + x_2 * CM_WT + x_3 * CM_HT + x_4 * CM_BG + x_5 * CM_CB
                               + x_6 * CM_BP + x_7 * CM_WP + x_8 * CM_PP)
    ReplaceCost = SFF * (CR_BG + CR_CB + CR_BP)

    ACS = InitialCost + MaintainCost + ReplaceCost      # minimize the cost = ACS
    
    NPC = ACS/CRF
    print("ACS = ", ACS)
    print(" Initial cost = ", InitialCost)
    print("Net present cost (NPC) = ", NPC) 
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
    plt.show()

  
if __name__ == "__main__":
    main()

def best():
    return bestIND

  
 