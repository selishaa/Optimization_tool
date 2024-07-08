from deap import base
from deap import creator
from deap import tools, algorithms

import random
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns



DIMENTIONS = 5
BOUND_LOW, BOUND_UP = [100,1,1,1,5],[300,100,100,100,100] 

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.5
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 10
CROWDING_FACTOR = 20
PENALTY_VALUE = 1000000
# Electrical equipment
# Initialization of power rating  for components in kW 
PRATING_PV = 0.1       # Power rating for PV solar of each array 0.1 kW
PRATING_WT = 1         # Power rating for each wind turbine 1kW
PRATING_HT = 1         # Power rating for each hydro turbine 1kW
PRATING_BG = 5         # Power rating for each biogas powered gererator 5kW
PRATING_CB = 24        # Power rating for battery 1kAh, 24V

#Capital cost for each electrical equipment per kWh
CC_PV = 3000 #$/kW
CC_WT = 1800 #$/kW
CC_HT = 2300 #$/kW
CC_BG = 1200 #$/kW
CC_CB = 1500 #$/kAh

r_dash = 0.0375    # nominal interest rate
f = 0.015          # annual inflation rate
n_proj = 25        # lifetime in years 

r = (r_dash- f)/(1+f)
CRF = r * (1+r)**n_proj / ((1+r)**n_proj - 1)

# Maintainance cost 
# for electrical equipment
n = 25 # the number of years to maintain equiment
CM_PV = 65 #$/year
CM_WT = 95 
CM_HT = 15
CM_BG = 100 
CM_CB = 50

# Replacement cost
# for electrical equipment in 25 years including biogas generator (BG) and battery (CB)
n_rep = 8 #the lifetime in year of a equiment
SFF = r/ ((1+r)**n_rep -1)
CR_BG = 1000 
CR_CB = 1500

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)

def randomFloat(low, up):
    return [random.randint (l,u) for l, u in zip(low, up)]


toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

def cost(individual):
    x_1 = individual[0]
    x_2 = individual[1]
    x_3 = individual[2]
    x_4 = individual[3]
    x_5 = individual[4]
    InitialCostElec = CRF * (x_1 *PRATING_PV *CC_PV + x_2 * PRATING_WT * CC_WT + x_3 * PRATING_HT * CC_HT
                         + x_4 *PRATING_BG * CC_BG + x_5 * PRATING_CB * CC_CB)
    MaintainCost = (1+f)**n * (x_1 * CM_PV + x_2 * CM_WT + x_3 * CM_HT + x_4 * CM_BG + x_5 * CM_CB)
    ReplaceCost = SFF * (CR_BG + CR_CB)
    ACS = InitialCostElec + MaintainCost + ReplaceCost
    return ACS,
toolbox.register("evaluate", cost)

# Constraint satisfaction
# PV generation:
Voc_stc = 29.61
Isc_stc = 4.11
Vmax = 23.17
Imax = 4.49
Pmax = 104
NCOT = 43
kv = -0.13
ki = 0.00468
npv = np.random.randint(60, 150)
Voc_stc = 29.61
Isc_stc = 4.11
Vmax = 23.17
Imax = 4.49
Pmax = 104
NCOT = 43
kv = -0.13
ki = 0.00468
npv = np.random.randint(60, 150)
Ns = 56
Np = npv - Ns
Tc = np.zeros(24)
Voc = np.zeros(24)
Isc = np.zeros(24)
P_pv = np.zeros(24)
i_pv = np.zeros(24)
G = np.array([0, 0, 0, 0, 0, 0.02, 0.2, 0.4, 0.5, 0.7, 0.9, 0.95, 1, 0.9, 0.7, 0.6, 0.4, 0.2, 0.05, 0, 0, 0, 0, 0])
Ta = np.array([25, 24, 24, 23, 23, 22, 22, 23, 25, 27, 29, 31, 32, 32, 33, 34, 34, 33, 32, 3, 30, 28, 21, 22])

# Wind generation
v = np.array([8.1, 8.5, 9.2, 8.1, 6.9, 4.6, 4.6, 0, 8.1, 6.9, 11.5, 17.3, 16, 15, 18.4, 12.7, 13.8, 10.4, 12.7, 13.8, 12.7, 8.1, 6.9, 10.4])
vin = 2.5
vo = 25
vr = 12
Pr = 600
p_wind = np.zeros(24)
wind = np.zeros(24)

# Hydro power generation
hydro_gen = np.zeros(24)
hydro_ele = np.array([56.1, 59, 60.8, 61.5, 60.5, 52.6, 31.1, 9.8, 10, 10, 3.4, 3.6, 6.8, 12.2, 23.7, 35.5, 44.2, 45.6, 44.6, 43.3, 45.4, 48.5, 53.5, 58.5])
Hd = 50
n_hydro = 0.7
g = 9.8

# Biogas generation
biogas_gen = np.zeros(24)
n_bio = 0.7
bio_elec = np.array([0, 0, 0, 0, 10, 25, 45, 50, 65, 85, 100, 100, 105, 110, 120, 125, 135, 135, 130, 115, 110, 105, 100, 100])

# Battery:
load = 200000 *np.ones(24)
DOD = 0.8
SOC_min = 1 -DOD
p_bat_total = np.zeros(24)
p_bat = np.zeros(24)
# V_dc = 24
# SOC = np.zeros((24,10))
rated_capacity = 1 # kAh
voltage = 24 #V
p_bat_factor = rated_capacity * voltage # kwh
n_inverter = 0.9 # check the number 

# load
LPS = np.zeros(24) # LPS is loss of power supply

# Constrain satisfaction
def feasible(individual):
    x_1 = individual[0]
    x_2 = individual[1]
    x_3 = individual[2]
    x_4 = individual[3]
    x_5 = individual[4]
    #PV generation:
    for t in range(24):
        Tc[t] = Ta[t] + ((NCOT - 20) / 800) * G[t]
        Voc[t] = Voc_stc - (kv * Tc[t])
        Isc[t] = (Isc_stc + (ki * (Tc[t] - 25))) * G[t] / 1000
        i_pv[t] = Ns * Np * Voc[t] * Isc[t] * Pmax / (Vmax * Imax)
        P_pv[t] = x_1 * i_pv[t]/1000
    # Wind generation:
    for t in range(24):
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
    for t in range(24):
        hydro_gen[t] = x_3* n_hydro * 1 * Hd * g *hydro_ele[t]
    # Biogas generation:
    for t in range(24):
        biogas_gen[t] = x_4 * 5.6 * bio_elec[t]/50

    # Battery:
    for t in range(1, 24):
        if t == 1:
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
    LPS = np.zeros(24)
    for t in range(24):
     # add more condition to the LPS in case the load is loss
       LPS[t] = (load[t]- (p_wind[t]+ P_pv[t] + hydro_gen[t] + biogas_gen[t]+ p_bat[t]))*n_inverter
       if LPS[t] <= 0:
          LPS[t] = 0
       else:
          LPS[t] = (load[t]- (p_wind[t]+ P_pv[t] + hydro_gen[t] + biogas_gen[t]+ p_bat[t]))*n_inverter
    LPSP = np.sum(LPS)/np.sum(load)
    return LPSP <=0.1 and x_3< 10

       
toolbox.decorate("evaluate",tools.DeltaPenalty(feasible,PENALTY_VALUE))

toolbox.register("select", tools.selTournament, tournsize = 2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/ DIMENTIONS)


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

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

if __name__ == "__main__":
    main()