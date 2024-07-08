from deap import base
from deap import creator
from deap import tools

import random
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns
import elitism

DIMENTIONS = 2
BOUND_LOW, BOUND_UP = -1.25, 1.25

POPULATION_SIZE = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.5
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 30 #retain 30 best individuals even they have lost in the selection, crossover, and mutation
CROWDING_FACTOR = 20
PENALTY_VALUE = 10.0

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# child class
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness= creator.FitnessMin)

def randomFloat(low, up):
    return [random.uniform(l,u) for l, u in zip([low]*DIMENTIONS, [up]* DIMENTIONS)]

toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat) # container, list/ tuple/...
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)  # container, function, n

def simionescu(individual):
    x = individual[0]
    y = individual [1]
    f = 0.1*x*y
    return f,




toolbox.register("evaluate", simionescu)

def feasible(individual):
    x = individual[0]
    y = individual[1]
    return x**2 + y**2 <= (1+0.2* math.cos(8*math.atan(x/y)))**2

toolbox.decorate("evaluate",tools.DeltaPenalty(feasible, PENALTY_VALUE))

toolbox.register("select", tools.selTournament, tournsize = 2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/ DIMENTIONS)

def main():
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

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

if __name__ == "__main__":
    main()