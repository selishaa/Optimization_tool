import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from deap import creator, base, tools

import random
import numpy as np 
import math
import pandas as pd 
 
import seaborn as sns


# Define energy types and their uncertainties (standard deviation)
energy_types = ["Electricity", "Solar", "Wind", "Hydropower", "Biogas"]
std_dev = [0.2, 0.2, 0.3, 0.15, 0.2]

# Define base load for each energy type (electricity only has a base load)
base_load = [15, 0, 0, 0, 0]  # Adjust base loads as needed

# Function to generate random energy demands with uncertainties
def generate_demand(base, std):
  return base + norm.rvs(size=1, loc=0, scale=std)[0]

# Perform Monte Carlo simulation with 1000 iterations
iterations = 1000
demands = np.zeros((iterations, len(energy_types)))
for i in range(iterations):
  for j in range(len(energy_types)):
    demands[i, j] = generate_demand(base_load[j], std_dev[j])

# Calculate mean and variance for each energy type
mean_demand = np.mean(demands, axis=0)
var_demand = np.var(demands, axis=0)

# Print results
print("Mean Demand:")
for i, energy in enumerate(energy_types):
  print(f"{energy}: {mean_demand[i]:.2f}")

print("\nVariance:")
for i, energy in enumerate(energy_types):
  print(f"{energy}: {var_demand[i]:.2f}")

# Define cost and reliability functions for each renewable energy
cost = [8, 5, 7, 10]  # Cost per unit of installed capacity (adjust as needed)
reliability = [0.75, 0.8, 0.85, 0.9]  # Probability of meeting base demand (adjust as needed)

# Define objective function (minimize total cost while ensuring reliability)
def objective_function(individual):
  total_cost = np.sum(individual * cost)
  total_reliability = np.prod(individual * reliability)
  if total_reliability < 0.9:  # Penalty for not meeting minimum reliability
    return total_cost + 1000
  else:
    return total_cost

# DEAP framework for genetic algorithm with elitism
creator.create("FitnessMin", base.Fitness, weights=(-1,))  # Minimize objective function
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand, 0.1, 0.8)  # Define gene boundaries (0.1 to 0.8)
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attr_float, len(cost))      # create individuals
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)             # create populationn

toolbox.register("evaluate", objective_function)
toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)  # Gaussian mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

# Implement elitism using variation operator
def elitism(population, ngen, k=1):
  # Select the best individual from the current population
  best_individual = tools.selBest(population, k=1)[0]
  for _ in range(ngen):
    # Perform the usual selection, variation, and evaluation steps
    offspring = toolbox.select(population, k=len(population))
    offspring = list(map(toolbox.clone, offspring))
    for ind in offspring:
      ind.fitness.values = toolbox.evaluate(ind)
    # Replace the worst individual with the elite individual
    worst_idx = sorted(range(len(population)), key=lambda i: population[i].fitness.values)[0]
    population[worst_idx] = toolbox.clone(best_individual)
    # Continue with the next generation
    yield population

# Genetic algorithm parameters
population_size = 10
generations = 10

# Run the genetic algorithm with elitism
population = toolbox.populationCreator(n=population_size)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

ngen = generations  # Avoid modifying generations variable for clarity
for gen in elitism(population, ngen):
  # ... usual loop steps (selection, variation, evaluation, recording statistics)
  tools.check(gen)

# Print results
best_individual = tools.selBest(population, k=1)[0]
print("Best individual:", best_individual)
print("Cost of best solution:", objective_function(best_individual))