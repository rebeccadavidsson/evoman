"""
Evolutionary Computing Task 1

This script runs .....

Names:
Date:
Group:
"""

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from helpersfunctions import crossover, tournament, limits, evaluate

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

experiment_name = 'specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 10 hidden neurons
dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0



if __name__ == "__main__":
    """DIT KAN ALLEMAAL IN EEN FOR LOOP!"""

    # Print state to log when game begins
    env.state_to_log()

    # Execute random game run and save population
    population_first = np.random.uniform(dom_l, dom_u, size=(n_vars,))
    fitness_first,p,e,t = env.play(pcont=population_first)

    # Make second population to compare later
    population_second = np.random.uniform(dom_l, dom_u, size=(n_vars,))
    fitness_second,p,e,t = env.play(pcont=population_second)

    print(fitness_first, fitness_second)

    # Remember best population
    if fitness_second >= fitness_first:
        population_final = population_second
        fitness_final    = fitness_second
    else:
        population_final = population_first
        fitness_final    = fitness_first

    # Update game with best solution
    solution = [population_final, fitness_final]

    # Perform crossover with better population (function declared in helpersfunctions.py)
    offspring = crossover(population_final)
    print("IK KOM HIER NIET UIT")
    fit_offspring = evaluate(offspring)
    print("EINDELIJK BEN IK ER UIT")

    # Make new population with best population
    new_population = np.vstack((population_final, offspring))
    fit_pop = np.append(fitness_final, fitness_offspring)

    # Get best solution
    best = np.argmax(fit_pop)
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    print(best_sol)
