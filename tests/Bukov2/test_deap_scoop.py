# A basic test of deap used with scoop.
# Test the execution of itself through scoop.
import subprocess
import sys
import os
import array
import numpy as np
from deap import algorithms, tools, creator, base
import scoop


def test_deap():
    one_max(100, map)
def test_deap_scoop():
    hostfile=os.path.abspath("deap_scoop_hostfile")
    with open(hostfile, "w") as f:
        f.write("localhost")
    script_path = os.path.abspath(__file__)
    cmd = [sys.executable, "-m", "scoop", "--hostfile", hostfile, "-vv", "-n", "4", script_path, "100"]
    subprocess.run(cmd, check=True)

def evalOneMax(individual):
    return sum(individual),


def one_max(l, map_fn):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", np.random.randint, 0, 1)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, l)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("map", map_fn)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=200,
                                   stats=stats, halloffame=hof, verbose=True)
    return pop, log

def main():
    if len(sys.argv) != 2:
        raise ImportError("Wrong number of program parameters. Give a length of OneMax problem.")
    one_max_len = int(sys.argv[1])
    pop, log = one_max(one_max_len, scoop.futures.map)
    with open("deap_scoop_out", "w") as f:
        f.write(str(pop))
        f.write(str(log))

if __name__ == '__main__':
    main()

