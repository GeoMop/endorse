from typing import *
import attrs
import numpy as np
from deap import algorithms, tools, creator, base
import scoop
import os
import sys
import subprocess

from endorse.Bukov2 import boreholes


@attrs.define
class BHConf:
    i_angle : Tuple[int, int]
    i_bh_loc: int
    packers: np.ndarray     # 4 positions, values in (0, 1)

Individual = List[BHConf]



# def make_individual(cfg, bh_set:boreholes.BoreholeSet):
#     angles = bh_set.draw_angles(cfg.n_boreholes - 1)
#     angles = [angles[0], *angles]
#     return [
#         make_bh_cfg(a)
#         for a in angle
#     ]
#     common_angle = (np.random.randint(bh_set.n_y_angles), np.random.randint(bh_set.n_z_angles)
#     bh_set.direction_lookup(common_angle)

def make_individual(cfg, bh_set) -> Individual:
    ind = creator.Individual(np.random.randint(0, 1, size=100, dtype='int'))
    return ind

def eval_individual(bh_conf_list : Individual) -> Tuple[float]:
    #sensitivity = 1
    sensitivity = sum(bh_conf_list)
    return sensitivity,

def cross_over(ind1:Individual, ind2:Individual) -> Tuple[Individual, Individual]:
    """
    Mating of ind1 and ind2
    :param ind1:
    :param ind2:
    :return:
    """
    return tools.cxTwoPoint(ind1, ind2)


def mutate(ind: Individual) -> Individual:
    return tools.mutFlipBit(ind, indpb=0.05)

def optimize(cfg, bh_set, map_fn):
    """
    Main optimization routine.
    :return:
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize result of individual evaluation.
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    #toolbox.register("attr_bool", np.random.randint, 0, 1)

    # Structure initializers
    toolbox.register("individual", make_individual, cfg, bh_set)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", cross_over)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3) # ?? is it a good choice
    toolbox.register("map", map_fn)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=200,
                                   stats=stats, halloffame=hof, verbose=True)
    return pop, log


# def test_deap_scoop():
#     hostfile = os.path.abspath("deap_scoop_hostfile")
#     with open(hostfile, "w") as f:
#         f.write("localhost")
#     cmd = [sys.executable, "-m", "scoop", "--hostfile", hostfile, "-vv", "-n", "4", script_path, "100"]
#     subprocess.run(cmd, check=True)

script_path = os.path.abspath(__file__)
pbs_script = f"""
#!/bin/bash
#PBS -j oe
#PBS -m e
set -x 
env | grep PBS_
echo "===="
{sys.executable} -m scoop --hostfile $PBS_NODEFILE -vv -n 4 {script_path} 100
"""


def pbs_submit(queue, n_workers):
    """
    Submit the optimization PBS job.
    :return:
    """
    pbs_filename = "borehole_optimize.pbs"
    with open(pbs_filename, "w") as f:
        f.write(pbs_script)
    cmd = ['qsub', '-q', queue, '-l', f'select={str(n_workers)}:ncpus=1', pbs_filename]
    subprocess.run(cmd, check=True)


######################


def evalOneMax(individual):
    return sum(individual),






def main():
    if len(sys.argv) > 2:
        raise ImportError("Wrong number of program parameters. Give a length of OneMax problem.")

    if len(sys.argv) == 1:
        print("Create Calling metacentrum PBS test.")
        pbs_test_deap_scoop()
        return

    one_max_len = int(sys.argv[1])
    pop, log = one_max(one_max_len, scoop.futures.map)

    out_file = "deap_scoop_out"
    out_file = os.path.abspath(out_file)
    print("Writing results to : ", out_file)
    with open(out_file, "w") as f:
        f.write(str(pop))
        f.write(str(log))


if __name__ == '__main__':
    """
    Direct call implies MPI run with PBS submit.
    """
    main()

