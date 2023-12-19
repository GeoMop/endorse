from typing import *
import attrs
import numpy as np
from deap import algorithms, tools, creator, base
import scoop
import os
import sys
import subprocess
import pickle
from pathlib import Path
from functools import cached_property
import pyvista as pv
script_path = Path(__file__).absolute()

from endorse.Bukov2 import boreholes, sa_problem
from endorse.sa import analyze

"""
Genetic optimization:


"""


@attrs.define
class BHConf:
    @staticmethod
    def from_array( int_array):
        ia_y, ia_z, p1, p2, p3, p4, il = int_array
        return BHConf((ia_y, ia_z), il, (p1, p2, p3, p4))

    i_bh: int
    packers: Tuple[int,int,int,int]     # 4 positions, int values in (0, n_positions)
    size = 7

    def to_array(self):
        return [*self.i_angle, *self.packers, self.i_bh_loc]

Individual = List[BHConf]

class OptSpace:
    """
    - Packer configuration is specific to given borehole, possibly could only be applied to the
      boreholes with similar angle. If we allow exchange of packer config only within the same BH
      or for similar angles. We con significantly decrease probablility of this exchange

    - In order to make data exchange simple, we store every individual as a list of ints.
    - Every borohole configuration is encoded as (borehole_id, *int_packer_positions).
    - float packer position = int_packer_position * packer_resolution, is from interval (-1, 1)
    - interval (-1, 1) is mapped to the point interval on the borhole
    - the points are distributed unevenly, but this quaranties similar meaning of packer configurations, possibly improving
      crossover efficiency
    - During evaluation the packer positions are interpreted as start positions of packers.

    - Only evaluation is distributed, we add one more parameter providing the file with precomputed data of current BHSet.
    - In order to capture certain invariants and prevent generation of equivalent individuals (permutation of boreholes).
      See the `project` method.


    """

    def __init__(self, cfg, bh_set:boreholes.BoreholeSet):
        self.cfg = cfg
        self.bhs = bh_set
        self.bhs_size = self.bhs.n_boreholes
        self.n_boreholes = self.bhs.n_boreholes_to_select
        self.n_packers = cfg.n_packers
        self.packer_resolution = 1000

    def project_decorator(self, func):
        def wrapper(*args, **kwargs):
            offspring = (None)
            while any(ind is None for ind in offspring):
                offspring = func(*args, **kwargs)
                offspring = (self.project_individual(ind) for ind in offspring)
            return offspring
        return wrapper

    def _project(self, individual:Individual):
        """
        Project an individual to canonical reprezentation.
        - return None if the individual is invalid (should not happen frequently)
        - return the individual modified in place

        Applied checks and transforms:
        - sort boreholes according to their ID
        - sort packer positions
        - check chamber sizes -> may lead to invalid individual
        - ind[0] ... copy angle and packers from ind[1]
        -
        """

        # Normalize first two parallel boreholes
        ind_bh = np.array(individual).reshape(self.n_boreholes, -1)
        # sort bh
        indices = np.argsort(ind_bh[:, 0])
        ind_bh = ind_bh[indices]
        # sort and check packers
        for bh_cfg in ind_bh:
            i_bh = bh_cfg[0]
            bh_cfg[1:].sort()
            f_packers = self.float_packers(bh_cfg[1:])
            for pa, pb in zip(f_packers[0:-1], f_packers[1:]):
                assert pb > pa
                if (pb - pa) < self.min_packer_diff(i_bh):
                    return None


        bh_0 = ind_bh[0, 0]
        i,j,k0 = self.bhs.angle_ijk(bh_0)
        i,j,k1 = self.bhs.angle_ijk(ind_bh[1, 0])
        ind_bh[0] = ind_bh[1]
        angle0_bh_list = self.bhs.angles_table[i][j]
        k0 = k0 % len(angle0_bh_list)
        ind_bh[0, 0] = k0
        return ind_bh.ravel().tolist()

    def int_packers(self, float_packers):
        return (np.maximum(-1.0, np.minimum(1.0, float_packers)) * self.packer_resolution).astype(int).tolist()

    def float_packers(self, int_packers):
        return np.array(int_packers).astype(float) / self.packer_resolution

    def random_bh(self):
        return np.random.randint(0, self.bhs_size)

    def random_packers(self):
        return self.int_packers(np.random.randn(self.n_packers) / 3.0)

    def make_individual(self) -> Individual:
        ind = None
        while ind is None:
            ind = [(self.random_bh(), *self.random_packers())  for i in self.n_boreholes]
            ind = self.project(ind)
        return creator.Individual(ind),


    def cross_over(self, ind1:Individual, ind2:Individual) -> Tuple[Individual, Individual]:
        """
        Mating of ind1 and ind2.
        Exchange one BHConf.
        :param ind1:
        :param ind2:
        :return:
        Ideas:
        - exachange whole BH only:
          - would preserve BH - packer compatibility
        """
        ind1, ind2 = tools.cxTwoPoint(ind1, ind2)
        # keep first two boreholes with the same direction and configuration
        ind1[BHConf.size:2*BHConf.size - 1] = ind1[0:BHConf.size - 1]
        ind2[BHConf.size:2*BHConf.size - 1] = ind2[0:BHConf.size - 1]

        # TODO:
        # cross over of
        return ind1, ind2


    def mutate(self, ind: Individual) -> Individual:
        """
        Ideas:
        - mutate packer positions, randomly by single step (would be better for uniform points)
        - mutate every bh index with some propability

        Would be good to have a popoulation of packers for every bh ID. So the optimization would be
        two staged: keeping population of packers for every BH separately

        Or we must improve crossover to by applied to any pair with common BH
        must be parformed by a special population global step.
        Would exchange packers between same BH.
        Other crossover would just exchange bh cfg between individuals.

        :param ind:
        :return:
        """
        ind2 = toolbox.clone(ind)

        return tools.mutFlipBit(ind, indpb=0.05)


# def make_individual(cfg, bh_set:boreholes.BoreholeSet):
#     angles = bh_set.draw_angles(cfg.n_boreholes - 1)
#     angles = [angles[0], *angles]
#     return [
#         make_bh_cfg(a)
#         for a in angle
#     ]
#     common_angle = (np.random.randint(bh_set.n_y_angles), np.random.randint(bh_set.n_z_angles)
#     bh_set.direction_lookup(common_angle)




_bh_set = None
def get_bh_set(f_path):
    global _bh_set
    if _bh_set is None:
        try:
            with open(f_path, "rb") as f:
                _bh_set = pickle.load(f)
        except FileNotFoundError:
            pass
    return _bh_set

def save_bh_data(f_path, bh_set):
    with open(f_path, 'wb') as f:
        pickle.dump(bh_set, f)





def eval_from_chambers_sa(chamber_data, problem):
    # chamber_data (n_chambers, n_times, n_samples)

    n_chambers, n_times, n_samples = chamber_data.shape
    assert n_chambers==3*6
    assert n_times == 10
    assert n_samples == 20 * 16

    ch_data = chamber_data.reshape(-1, n_samples)
    sobol_array = analyze.sobol_vec(ch_data, problem, problem['second_order'])
    max_sobol = analyze.sobol_max(sobol_array)  # max over total indices
    return np.sum(max_sobol[:, 0]), max_sobol



def eval_individual(ind : Individual, bh_data_file: Path) -> Tuple[float, Any]:
    bh_set = get_bh_set(bh_data_file)  #
    bh_field = bh_set._bh_field                  # Bit of hack.
    individual_shape = bh_set._individual_shape  # bigger heck, must be injected by OptSpace
    problem = bh_set._sa_problem

    # We interpret the packer positions as the points directly.
    # The packer size would be fixed with respect to point size.
    packer_size = bh_set.packer_size

    chamber_means = []
    for bh in np.array(ind).reshape(individual_shape):
        i_bh = bh[0]
        packers = bh[1:]
        for begin, end in zip(packers[0:-1],packers[1:]):
            mean_value = (bh_field[i_bh, end] - bh_field[i_bh, begin + packer_size]) / (end - begin - packer_size)
            chamber_means.append(mean_value)

    chamber_means = np.array(chamber_means)

    # Final evaluation
    total_sensitivity, sensitivity_info = eval_from_chambers_sa(chamber_means, problem)

    return total_sensitivity, sensitivity_info

def optimize(cfg, bh_set, map_fn, checkpoint=None):
    """
    Main optimization routine.
    :return:
    """
    opt_space = OptSpace(cfg, bh_set)
    bh_data_file = script_path.parent / "bh_set.pickle"
    get_bh_set(bh_data_file)
    bh_set._individual_shape = (opt_space.n_boreholes, opt_space.n_packers + 1)
    bh_set._sa_problem = sa_problem.sa_dict(cfg.problem)
    save_bh_data(bh_data_file, bh_set)


    checkpoint_freq = 10

    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize result of individual evaluation.
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    #toolbox.register("attr_bool", np.random.randint, 0, 1)

    # Structure initializers
    toolbox.register("individual", opt_space.make_individual)
    toolbox.decorate("individual", opt_space.project_decorator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", opt_space.cross_over)
    toolbox.decorate("mate", opt_space.project_decorator)
    toolbox.register("mutate", opt_space.mutate)
    toolbox.decorate("mutate", opt_space.project_decorator)

    toolbox.register("select", tools.selTournament, tournsize=3) # ?? is it a good choice
    toolbox.register("map", map_fn)

    if checkpoint:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        np.random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
        population = toolbox.population(n=cfg.population_size)
        start_gen = 0
        hof = tools.HallOfFame(maxsize= 10 * cfg.population_size)
        logbook = tools.Logbook()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for gen in range(start_gen, cfg.n_generations):
        population = algorithms.varAnd(population, toolbox,
                                       cxpb=cfg.crossover_probability, mutpb=cfg.mutation_probability)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        outcome = toolbox.map(toolbox.evaluate, invalid_ind, bh_data_file)
        for ind, eval_out in zip(invalid_ind, outcome):
            fitness, eval_info = eval_out
            ind.fitness.values = fitness,
            ind.eval_info = eval_info

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        population = toolbox.select(population, k=len(population))

        if gen % checkpoint_freq == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=np.random.getstate())
            with open("checkpoint_name.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)
    return pop, log


# def test_deap_scoop():
#     hostfile = os.path.abspath("deap_scoop_hostfile")
#     with open(hostfile, "w") as f:
#         f.write("localhost")
#     cmd = [sys.executable, "-m", "scoop", "--hostfile", hostfile, "-vv", "-n", "4", script_path, "100"]
#     subprocess.run(cmd, check=True)

pbs_script = f"""
#!/bin/bash
#PBS -j oe
#PBS -m e
set -x 
env | grep PBS_
echo "===="
{sys.executable} -m scoop --hostfile $PBS_NODEFILE -vv -n 4 {str(script_path)} 100
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

