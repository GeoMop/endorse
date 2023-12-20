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

from endorse.Bukov2 import boreholes, sa_problem, sample_storage
from endorse.sa import analyze
from endorse import common

"""
Genetic optimization:


"""

@attrs.define
class PackerConfig:
    packers: np.ndarray     # (n_packers,) int
    param_values: np.ndarray  # (n_chambers, n_params,) float


Individual = List[int]

@attrs.define(slots=False)
class PackerOptSpace:
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
    n_packers: int
    packer_size: int
    min_packer_distance: int
    line_bounds: Tuple[int, int]    # packer index must be in this range: min <= i < max
    line_data: np.ndarray

    @staticmethod
    def from_bh_set(cfg, bh_set:boreholes.BoreholeSet, i_bh):
        bh_field = bh_set.project_field(None, None, cached=True)
        min_packer_distance = cfg.packer_size + cfg.min_chamber_size
        PackerOptSpace(
            cfg.n_packers,
            cfg.packer_size,
            min_packer_distance,
            bh_set.line_bounds[i_bh],
            bh_field[i_bh]
        )



    def project_decorator(self, func):
        def wrapper(*args, **kwargs):
            offspring = (None,)
            while any(ind is None for ind in offspring):
                offspring = func(*args, **kwargs)
                offspring = (self.project_individual(ind) for ind in offspring)
            return offspring
        return wrapper

    def project_individual(self, ind:Individual):
        """
        Project an individual to canonical reprezentation.
        - return None if the individual is invalid (should not happen frequently)
        - return the individual modified in place

        Applied checks and transforms:
        - sort packer positions
        - check packer bounds
        - check chamber sizes -> may lead to invalid individual

        TODO: try with assert, we can design mating and mutation to preserve these
        invariants
        """
        # sort and check distance of packers
        ind.sort()
        if ind[0] < self.line_bounds[0] or ind[-1] >= self.line_bounds[1]:
            return None
        for pa, pb in zip(ind[0:-1], ind[1:]):
            if (pb - pa) < self.min_packer_distance:
                return None
        return ind

    def make_individual(self) -> Tuple[Individual]:
        ind = np.random.randint(*self.line_bounds, self.n_packers)
        # TODO: Apply projection manually as we expect to aviod projection
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



def optimize_borehole(workdir, cfg, bh_set, i_bh):
    opt_space = PackerOptSpace.from_bh_set(cfg, bh_set, i_bh)



######################


def evalOneMax(individual):
    return sum(individual),


def load(cfg_file):
    workdir = cfg_file.parent
    cfg = common.config.load_config(cfg_file)
    return workdir, cfg

def optimize_borehole_wrapper(item):
    cfg_file, i_bh = item
    wc = load(cfg_file)
    bh_set = borehole_set(*wc)
    return optimize_borehole(*wc, bh_set, i_bh)

def borehole_set(workdir, cfg):
    # Prepare BoreholeSet with simulation data loaded
    force = cfg.boreholes.force
    bh_set_file = workdir / "bh_set.pickle"
    bh_set = optimize.get_bh_set(bh_set_file)
    if force or bh_set is None:
        bh_set = boreholes.BoreholeSet.from_cfg(cfg.boreholes.zk_30)
        mesh = boreholes.get_clear_mesh(workdir / cfg.sim.mesh)
        field_samples = sample_storage.get_hdf5_field(workdir / cfg.sim.hdf)
        bh_set.project_field(mesh, field_samples, cached=True)
    optimize.save_bh_data(workdir / "bh_set.pickle", bh_set)
    return bh_set


def optimize_bh_set(cfg_file, map_fn):
    """
    Provides optimized packer positions for each borehole.

    :return: borehole_id ->  List[PackerConfig]
    """

    # prepare and serialize BH with dataset
    bh_set = borehole_set(*load(cfg_file))
    data = [(cfg_file, i_bh) for i_bh in range(bh_set.n_boreholes)]
    opt_bh_packers = map_fn(optimize_borehole_wrapper, data) # (n_boreholes, n_prameters, n_varaints)
    return opt_bh_packers


def main_scoop(cfg_file):
    return optimize_bh_set(cfg_file, scoop.futures.map)


pbs_script_template = """
#!/bin/bash
#PBS -j oe
#PBS -m e
set -x 
env | grep PBS_
echo "===="
{python} -m scoop --hostfile $PBS_NODEFILE -vv -n {n_workers} {str(script_path)} {cfg_file}
"""
def pbs_submit(cfg_file):
    """
    Submit the optimization PBS job.
    :return:
    """
    cfg = common.config.load_config(cfg_file)
    n_workers = cfg.pbs.n_workers
    queue = cfg.pbs.queue
    pbs_filename = "borehole_optimize.pbs"
    with open(pbs_filename, "w") as f:
        pbs_script = pbs_script_template.format(dict(python=sys.executable, n_workers=n_workers, script_path=script_path, cfg_file=cfg_file))
        f.write(pbs_script)
    cmd = ['qsub', '-q', queue, '-l', f'select={str(n_workers)}:ncpus=1', pbs_filename]
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'submit':
        print("Create Calling metacentrum PBS test.")
        pbs_submit('charon', int(sys.argv[2]))
        return

    if len(sys.argv) == 2:
        main_scoop(sys.argv[1])

    raise ImportError("Wrong number of program parameters. Give a length of OneMax problem.")


if __name__ == '__main__':
    """
    Direct call implies MPI run with PBS submit.
    """
    main()

