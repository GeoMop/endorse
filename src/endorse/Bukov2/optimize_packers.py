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

from endorse.Bukov2 import boreholes, sa_problem, sample_storage, optimize, bh_chambers
from endorse.sa import analyze
from endorse import common

"""
Genetic optimization:


"""

from scipy.stats import norm




@attrs.define
class PackerConfig:
    packers: np.ndarray     # (n_packers,) int
    sobol_indices: np.ndarray # (n_chambers, n_param, n_sobol_indices) float

    @property
    def n_param(self):
        return self.sobol_indices.shape[1]

    @property
    def param_values(self):
        # Deprecated
        return self.chamber_sensitivity

    @property
    def chamber_sensitivity(self):
        """
        For each chamber and parameter provides total sensitivity Sobol index
        of the chamber pressure with respect to the parameter.
        shape: (n_chambers, n_params)
        """
        return self.sobol_indices[:,:, 0]   # Total sensitivity index

    @property
    def param_sensitivity(self):
        return self.sobol_indices[:,:,0].max(axis=0)

    def __str__(self):
        return f"Packers{self.packers.tolist()} = sens {self.param_sensitivity.tolist()}"
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
    cfg: common.config.dotdict      # cfg.optimize
    min_packer_distance: int
    line_bounds: Tuple[int, int]    # packer index must be in this range: min <= i < max
    line_data: np.ndarray           # (n_points, n_times, n_samples)
    sa_problem: Dict[str, Any]
    chambers: bh_chambers.Chambers

    @staticmethod
    def from_bh_set(workdir, cfg, chambers, bh_set:boreholes.BoreholeSet, i_bh):
        cfg_opt = cfg.optimize
        bh_field = bh_set.project_field(None, None, cached=True)
        min_packer_distance = cfg_opt.packer_size + cfg_opt.min_chamber_size
        sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
        problem = sa_problem.sa_dict(sim_cfg)

        return PackerOptSpace(
            cfg_opt,
            min_packer_distance,
            bh_set.line_bounds[i_bh],
            bh_field[i_bh],
            problem
        )

    @property
    def population_size(self):
        return self.cfg.population_size

    @property
    def n_packers(self):
        return self.cfg.n_packers

    @property
    def packer_size(self):
        return self.cfg.packer_size

    @property
    def n_params(self):
        return self.sa_problem['num_vars']

    def project_decorator(self, func):
        def wrapper(*args, **kwargs):
            offspring = (None,)
            while any(ind is None for ind in offspring):
                offspring = func(*args, **kwargs)
                offspring = [self.project_individual(ind) for ind in offspring]
            return offspring
        return wrapper

    def project_individual(self, packers):
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
        packers.sort()
        if packers[0] < self.line_bounds[0] or packers[-1] >= self.line_bounds[1]:
            return None
        for pa, pb in zip(packers[0:-1], packers[1:]):
            if (pb - pa) < self.min_packer_distance:
                return None
        return packers

    def _make_packers(self) -> Tuple[Individual]:
        ind = None
        while ind is None:
            ind = self.project_individual(np.random.randint(*self.line_bounds, self.n_packers))
        return ind

    def make_individual(self) -> Individual:
        ind = []
        for _ in range(self.n_params):
            ind.extend(self._make_packers())
        return creator.Individual(ind)

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
        #ind1, ind2 = tools.cxTwoPoint(ind1, ind2)

        # TODO:
        # cross over of
        return ind1, ind2


    def mutate(self, ind: Individual) -> Individual:
        """
        Ideas:
        - mutate packer positions, by a normal distr within bounds.
        """
        # for i in range(self.n_params):
        #     # move single packer in every param configuration
        #     i_pack = np.random.randint(0,4)
        #     pack_pos = ind[i * self.n_packers + i_pack]
        #     pack_pos + np.random.randn()
        #
        # ind2 = toolbox.clone(ind)

        return ind,

    def select(self, population, new_size, toolbox):
        """
        Consider population of P individuals of size N configurations for N parameters
        as a single population of size N * P. For every parameter:
        1. random N-groups
        2. select the best in every group
        3. use in P individual of the new population
        :param pop:
        :return:
        """
        # Pairs: (config, fittnes vector)
        # ind.eval_inf shape: (n_params_ind, n_params_sens)
        pairs = [(cfg, cfg_fit) for ind in population for cfg, cfg_fit in zip(ind, ind.eval_info)]
        configs, fit = zip(pairs)

        permutation = np.arange(len(configs))
        np.random.shuffle(permutation)
        configs = configs[permutation]
        fit = configs[permutation]
        fit = fit.reshape(self.n_params, -1, self.n_params)
        ind_max = np.argmax(fit, axis=0)
        configs = np.array(configs, dtype=np.int32).rehape(self.n_params, -1, self.n_packers)
        np.take_along_axis(configs, ind_max) # TODO finish
        return population
        #offspring = [toolbox.clone(ind) for ind in population]





    def eval_from_chambers_sa(self, chamber_data):
        # chamber_data (n_params_ind, n_chambers, n_times, n_samples)

        n_params, n_chambers, n_times, n_samples = chamber_data.shape
        ch_data = chamber_data.reshape(-1, n_samples)
        sobol_array = self.sobol_fn(ch_data, self.sa_problem)
        sobol_array = sobol_array.reshape(n_params, n_chambers, n_times, n_params, -1)# n_chambers
        max_sobol = analyze.sobol_max(sobol_array, axis = 2)  # max over total indices, result shape: (
        fittness = max_sobol[...,0].max(axis=1)    # max over chambers, ST
        # shape (n_param, n_param)
        fittness = np.min(np.diag(fittness))

        return fittness, max_sobol

    def to_array(self, ind):
        return np.array(ind).reshape(self.n_params, self.n_packers)

    def to_individual(self, ind_array):
        return creator.Individual(ind_array.ravel().tolist())

    def eval_individual(self, ind : Individual) -> np.ndarray:
        # We interpret the packer positions as the points directly.
        # The packer size would be fixed with respect to point size.
        packers = self.to_array(ind)   # Packer positions
        # Evaluate chamber sensitivities
        chambers = zip(packers[:, 0:-1].T, packers[:, 1:].T)
        sens_info = self.chambers.eval_chambers(chambers)
        fittness = sens_info[...,0].max(axis=1)    # max over chambers, ST
        # shape (n_param, n_param)
        fittness = np.min(np.diag(fittness))

        return fittness, sens_info



    def optimize(self, sobol_fn = analyze.sobol_vec, checkpoint=None):
        """
        Main optimization routine.
        :return:
        """
        self.sobol_fn = sobol_fn
        checkpoint_freq = 10


        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize result of individual evaluation.
        creator.create("Individual", list, fitness=creator.FitnessMax)
        # Individual is n_params * packer_configuration, i.e. n_params * n_packers  ints
        # These groups are only weakly coupled in order to evaluate the Fitness as a single value
        # and facilitate usage with Hall of Fame and statistics
        # We can easily reassamle better N groups  at the very end.
        # more over we introduce a kind of mixing in the custom selection step.
        # on the other hand the srossover only works on pairs of configurations for a single parameter.


        toolbox = base.Toolbox()

        # Attribute generator
        #toolbox.register("attr_bool", np.random.randint, 0, 1)

        # Structure initializers
        toolbox.register("individual", self.make_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.eval_individual)
        toolbox.register("mate", self.cross_over)
        #toolbox.decorate("mate", self.project_decorator)
        toolbox.register("mutate", self.mutate)
        #toolbox.decorate("mutate", self.project_decorator)

        toolbox.register("select", self.select, toolbox=toolbox)
        toolbox.register("map", map)

        if checkpoint:
            # A file name has been given, then load the data from the file
            with open(checkpoint, "rb") as cp_file:
                cp = pickle.load(cp_file)
            population = cp["population"]
            start_gen = cp["generation"]
            hof = cp["halloffame"]
            logbook = cp["logbook"]
            np.random.set_state(cp["rndstate"])
        else:
            # Start a new evolution
            population = toolbox.population(n=self.population_size)
            start_gen = 0
            hof = tools.HallOfFame(maxsize= 10 * self.population_size)
            logbook = tools.Logbook()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        for gen in range(start_gen, self.cfg.n_generations):
            population = algorithms.varAnd(population, toolbox,
                                           cxpb=self.cfg.crossover_probability, mutpb=self.cfg.mutation_probability)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            outcome = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, eval_out in zip(invalid_ind, outcome):
                fitness, eval_info = eval_out
                ind.fitness.values = fitness,
                ind.eval_info = eval_info

            hof.update(population)
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)

            population = toolbox.select(population, len(population))

            if gen % checkpoint_freq == 0:
                # Fill the dictionary using the dict(key=value[, ...]) constructor
                cp = dict(population=population, generation=gen, halloffame=hof,
                          logbook=logbook, rndstate=np.random.get_state())
                with open("checkpoint_name.pkl", "wb") as cp_file:
                    pickle.dump(cp, cp_file)
        return population, hof, logbook



    def get_best_configs(self, pop, k_best):
        cfg_array = np.stack([np.array(ind).reshape(self.n_params, -1) for ind in pop])  # (n_pop, n_param_ind, n_packers)
        # (n_pop, n_param_configs, n_packers)
        chamber_fitness_array = np.stack([ind.eval_info for ind in pop])  # (n_pop, n_param_sens, n_param_ind, n_chamber, SobolIndices)
        n_pop, n_param_configs, n_chambers, n_params_sens, n_sobol = chamber_fitness_array.shape
        assert n_param_configs == self.n_params
        assert n_params_sens == self.n_params

        # to compare we need fittness of every config, that is n_pop * n_param_configs; wit respect to every param sensitivity
        # fittness is max over chambers, taking i_sobol = 0 ... total sensitivity
        fitness_array = chamber_fitness_array[...,0].max(axis=2)
        # (n_pop, n_param_configs, n_params_sens)
        best_array = []
        for i_param in range(self.n_params):
            i_sorted = np.argsort(fitness_array[:, :, i_param].ravel())[-k_best:]
            cfg_best = cfg_array.reshape(-1, self.n_packers)[i_sorted, :]
            sobol_indices_best = chamber_fitness_array.reshape(-1, n_chambers, self.n_params, n_sobol)
            sobol_indices_best = sobol_indices_best[i_sorted, ...]
            for idx in range(k_best):
                best_array.append(PackerConfig(cfg_best[idx], sobol_indices_best[idx]))
        return best_array

    def get_best_k_groups(self, pop, k_best):
        """

        :param pop:
        :param k_best:
        :return:
        """
        cfg_array = np.stack([np.array(ind).reshape(self.n_params, -1) for ind in pop])  # (n_pop, n_param_ind, n_packers)
        chamber_fitness_array = np.stack([ind.eval_info for ind in pop])  # (n_pop, n_param_sens, n_param_ind, n_chamber)
        chamber_fitness_array = chamber_fitness_array.transpose(1, 0, 2, 3)
        fitness_array = chamber_fitness_array.max(axis=3)
        best_array = []
        for i_param in range(self.n_params):

            i_sorted = np.argsort(fitness_array[i_param,:, :].ravel())[-k_best:]
            for idx in i_sorted:
                packers = cfg_array.reshape(-1, self.n_packers)[idx]
                param_values = chamber_fitness_array.reshape(self.n_params, -1, self.n_packers - 1)[:, idx, :]
                best_array.appand(PackerConfig(packers, param_values))
        return best_array

# def test_deap_scoop():
#     hostfile = os.path.abspath("deap_scoop_hostfile")
#     with open(hostfile, "w") as f:
#         f.write("localhost")
#     cmd = [sys.executable, "-m", "scoop", "--hostfile", hostfile, "-vv", "-n", "4", script_path, "100"]
#     subprocess.run(cmd, check=True)



def optimize_borehole(workdir, cfg, bh_set, i_bh, sobol_fn=analyze.sobol_vec):
    opt_space = PackerOptSpace.from_bh_set(workdir, cfg, bh_set, i_bh)
    population, hof, logbook = opt_space.optimize(sobol_fn=sobol_fn)
    best_cfg = opt_space.get_best_configs(population, k_best=cfg.optimize.n_best_packer_conf)
    # TODO: exctract suitable candidates from the population
    with open(workdir / "logbook.txt", 'w') as f:
        f.write(str(logbook))
    with open(workdir / "hof.txt", 'w') as f:
        f.write(str(hof))
    return best_cfg


######################

def pkl_write(workdir, data, name):
    with open(workdir / name, 'wb') as f:
        pickle.dump(data, f)

def pkl_read(workdir, name):
    try:
        with open(workdir / name, 'rb') as f:
            opt_results = pickle.load(f)
    except Exception:
        opt_results = None
    return opt_results

def memoize(func):
    def wrapper(workdir, *args, **kwargs):
        fname = f"{func.__name__}.pkl"
        val = pkl_read(workdir, fname)
        if val is None:
            val = func(args, kwargs)
            pkl_write(workdir, val, fname)
        return val
    return wrapper


def write_optimization_results(workdir, borehole_optim_configs):
    pkl_write(workdir, borehole_optim_configs, "borhole_optim_configurations.pkl")

def read_optimization_results(workdir):
    pkl_read(workdir, "borhole_optim_configurations.pkl")

# def precompute_bh_configurations(item):
#     cfg_file, i_bh = item
#     wc = load(cfg_file)
#     evals = pkl_read(wc, f"eval_b_configs_{i_bh:3d}.pkl")
#     if evals is None:
#
#     return evals
#     bh_set = borehole_set(*wc)


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
        bh_set.load_data(workdir, cfg)

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

