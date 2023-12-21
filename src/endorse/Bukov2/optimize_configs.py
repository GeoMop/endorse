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
workdir = Path('/home/stebel/dev/git/endorse-experiment/tests/Bukov2')
bh_data_file = workdir / "bh_set.pickle"

from endorse.Bukov2 import boreholes, sa_problem
from endorse import common
from endorse.sa import analyze
import endorse.Bukov2.optimize_packers as opt_pack

"""
Genetic optimization:
Find set of N(=6) boreholes with given packer configurations that maximize the objective function, which is given as

    min_i max_j (sensitivity of i-th parameter in j-th chamber).

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

    def __init__(self, cfg, bh_set:boreholes.BoreholeSet, bh_opt):
        self.cfg = cfg
        self.bhs = bh_set
        self.bhs_size = self.bhs.n_boreholes
        self.n_boreholes = cfg.boreholes.zk_30.n_boreholes_to_select
        self.n_configs = len(bh_opt[0])
        self.n_chambers = cfg.optimize.n_packers-1
        self.n_params = bh_opt[0][0].param_values.shape[1]
        self._individual_shape = (self.n_boreholes, 2)

    def project_decorator(self, func):
        def wrapper(*args, **kwargs):
            offspring = (None,)
            while any(ind is None for ind in offspring):
                offspring = func(*args, **kwargs)
                offspring = (self.project_individual(ind) for ind in offspring)
            return offspring
        return wrapper

    def project_individual(self, individual:Individual):
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
            i_packers = bh_cfg[1:]
            for pa, pb in zip(i_packers[0:-1], i_packers[1:]):
                if (pb - pa) < self.min_packer_distance:
                    return None


        bh_0 = ind_bh[0, 0]
        i,j,k0 = self.bhs.angle_ijk(bh_0)
        i,j,k1 = self.bhs.angle_ijk(ind_bh[1, 0])
        ind_bh[0] = ind_bh[1]
        angle0_bh_list = self.bhs.angles_table[i][j]
        k0 = k0 % len(angle0_bh_list)
        ind_bh[0, 0] = k0
        return ind_bh.ravel().tolist()

    # def int_packers(self, float_packers):
    #     return (np.maximum(-1.0, np.minimum(1.0, float_packers)) * self.packer_resolution).astype(int).tolist()
    #
    # def float_packers(self, int_packers):
    #     return np.array(int_packers).astype(float) / self.packer_resolution

    def random_bh(self):
        i_bh = np.random.randint(0, self.bhs_size)
        i_cfg = np.random.randint(0, self.n_configs)
        return i_bh, i_cfg

    def make_individual(self) -> Tuple[Individual]:
        ind = [self.random_bh()  for i in range(self.n_boreholes)]
        print(creator.Individual(ind))
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
        # ind1, ind2 = tools.cxTwoPoint(ind1, ind2)
        # # keep first two boreholes with the same direction and configuration
        # ind1[BHConf.size:2*BHConf.size - 1] = ind1[0:BHConf.size - 1]
        # ind2[BHConf.size:2*BHConf.size - 1] = ind2[0:BHConf.size - 1]

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
        # ind2 = toolbox.clone(ind)

        # return tools.mutFlipBit(ind, indpb=0.05)
        i = np.random.randint(self.n_boreholes)
        ind[i] = self.random_bh()
        return ind,


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


_bhs_opt_config = None

def get_opt_results(f_path):
    global _bhs_opt_config
    if _bhs_opt_config is None:
        try:
            _bhs_opt_config = opt_pack.read_optimization_results(f_path)
        except FileNotFoundError:
            pass
    return _bhs_opt_config


_opt_space = None

def get_opt_space():
    global _opt_space
    if _opt_space is None:
        try:
            cfg_file = workdir / "Bukov2_mesh.yaml"
            cfg = common.config.load_config(cfg_file)
            _opt_space = OptSpace(cfg, _bh_set, _bhs_opt_config)
        except FileNotFoundError:
            pass
    return _opt_space


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


# def get_bhs_opt_config():
#     cfg = np.empty((10,2), dtype=opt_pack.PackerConfig)
#     for i in range(10):
#         for j in range(2):
#             pkrs = np.arange(4)
#             vals = np.arange(8).reshape(4,2) #+ i*2+j
#             pc = opt_pack.PackerConfig(pkrs, vals)

# bhs_opt_config = get_bhs_opt_config()  # dummy getter of sensitivity data as 2d array of packerConfig's
def eval_individual(ind:Individual) -> Tuple[float, Any]:

    get_bh_set(bh_data_file)
    get_opt_results(workdir)
    get_opt_space()

    # constants below should be replaced from elsewhere
    N_bhs = _opt_space.n_boreholes  # number of boreholes in Individual
    # N_params = bhs_opt_config[0][0].sobol_indices.shape[1] # number of sensitivity parameters
    N_params = _opt_space.n_params  # number of sensitivity parameters

    param_max = np.zeros((N_params, N_bhs))
    i = 0
    for bh in np.array(ind).reshape(N_bhs,2):
        i_bh = bh[0]
        i_cfg = bh[1]
        cfg = _bhs_opt_config[i_bh][i_cfg]
        # print(cfg.param_values)
        param_max[:,i] = np.max(cfg.param_values, axis=0) # axis=0 fixes param and maximizes over chambers
        i = i + 1

    return np.min( np.max(param_max, axis=1) ) # axis=1 fixes param and maximizes over borehole maximums


def optimize(cfg, map_fn, checkpoint=None):
    """
    Main optimization routine.
    :return:
    """

    get_bh_set(bh_data_file)
    get_opt_results(workdir)
    get_opt_space()

    # _bh_set._sa_problem = sa_problem.sa_dict(cfg.problem)
    # save_bh_data(bh_data_file, bh_set)

    checkpoint_freq = 10

    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize result of individual evaluation.
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    #toolbox.register("attr_bool", np.random.randint, 0, 1)

    # Structure initializers
    toolbox.register("individual", _opt_space.make_individual)
    # toolbox.decorate("individual", opt_space.project_decorator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", _opt_space.cross_over)
    # toolbox.decorate("mate", opt_space.project_decorator)
    toolbox.register("mutate", _opt_space.mutate)
    # toolbox.decorate("mutate", opt_space.project_decorator)

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
        population = toolbox.population(n=cfg.optimize.population_size)
        start_gen = 0
        hof = tools.HallOfFame(maxsize= 10 * cfg.optimize.population_size)
        logbook = tools.Logbook()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for gen in range(start_gen, cfg.optimize.n_generations):
        population = algorithms.varAnd(population, toolbox,
                                       cxpb=cfg.optimize.crossover_probability, mutpb=cfg.optimize.mutation_probability)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        outcome = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, eval_out in zip(invalid_ind, outcome):
        #     fitness, eval_info = eval_out
            ind.fitness.values = eval_out,
        #     ind.eval_info = eval_info

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        population = toolbox.select(population, k=len(population))

        if gen % checkpoint_freq == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=hof,
                      logbook=logbook, rndstate=np.random.get_state())
            with open("checkpoint_name.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)
    print(cp)
    return #pop, log


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




def main():
    if len(sys.argv) > 1:
        raise ImportError("Wrong number of program parameters.")

    cfg_file = workdir / "Bukov2_mesh.yaml"
    cfg = common.config.load_config(cfg_file)

    optimize(cfg, map)

    # out_file = "deap_scoop_out"
    # out_file = os.path.abspath(out_file)
    # print("Writing results to : ", out_file)
    # with open(out_file, "w") as f:
    #     f.write(str(pop))
    #     f.write(str(log))


if __name__ == '__main__':
    """
    Direct call implies MPI run with PBS submit.
    """
    main()

