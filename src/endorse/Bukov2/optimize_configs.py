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
workdir = Path('/home/stebel/dev/git/endorse-experiment/tests/Bukov2/PE_01_02_zk30_32')
bh_data_file = workdir / "all_bh_configs.pkl"
cfg_file = workdir / "Bukov2_mesh.yaml"

from endorse.Bukov2 import boreholes, plot_boreholes, sa_problem, process_bh
import endorse.Bukov2.bukov_common as bcommon
from endorse import common
from endorse.sa import analyze
import endorse.Bukov2.optimize_packers as opt_pack
from endorse.Bukov2.bukov_common import pkl_read
from copy import deepcopy

"""
Genetic optimization:
Find set of N(=6) boreholes with given packer configurations that maximize the objective function.

The following variants of objective function are provided:
* eval_individual_max: min_p max_b max_c (sensitivity of p-th parameter in c-th chamber of b-th boreholes).
    This function prefers individuals, where all parameters have the highest possible sensitivity in some chamber
     of some borehole. It does not detect individuals with low-sensitivity configs.
    
* eval_individual_l1: sum_b min_p max_c ( sensitivity of p-th parameter in c-th chamber of b-th boreholes).
    This function sums over boreholes the maximal parameter sensitivities, thus preferring individuals with all configs
    highly sensitive.

"""


Individual = List[tuple]

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
    param_norms: np.ndarray # normalized values of parameter sensitivities

    def __init__(self, cfg, bh_opt):
        # bh_opt is List[n_boreholes] of List[n_params] of List[n_configs] of PackerConfig
        self.cfg = cfg
        self.bhs_size = len(bh_opt)
        self.n_boreholes = cfg.boreholes.common.n_boreholes_to_select
        self.n_configs = len(bh_opt[0][0])
        self.n_chambers = cfg.optimize.n_packers-1
        self.n_params = len(bh_opt[0])
        self._individual_shape = (self.n_boreholes, 2)

        param_sensitivities = np.array([np.sum(c.chamber_sensitivity, axis=0) for b in bh_opt for p in b for c in p])
        self.param_norms = np.quantile(param_sensitivities, 0.95, axis=0)

    def project_decorator(self, func):
        def wrapper(*args, **kwargs):
            offspring = (None,)
            while any(ind is None for ind in offspring):
                offspring = func(*args, **kwargs)
                if type(offspring) is tuple:
                    for o in offspring:
                        o.sort()
                        o = self.rm_duplicate_bhs(o)
                else:
                    offspring.sort()
                    offspring = self.rm_duplicate_bhs(offspring)
            return offspring
        return wrapper

    # def project_individual(self, individual:Individual):
    #     """
    #     Project an individual to canonical reprezentation.
    #     - return None if the individual is invalid (should not happen frequently)
    #     - return the individual modified in place
    #
    #     Applied checks and transforms:
    #     - sort boreholes according to their ID
    #     - sort packer positions
    #     - check chamber sizes -> may lead to invalid individual
    #     - ind[0] ... copy angle and packers from ind[1]
    #     -
    #     """

        # # Normalize first two parallel boreholes
        # ind_bh = np.array(individual).reshape(self.n_boreholes, -1)
        # # sort bh
        # indices = np.argsort(ind_bh[:, 0])
        # ind_bh = ind_bh[indices]
        # # sort and check packers
        # for bh_cfg in ind_bh:
        #     i_bh = bh_cfg[0]
        #     bh_cfg[1:].sort()
        #     i_packers = bh_cfg[1:]
        #     for pa, pb in zip(i_packers[0:-1], i_packers[1:]):
        #         if (pb - pa) < self.min_packer_distance:
        #             return None
        #
        #
        # bh_0 = ind_bh[0, 0]
        # i,j,k0 = self.bhs.angle_ijk(bh_0)
        # i,j,k1 = self.bhs.angle_ijk(ind_bh[1, 0])
        # ind_bh[0] = ind_bh[1]
        # angle0_bh_list = self.bhs.angles_table[i][j]
        # k0 = k0 % len(angle0_bh_list)
        # ind_bh[0, 0] = k0
        # return ind_bh.ravel().tolist()
        # return individual[0].sort()

    # def int_packers(self, float_packers):
    #     return (np.maximum(-1.0, np.minimum(1.0, float_packers)) * self.packer_resolution).astype(int).tolist()
    #
    # def float_packers(self, int_packers):
    #     return np.array(int_packers).astype(float) / self.packer_resolution

    def random_bh(self):
        i_bh = np.random.randint(0, self.bhs_size)
        i_prm = np.random.randint(0, self.n_params)
        i_cfg = np.random.randint(0, self.n_configs)
        return i_bh, i_prm, i_cfg

    def make_individual(self) -> Tuple[Individual]:
        ind = []
        seen = set()
        for i in range(self.n_boreholes):
            t = self.random_bh()
            while t[0] in seen:
                t = self.random_bh()
            seen.add(t[0])
            ind.append(t)
        return creator.Individual(ind)

    def rm_duplicate_bhs(self, ind:Individual) -> Tuple[Individual]:
        seen = set()
        i = 0
        for t in ind:
            while t[0] in seen:
                t = self.random_bh()
            ind[i] = t
            seen.add(t[0])
            i = i + 1
        return ind


    def cross_over(self, ind1:Individual, ind2:Individual) -> Tuple[Individual, Individual]:
        """
        Mating of ind1 and ind2.
        Exchange one borehole configuration.
        :param ind1:
        :param ind2:
        :return:
        """
        i = np.random.randint(self.n_boreholes)
        tup = ind1[i]
        ind1[i] = ind2[i]
        ind2[i] = tup
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

        # replace a random borehole or configuration by another random one
        i = np.random.randint(self.n_boreholes)
        j = np.random.randint(3)
        bh, prm, cfg = ind[i]
        if j == 0:
            bh = np.random.randint(self.bhs_size)
        elif j == 1:
            prm = np.random.randint(self.n_params)
        else:
            cfg = np.random.randint(self.n_configs)
        ind[i] = (bh,prm,cfg)
        return ind,

    def similar(self, ind: Individual, hofer: Individual):
        if [i[0] for i in ind] == [i[0] for i in hofer]:
            return True
        else:
            return False


def export_vtk_optim_set(ind: Individual, fname, plot=False):
    wdir, cfg = bcommon.load_cfg(cfg_file)
    sim_cfg = common.load_config(wdir / cfg.simulation.cfg)

    # save boreholes
    bh_set = boreholes.make_borehole_set(wdir, cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    param_names = problem['names']
    packer_indices = [_bhs_opt_config[i[0]][i[1]][i[2]].packers for i in ind]
    all_packer_coords, bounds = bh_set.points(cfg)
    packer_coords = [[(all_packer_coords[i[0],pi,0]-bh_set.boreholes[i[0]].start[0])/bh_set.boreholes[i[0]].unit_direction[0] for pi in pids] for (pids,i) in zip(packer_indices,ind)]
    sensitivities = [_bhs_opt_config[i[0]][i[1]][i[2]].param_values for i in ind]
    chamber_data = [packer_coords, sensitivities, param_names]
    plot_boreholes.export_vtk_bh_set(wdir, bh_set.subset([i[0] for i in ind]), chamber_data=chamber_data, fname=fname)

    # save tunnel and cylinders
    cylinders = plot_boreholes._make_cylinders(bh_set.lateral)
    tunnel = plot_boreholes._make_main_tunnel(cfg.geometry.main_tunnel)
    scene = list(cylinders)
    scene.append(tunnel)
    scene = pv.merge(scene, merge_points=False)
    scene.save(wdir / "scene.vtk")

    if plot:
        plotter = pv.Plotter(off_screen=False)
        plotter.add_mesh(scene, opacity=0.1) # = plot_boreholes.create_scene(plotter, cfg.geometry)
        # plot_boreholes.add_cylinders(plotter, bh_set)
        # plot_bh_set(plotter, bh_set)
        # for i in ind:
        #     add_bh(plotter, bh_set, i[0])
        plotter.add_mesh( plot_boreholes.make_mesh_bh_set(bh_set.subset([i[0] for i in ind]), chamber_data=chamber_data) )
        plotter.camera.parallel_projection = True
        plotter.show()


def export_vtk_bh_chamber_set(bh_pk_ids, fname, plot=False):
    # bh_pk_ids = list of tuples (borehole ID (str), list of packer positions)
    wdir, cfg = bcommon.load_cfg(cfg_file)
    sim_cfg = common.load_config(wdir / cfg.simulation.cfg)

    # save boreholes
    bh_set = boreholes.make_borehole_set(wdir, cfg)
    bh_ids = [bi for (bi,pi) in bh_pk_ids]
    bh_indices = [idx for idx, bh in enumerate(bh_set.boreholes) if bh.id in bh_ids]
    problem = sa_problem.sa_dict(sim_cfg)
    param_names = problem['names']
    all_packer_coords, bounds = bh_set.points(cfg)
    packer_coords = [[(all_packer_coords[bi,pi,0]-bh_set.boreholes[bi].start[0])/bh_set.boreholes[bi].unit_direction[0] for pi in pids] for (bi,(bid,pids)) in zip(bh_indices,bh_pk_ids)]
    sensitivities = []
    for (bi,(bid,pids)) in zip(bh_indices,bh_pk_ids):
        bh_workdir = process_bh.borehole_dir(workdir, bi)
        index, data = process_bh._chamber_sensitivities(bh_workdir, cfg, chambers=None)
        bh_sens = [data[index[pids[i],pids[i+1]],:] for i in range(len(pids)-1)]
        sensitivities.append(bh_sens)
    chamber_data = [packer_coords, sensitivities, param_names]
    plot_boreholes.export_vtk_bh_set(wdir, bh_set.subset(bh_indices), chamber_data=chamber_data, fname=fname)

    # save tunnel and cylinders
    cylinders = plot_boreholes._make_cylinders(bh_set.lateral)
    tunnel = plot_boreholes._make_main_tunnel(cfg.geometry.main_tunnel)
    scene = list(cylinders)
    scene.append(tunnel)
    scene = pv.merge(scene, merge_points=False)
    scene.save(wdir / "scene.vtk")

    if plot:
        plotter = pv.Plotter(off_screen=False)
        plotter.add_mesh(scene, opacity=0.1) # = plot_boreholes.create_scene(plotter, cfg.geometry)
        # plot_boreholes.add_cylinders(plotter, bh_set)
        # plot_bh_set(plotter, bh_set)
        # for i in ind:
        #     add_bh(plotter, bh_set, i[0])
        plotter.add_mesh( plot_boreholes.make_mesh_bh_set(bh_set.subset(bh_indices), chamber_data=chamber_data) )
        plotter.camera.parallel_projection = True
        plotter.show()



_bhs_opt_config = None


def get_opt_results(f_path):
    global _bhs_opt_config
    if _bhs_opt_config is None:
        try:
            _bhs_opt_config = pkl_read(f_path, bh_data_file) #opt_pack.read_optimization_results(f_path)
        except FileNotFoundError:
            pass
    return _bhs_opt_config


_opt_space = None

def get_opt_space():
    global _opt_space
    if _opt_space is None:
        try:
            cfg = common.config.load_config(cfg_file)
            _opt_space = OptSpace(cfg, _bhs_opt_config)
        except FileNotFoundError:
            pass
    return _opt_space



def eval_individual_max(ind:Individual) -> Tuple[float, Any]:

    get_opt_results(workdir)
    get_opt_space()

    N_bhs = _opt_space.n_boreholes  # number of boreholes in Individual
    N_params = _opt_space.n_params  # number of sensitivity parameters

    param_max = np.zeros((N_params, N_bhs))
    i = 0
    for bh in np.array(ind):
        i_bh = bh[0]
        i_prm = bh[1]
        i_cfg = bh[2]
        cfg = _bhs_opt_config[i_bh][i_prm][i_cfg]
        param_max[:, i] = np.max(cfg.param_values, axis=0)/_opt_space.param_norms # axis=0 fixes param and maximizes over chambers
        i = i + 1
    return np.min( np.max(param_max, axis=1) ) # axis=1 fixes param and maximizes over borehole maximums


def eval_individual_l1(ind:Individual) -> Tuple[float, Any]:

    get_opt_results(workdir)
    get_opt_space()

    N_bhs = _opt_space.n_boreholes  # number of boreholes in Individual
    N_params = _opt_space.n_params  # number of sensitivity parameters

    param_max = np.zeros((N_params, N_bhs))
    i = 0
    for bh in np.array(ind):
        i_bh = bh[0]
        i_prm = bh[1]
        i_cfg = bh[2]
        cfg = _bhs_opt_config[i_bh][i_prm][i_cfg]
        param_max[:, i] = np.max(cfg.param_values, axis=0)/_opt_space.param_norms # axis=0 fixes param and maximizes over chambers
        i = i + 1
    return np.sum( np.min(param_max, axis=0) ) # axis=0 fixes borehole and minimizes over params


def eval_individual_max_l1(ind:Individual) -> Tuple[float, Any]:

    get_opt_results(workdir)
    get_opt_space()

    N_bhs = _opt_space.n_boreholes  # number of boreholes in Individual
    N_params = _opt_space.n_params  # number of sensitivity parameters

    param_max = np.zeros((N_params, N_bhs))
    i = 0
    for bh in np.array(ind):
        i_bh = bh[0]
        i_prm = bh[1]
        i_cfg = bh[2]
        cfg = _bhs_opt_config[i_bh][i_prm][i_cfg]
        param_max[:, i] = np.max(cfg.param_values, axis=0)/_opt_space.param_norms # axis=0 fixes param and maximizes over chambers
        i = i + 1

    max_over_ind = np.max(param_max, axis=1)
    min_over_par = np.min(param_max, axis=0)
    return np.min( max_over_ind ) + 0.2*np.sum( min_over_par )  # coefficient fitted to balance both values


def optimize(cfg, map_fn, eval_fn, checkpoint=None):
    """
    Main optimization routine.
    :return:
    """

    get_opt_results(workdir) #, randomize=True) # True means generate random sensitivities
    get_opt_space()

    checkpoint_freq = 10

    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize result of individual evaluation.
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    #toolbox.register("attr_bool", np.random.randint, 0, 1)

    # Structure initializers
    toolbox.register("individual", _opt_space.make_individual)
    toolbox.decorate("individual", _opt_space.project_decorator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_fn)
    toolbox.register("mate", _opt_space.cross_over)
    toolbox.decorate("mate", _opt_space.project_decorator)
    toolbox.register("mutate", _opt_space.mutate)
    toolbox.decorate("mutate", _opt_space.project_decorator)

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
        hof = tools.HallOfFame(maxsize = 10, similar=_opt_space.similar) # * cfg.optimize.population_size)
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
            ind.fitness.values = eval_out,

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

    print('Hall of fame:')
    for (i,ind) in enumerate(hof):
        bh_pk_ids = [(i[0], list(_bhs_opt_config[i[0]][i[1]][i[2]].packers)) for i in ind]
        print(toolbox.evaluate(ind), bh_pk_ids)
        # export_vtk_optim_set(ind, "boreholes_opt_cfg." + str(i) + ".vtk", plot=False)
        export_vtk_bh_chamber_set(bh_pk_ids, "boreholes_opt_cfg." + str(i) + ".vtk", plot=False)

    # list of borehole configs sorted by sum of evaluations
    print('Most popular configs:')
    ranks = dict()
    for ind in hof:
        e = toolbox.evaluate(ind)
        for t in ind:
            if not t in ranks:
                ranks[t] = 0
            ranks[t] = ranks[t] + e
    print(sorted(ranks.items(), key=lambda item: -item[1])[:6])

    del creator.FitnessMax
    del creator.Individual

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

    cfg = common.config.load_config(cfg_file)

    optimize(cfg, map, eval_individual_max_l1)
    # optimize(cfg, map, eval_individual_l1)

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

