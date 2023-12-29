from typing import *
import attrs
import itertools
from endorse.Bukov2 import boreholes
from endorse.sa import analyze
from endorse.Bukov2 import sobol_fast
import numpy as np
from endorse.sa.analyze import sobol_vec
from functools import cached_property

@attrs.define(slots=False)
class Chambers:
    sa_problem : Dict[str, Any]
    bh_data : np.ndarray
    bounds: Tuple[int, int]
    packer_size : int
    min_chamber_size : int
    sobol_fn : Any = sobol_fast.vec_sobol_total_only                 # = analyze.sobol_vec



    @classmethod
    def from_bh_set(cls, workdir, cfg, bh_set:boreholes.BoreholeSet,  i_bh:int, sa_problem:Dict[str, Any], sobol_fn) -> 'Chambers':
        bh_data, bh_bounds = bh_set.borohole_data(workdir, cfg, i_bh)
        bounds = bh_set.line_bounds[i_bh]
        return cls(
            sa_problem,
            bh_data,
            bounds,
            sobol_fn = sobol_fn,
            **cfg.chambers)

    @property
    def n_points(self):
        return self.bh_data.shape[0]

    @property
    def n_params(self):
        return self.sa_problem['num_vars']


    @property
    def min_packer_distance(self):
        return self.packer_size + self.min_chamber_size

    @staticmethod
    def sobol_max(array, axis=0, index=0):
        # n_variants, n_param, n_indices = array.shape
        # return max over n_variants in given sobol index
        # default is ST
        i_variants = np.argmax(array[..., index], axis=axis, keepdims=True)
        i_variants = i_variants[..., None]
        amax_array = np.take_along_axis(array, i_variants, axis=axis).squeeze(axis=axis)
        return amax_array

    def eval_from_chambers_sa(self, chamber_data, sobol_fn):
        """
        chamber_data (n_chambers, n_times, n_samples)

        Compute sensitivities over n_samples and then max over times.
        returns: (n_chambers, n_param, n_indices)
        :param chamber_data:
        :return:
        """

        n_chambers, n_times, n_samples = chamber_data.shape
        ch_data = chamber_data.reshape(-1, n_samples)
        sobol_array = sobol_fn(ch_data, self.sa_problem)
        sobol_array = np.nan_to_num(sobol_array, nan=0.0)
        sobol_array[np.isinf(sobol_array)] = 0
        sobol_array = sobol_array.reshape(n_chambers, n_times, self.n_params, -1)# n_chambers

        # Compute maximum over times
        max_sobol = self.sobol_max(sobol_array, axis = 1)  # max over times with respect to total indices

        return max_sobol

    def eval_chambers(self, chambers, sobol_fn = None):
        if sobol_fn is None:
            sobol_fn =  self.sobol_fn
        chamber_means = []
        for begin, end in chambers:
                chamber_begin = begin
                chamber_size = (end - chamber_begin)
                mean_value = (self.bh_data[end, :, :] - self.bh_data[chamber_begin, :, :]) / chamber_size
                chamber_means.append(mean_value)

        chamber_means = np.stack(chamber_means)
        # (n_chambers, n_times, n_samples)

        # Evaluate chamber sensitivities
        return self.eval_from_chambers_sa(chamber_means, sobol_fn)

    @cached_property
    def all_chambers(self):
        """
        Precalculates ST indices of all chamber variants.
        (n_chamber_variants, n_params)
        :return:
        """
        min_idx, max_idx = self.bounds
        chambers = [
            (begin, end)
            for begin in range(min_idx, max_idx - self.min_chamber_size)
                for end in range(begin + self.min_chamber_size, max_idx)
            ]
        index = np.full((self.n_points, self.n_points), -1, dtype=np.int32)
        for i_chamber, (begin, end) in enumerate(chambers):
            index[begin, end] = i_chamber
        return index, self.eval_chambers(chambers)[:, :, 0]

    def chamber(self, i, j):
        index, data = self.all_chambers
        i_chamber = index[i, j]
        if i_chamber > 0:
            return data[i_chamber, :]
        else:
            return np.full(self.n_params, np.nan)
    #
    # @property
    # def index(self):
    #     return self.all_chambers[0]

    def packer_config(self, packers):
        chambers = zip(packers[:-1], packers[1:])
        full_sobol = self.eval_chambers(chambers, sobol_fn = sobol_vec)
        return PackerConfig(packers, full_sobol)

@attrs.define(slots=False)
class PackerConfig:
    packers: np.ndarray       # (n_packers,) int ; positions of the ends of the packers,
    sobol_indices: np.ndarray # (n_chambers, n_param, n_sobol_indices) float

    def __getstate__(self):
        state = self.__dict__.copy()
        # Optionally remove the cached_property data if not needed
        #state.pop('expensive_computation', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

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


def packers_eval(chambers_obj, packers, weights):
    chambers = zip(packers[:-1], packers[1:])
    sensitivites = np.array([chambers_obj.chamber(i, j - chambers_obj.packer_size) for i, j in chambers])
    sensitivites = sensitivites[ ~sensitivites.isnan()]
    # shape: (n_chamberes = 3, n_params)
    return weights[0] * np.max(sensitivites, axis=0) + weights[1] * np.mean(sensitivites, axis=0)

def combination_to_packers(chambers, comb):
    i_chamber = np.arange(len(comb), dtype=np.int32)
    return (i_chamber * (chambers.min_packer_distance - 1)) + comb


def optimize_packers(cfg, chambers: Chambers):
    cfg_opt = cfg.optimize
    n_points = chambers.n_points
    n_packers = cfg_opt.n_packers
    n_largest = cfg_opt.n_best_packer_conf
    n_params = chambers.n_params
    weights = cfg_opt.weights

    total_items = n_points - (n_packers - 1) * chambers.min_packer_distance + n_packers - 1
    combinations = list(itertools.combinations(range(total_items), n_packers))
    packers = np.array([combination_to_packers(chambers, comb) for comb in combinations], dtype=np.int32)
    values = [packers_eval(chambers, p, weights) for p in packers]
    # shape (n_combinations, n_params)
    indices = np.argpartition(values, -n_largest, axis=0)[-n_largest:]
    opt_packer_configs = [[chambers.packer_config(packers[indices[i_best, i_param]]) for i_best in range(n_largest)]
        for i_param in range(n_params)]

    return opt_packer_configs

