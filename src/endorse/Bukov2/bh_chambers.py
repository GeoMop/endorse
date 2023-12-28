from typing import *
import attrs
from endorse.Bukov2 import boreholes
from endorse.sa import analyze
from endorse.Bukov2 import sobol_fast
import numpy as np

from functools import cached_property

@attrs.define(slots=False)
class Chambers:
    sa_problem : Dict[str, Any]
    bh_data : np.ndarray
    packer_size : int
    min_chamber_size : int
    sobol_fn : Any = sobol_fast.vec_sobol_total_only                 # = analyze.sobol_vec



    @classmethod
    def from_bh_set(cls, workdir, cfg, bh_set:boreholes.BoreholeSet,  i_bh:int, sa_problem:Dict[str, Any], sobol_fn) -> 'Chambers':
        bh_data, bh_bounds = bh_set.borohole_data(workdir, cfg, i_bh)

        return cls(
            sa_problem,
            bh_data,
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

    def eval_from_chambers_sa(self, chamber_data):
        """
        chamber_data (n_chambers, n_times, n_samples)

        Compute sensitivities over n_samples and then max over times.
        returns: (n_chambers, n_param, n_indices)
        :param chamber_data:
        :return:
        """

        n_chambers, n_times, n_samples = chamber_data.shape
        ch_data = chamber_data.reshape(-1, n_samples)
        sobol_array = self.sobol_fn(ch_data, self.sa_problem)
        sobol_array = np.nan_to_num(sobol_array, nan=0.0)
        sobol_array = sobol_array.reshape(n_chambers, n_times, self.n_params, -1)# n_chambers

        # Compute maximum over times
        max_sobol = self.sobol_max(sobol_array, axis = 1)  # max over times with respect to total indices

        return max_sobol

    def eval_chambers(self, chambers):
        chamber_means = []
        for begin, end in chambers:
                chamber_begin = begin
                chamber_size = (end - chamber_begin)
                mean_value = (self.bh_data[end, :, :] - self.bh_data[chamber_begin, :, :]) / chamber_size
                chamber_means.append(mean_value)

        chamber_means = np.stack(chamber_means)
        # (n_chambers, n_times, n_samples)

        # Evaluate chamber sensitivities
        return self.eval_from_chambers_sa(chamber_means)

    @cached_property
    def all_chambers(self):
        """
        Precalculates ST indices of all chamber variants.
        (n_chamber_variants, n_params)
        :return:
        """
        chambers = [
            (begin, end)
            for begin in range(self.n_points - self.min_chamber_size)
                for end in range(begin + self.min_chamber_size, self.n_points)
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
            return None
    #
    # @property
    # def index(self):
    #     return self.all_chambers[0]
