from typing import *
import attrs
import itertools
from endorse.Bukov2 import boreholes
from endorse.sa import analyze
from endorse.Bukov2 import sobol_fast, bukov_common as bcommon
from endorse import common
import numpy as np
from endorse.sa.analyze import sobol_vec
from functools import cached_property

@attrs.define(slots=False)
class Chambers:
    sa_problem : Dict[str, Any]
    orig_bh_data : np.ndarray
    bounds: Tuple[int, int]
    packer_size : int
    min_chamber_size : int
    noise: float
    sobol_fn : Any = sobol_fast.vec_sobol_total_only                 # = analyze.sobol_vec



    @classmethod
    def from_bh_set(cls, workdir, cfg, bh_field:boreholes.BoreholeField,  i_bh:int, sa_problem:Dict[str, Any], sobol_fn) -> 'Chambers':
        # load data
        bh_data = bh_field.borohole_data(i_bh)
        bounds = bh_field.point_bounds[i_bh]
        return cls(
            sa_problem,
            bh_data[:, 1:,:],   # remove zero time, as it has zero variance
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
        # n_times, n_param, n_indices = array.shape
        # return max over n_times in given sobol index
        # default is ST
        i_variants = np.argmax(array[..., index], axis=axis, keepdims=True)
        i_variants = i_variants[..., None]
        amax_array = np.take_along_axis(array, i_variants, axis=axis).squeeze(axis=axis)
        return amax_array

    @cached_property
    def _detect_outliers(self):
        arr = self.orig_bh_data
        # Create a mask for NaNs and Infs
        nan_inf_mask = np.isnan(arr) | np.isinf(arr)

        # Create a copy of the array for further processing
        arr_copy = arr.copy()

        # Replace Infs with NaNs in the copy
        arr_copy[nan_inf_mask] = np.nan

        # Calculate quartiles and IQR across the last axis, ignoring NaNs
        Q1 = np.nanpercentile(arr_copy, 25, axis=2, keepdims=True)
        Q3 = np.nanpercentile(arr_copy, 75, axis=2, keepdims=True)
        IQR = Q3 - Q1

        # Determine outlier criteria
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Detect outliers on the copy
        outlier_mask = (arr_copy < lower_bound) | (arr_copy > upper_bound)
        outlier_mask = nan_inf_mask

        # Calculate mean of non-outliers, ignoring NaNs
        non_outlier_mean = np.nanmean(arr_copy, axis=2, keepdims=True)

        # Replace outliers in the original array with the mean of non-outliers
        np.putmask(arr_copy, outlier_mask, non_outlier_mean)
        assert arr_copy.shape == arr.shape

        return outlier_mask, arr_copy

    @property
    def outlier_mask(self):
        return self._detect_outliers[0]

    @property
    def bh_data(self):
        """
        (n_points, n_times, n_samples)
        :return:
        """
        return self._detect_outliers[1]

    @cached_property
    def cumul_bh_data(self):
        lim_pressure = bcommon.soft_lim_pressure(self.bh_data)
        return np.cumsum(lim_pressure, axis=0)

    @property
    def n_groups(self):
        n_params = self.n_params
        group_size = 2 * (n_params + 1)
        return self.bh_data.shape[-1] // group_size

    @cached_property
    def noise_vectors(self):
        a_noise = self.noise * np.random.randn(self.n_groups)
        b_noise = self.noise * np.random.randn(self.n_groups)
        return a_noise, b_noise

    def eval_from_chambers_sa(self, chamber_data, sobol_fn):
        """
        chamber_data (n_chambers, n_times, n_samples)

        Compute sensitivities over n_samples and then max over times.
        returns: (n_chambers, n_param, n_indices)
        :param chamber_data:
        :return:
        """

        n_chambers, n_times, n_samples = chamber_data.shape
        n_params = self.n_params
        group_size = 2 * (n_params + 1)
        n_groups = n_samples // group_size

        ch_data = chamber_data.reshape(-1, n_groups, group_size)
        a_noise, b_noise = self.noise_vectors

        group_size = 2 * (n_params + 1) + 2
        ch_data_with_noise = np.empty((ch_data.shape[0], n_groups, group_size))
        # A matrix eval
        ch_data_with_noise[:, :, 0] = ch_data[:,:,0] + a_noise
        # AB matrix eval
        ch_data_with_noise[:, :, 1:n_params + 1] = ch_data[:, :, 1:n_params + 1] + a_noise[None, :, None]
        ch_data_with_noise[:, :, n_params + 1] = ch_data[:, :, 0] + b_noise
        # BA matrix eval
        ch_data_with_noise[:, :, n_params+2:2*n_params + 2] = ch_data[:, :, n_params + 1:2*n_params+1] + b_noise[None, :, None]
        ch_data_with_noise[:, :, 2*n_params + 2] = ch_data[:, :, 2*n_params+1] + a_noise
        # B matrix eval
        ch_data_with_noise[:, :, 2 * n_params + 3] = ch_data[:,:, 2*n_params+1] + b_noise

        problem = dict(self.sa_problem)
        problem['num_vars'] += 1

        sobol_array = sobol_fn(ch_data_with_noise.reshape(ch_data.shape[0], -1), problem)    # (:, n_indices)
        sobol_array = np.nan_to_num(sobol_array, nan=0.0)
        sobol_array[np.isinf(sobol_array)] = 0
        #var = np.var(ch_data, axis=1)
        #noise_scale = var / (var + self.noise**2)
        #sobol_array[:, :, :] *= noise_scale[:, None, None]
        sobol_array = sobol_array.reshape(n_chambers, n_times, self.n_params+1, -1)

        # remove noise indices
        sobol_array = sobol_array[:, :, :-1, :]

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
                mean_value = (self.cumul_bh_data[end, :, :] - self.cumul_bh_data[chamber_begin, :, :]) / chamber_size
                chamber_means.append(mean_value)

        chamber_means = np.stack(chamber_means)
        # (n_chambers, n_times, n_samples)

        # Evaluate chamber sensitivities
        sensitivities =  self.eval_from_chambers_sa(chamber_means, sobol_fn)
        # (n_chambers, n_params, 1)
        return sensitivities

    @cached_property
    def _all_chambers(self):
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
        index = np.full((self.n_points, self.n_points), len(chambers), dtype=np.int32)
        for i_chamber, (begin, end) in enumerate(chambers):
            index[begin, end] = i_chamber
        total_sensitivities = self.eval_chambers(chambers)[:, :, 0] # remove index axis of size one
        n_chambers, n_params = total_sensitivities.shape

        # add zero sensitivities item for invalid bigin, end pairs
        total_sensitivities = np.concatenate((total_sensitivities, np.zeros((1, n_params))), axis=0)

        return index, total_sensitivities

    @property
    def index(self):
        index, data = self._all_chambers
        return index

    @property
    def chambers_sensitivities(self):
        index, data = self._all_chambers
        return data

    @cached_property
    def chambers_norm_sensitivities(self):
        data = self.chambers_sensitivities
        """
        All sensitivities have skewed PDF with steep right edge, high quantile or even maximum would be good scaling 
        function.
        """
        scaling_measure = lambda x, **kw : np.quantile(data, q=0.95, **kw)
        return bcommon.normalize_sensitivities(data, scaling_measure)

    # @property
    # def index(self):
    #     return self.all_chambers[0]

    def packer_config(self, packers, opt_values):
        chambers_begin = packers[:-1]
        chambers_end = packers[1:] - self.packer_size
        i_chambers = self.index[chambers_begin, chambers_end]
        sensitivites = self.chambers_sensitivities[i_chambers]

        chambers = zip(chambers_begin, chambers_end)
        full_sobol = self.eval_chambers(chambers, sobol_fn = sobol_vec)
        return PackerConfig(packers, sensitivites, opt_values, full_sobol)

@attrs.define(slots=False)
class PackerConfig:
    packers: np.ndarray          # (n_packers,) int ; positions of the ends of the packers,
    st_values: np.ndarray        # chambers sensitivities, shape (3, n_params)
    opt_values: np.ndarray       # value of self in the view of every parameter (n_params, )
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


def combination_to_packers(chambers, comb):
    i_chamber = np.arange(len(comb), dtype=np.int32)
    return (i_chamber * (chambers.min_packer_distance - 1)) + comb


def optimal_configs(chambers:Chambers, packers, chamber_sensitivities, n_largest, weights):
    """
    :param packers: (n_combinations, n_packers)  # packer positions
    :param chamber_sensitivities: (n_combinations, n_chambers, n_param)
    :param n_largest: int
    :return: (n_param, n_largest) x PackerConfig
    """
    n_params = chambers.n_params
    # first criteria
    param_max_over_chambers = np.max(chamber_sensitivities, axis=1)

    # second criteria
    sum_over_chambers = np.sum(chamber_sensitivities, axis=1)
    sum_over_other_params = np.sum(sum_over_chambers, axis=1)[:, None] - sum_over_chambers

    values = weights[0] * param_max_over_chambers + weights[1] * sum_over_other_params
    # shape (n_combinations, n_params)

    # Now we select for each parameter n_largest combinations
    indices = np.argpartition(values, -n_largest, axis=0)[-n_largest:]
    single_best_config = lambda idx : chambers.packer_config(packers[idx], values[idx])
    best_configs_for_param = lambda i_param_configs : [single_best_config(i_param_configs[i_best]) for i_best in range(n_largest)]
    opt_packer_configs = [ best_configs_for_param(indices[:, i_param]) for i_param in range(n_params)]
    # shape: (n_params, n_largest) of PackerConfig
    return opt_packer_configs


def optimize_packers(cfg, chambers: Chambers):
    cfg_opt = cfg.optimize
    n_points = chambers.n_points
    n_packers = cfg_opt.n_packers
    n_largest = cfg_opt.n_best_packer_conf
    n_params = chambers.n_params
    weights = cfg_opt.weights

    # Chambers implicitely evaluates all_chambers, computing sensitivities for all possible chamber begin - end
    # pairs.

    # Next we evaluate all combinations of 3 chambers
    n_chambers = n_packers - 1
    total_items = n_points - n_chambers * chambers.min_packer_distance + n_chambers
    combinations = list(itertools.combinations(range(total_items), n_packers))
    n_comb = len(combinations)
    packers = np.array([combination_to_packers(chambers, comb) for comb in combinations], dtype=np.int32)
    np.random.shuffle(packers)    # to avoid prior preferences in position
    chambers_ranges = np.stack( (packers[:, :-1], packers[:, 1:]), axis=2) # (n_packer_configs, n_chambers, 2)
    assert chambers_ranges.shape[1] == n_chambers
    chambers_ranges = chambers_ranges.reshape(-1, 2)
    begins = chambers_ranges[:, 0]
    ends = chambers_ranges[:, 1] - chambers.packer_size
    chamber_indices = chambers.index[begins, ends]
    assert chamber_indices.shape == (chambers_ranges.shape[0],)
    chamber_sensitivites = chambers.chambers_sensitivities[chamber_indices].reshape(n_comb, n_chambers, n_params)
    chamber_norm_sensitivites = chambers.chambers_norm_sensitivities[chamber_indices].reshape(n_comb, n_chambers, n_params)

    opt_fn = lambda x : optimal_configs(chambers, packers, x, n_largest, weights)
    opt_for_unscaled = opt_fn(chamber_sensitivites)
    opt_for_normalized = opt_fn(chamber_norm_sensitivites)
    opt_packer_configs = [ [*unscaled, *normalized] for unscaled, normalized in zip(opt_for_unscaled, opt_for_normalized)]
    print( len(opt_packer_configs), len(opt_packer_configs[0]))
    return opt_packer_configs

