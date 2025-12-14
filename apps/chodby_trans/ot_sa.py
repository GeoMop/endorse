from __future__ import annotations

from typing import *
from functools import cached_property
import xarray as xr
import sys
import numpy as np
import struct
import hashlib
import attrs
import openturns as ot

import pandas as pd
from chodby_trans import job

# ===== float lattice constants derived from dtype
_FLOAT = np.float64
_FIN = np.finfo(_FLOAT)

# significand bits = mantissa bits (excluding hidden 1) + 1
_SIGNIFICAND_BITS = int(round(-np.log2(_FIN.eps))) + 1          # 53 for float64
_TWO_TO_SIGNIF = _FLOAT(2.0 ** _SIGNIFICAND_BITS)               # 2^signif
_MASK_SIGNIF = np.uint64((1 << _SIGNIFICAND_BITS) - 1)          # mask for that many bits
_ONE_MINUS_ULP = np.nextafter(_FLOAT(1.0), _FLOAT(0.0))         # largest < 1.0


def LogNormal(*args, **kwargs):
    """
    Override ot.LogNormal to allow initialization from different parameters.

    Parameters of associated normal distribution of exponent for base 'e'.
    mean_log: 
    std_log: 

    Parameters of associated normal distribution of exponent for base '10'.
    mean_log10: 
    std_log10:

    Engineer's parameters geometric mean and 95% confidence interval factor 
    scale factor:
    g_mean:
    ci95_factor: The two side 95% CI is: [g_mean / ci95_factor , gmean * ci95_factor]

    shift: Shift of the resulting random values.
    
    $$ log(X - shift) ~ Normal(mean_log, std_log^2) $$
    """
    if args:
        kwargs['mean_log'] = args[0]
        kwargs['std_log'] = args[1]
        if len(args) == 3:
            kwargs['shift'] = args[2]

    if 'mean_log' in kwargs:
        mean = float(kwargs['mean_log'])
        std = float(kwargs['std_log'])
    elif 'mean_log10' in kwargs:
        # parameters of the associated normal distr for base 10.
        mean = float(kwargs.get('mean_log10')) * np.log(10)
        std = float(kwargs['std_log10']) * np.log(10)
    elif 'g_mean' in kwargs:
            # parameters of the associated normal distr for base 10.
            mean = np.log(float(kwargs.get('g_mean')))
            q975 = 1.95996  # scipy.stats.norm.ppf(0.975)
            std = (np.log(float(kwargs['ci95_factor'])) - mean) / q975
    shift = float(kwargs.get('shift', '0.0'))
    return ot.LogNormal(mean, std, shift)


def Population(*args, **kwargs):
    """
    :param args:
    :param kwargs:
        file: CSV file with sample points of the population
        column: column header to select the points
    :return:
    """
    if args:
        kwargs['file'] = args[0]
        kwargs['column'] = args[1]

    df = pd.read_csv(job.input.dir_path / kwargs["file"])
    points_vec = df[kwargs["column"]].to_numpy().reshape(-1,1)
    points = ot.Sample(points_vec)
    return ot.UserDefined(points)


class Seed(ot.PythonDistribution):
    def __init__(self):
        self._distr = ot.Uniform(0.0, 1.0)

    def __getattr__(self, item):
        return getattr(self._distr, item)
    
    @staticmethod
    def get_seedsequence(x: float) -> np.random.SeedSequence:
        """
        Prefered way to initialize numpy generator:

        ss = Seed.seedsequence(float_uniform_seed)
        rng = np.random.default_rng(ss)
        
        Use ss.spawn to create (most probably) independent rng chains:
        
        chain_seeds = ss.spawn(4)   # spwn 4 chains
        """
        s64 = Seed.get_int64(x)
        lo = np.uint32(s64 & np.uint64(0xFFFFFFFF))
        hi = np.uint32(s64 >> np.uint64(32))
        return np.random.SeedSequence([int(lo), int(hi)])

    def get_int64(x: float) -> int:
        salt: bytes = b""
        if x == 0.0:
            x = 0.0 # always +0.0        
        data = struct.pack(">d", x)
        digest_size: int = 8    # in64 == 8 bytes
        h = hashlib.blake2b(data + salt, digest_size=digest_size).digest()
        return np.frombuffer(h, dtype=">u8")[0].astype(np.uint64)

@attrs.define
class Parameter:
    name: str
    group: str
    distribution: ot.Distribution
    param_hash: bytes

    @staticmethod
    def distr_factory(distr_name, args):
        try:
            # resolve in custom distributions
            this_module = sys.modules[__name__]
            distr_class = getattr(this_module, distr_name)
            if isinstance(args, dict):
                return distr_class(**args)
            else:
                return distr_class(*args)
        except AttributeError:       
            distr_class = getattr(ot, distr_name)
            return distr_class(*args)
        
    @staticmethod
    def from_cfg(name, cfg, mixing_name=None):

        if mixing_name is None:
            mixing_name = name

        param_hash = Parameter._param_hash(mixing_name)

        distr = Parameter.distr_factory(
            cfg.get('distr', 'Uniform'),
            cfg.get('args', []))
        group = cfg.get('group', name)
        return Parameter(name, group, distr, param_hash)

    def __attrs_post_init__(self):
        if self.group is None:
            self.group = self.name

    @staticmethod
    def _param_hash(name):
        """
        Hash param name.
        masked to significand bits.
        """
        return hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
 

    def map_from_group(self, u_col: np.ndarray) -> np.ndarray:
        """
        Vectorized mapping: scramble whole group column -> vector quantile.
        """
        group_uniform = np.asarray(u_col, dtype=_FLOAT)
        assert group_uniform.ndim == 1
        group_uniform_cut = np.minimum(group_uniform, _ONE_MINUS_ULP)  # keep strictly < 1.0
        group_int56 = np.floor(group_uniform_cut * _TWO_TO_SIGNIF).astype(np.uint64) # integers on the lattice
        param_hash_56 = np.uint64(int.from_bytes(self.param_hash, "big") & int(_MASK_SIGNIF))


        group_param_int56 = np.bitwise_xor(group_int56, param_hash_56)
        param_uniform = (group_param_int56.astype(_FLOAT) + _FLOAT(0.5)) / _TWO_TO_SIGNIF
    
        # Vector quantiles: pass a sequence of probs; computeQuantile returns ot.Sample (N,1)
        uniform_list = param_uniform.tolist()
        qs = self.distribution.computeQuantile(uniform_list)
        return np.asarray(qs)[:, 0]

@attrs.define(frozen=False)
class InputDesign:
    groups: List[str]
    param_names: List[str]
    group_mat: np.ndarray
    param_mat: np.ndarray
    confidence_level: float = 0.95    


    @cached_property
    def saltelli_layout(self):
        return infer_saltelli_layout(self.param_mat, self.n_groups)

    @property
    def name_to_col(self):
        return {k:i for i, k in enumerate(self.param_names)}

    @property
    def n_groups(self):
        return self.group_mat.shape[1]

    @property
    def n_evals(self):
        return self.group_mat.shape[0]

    @property
    def n_samples(self):
        return len(np.unique(self.i_sample))

    @property
    def n_saltelli(self):
        return len(np.unique(self.i_saltelli))

    @property
    def i_sample(self):
        i_sample, i_saltelli, A_mask = self.saltelli_layout
        return i_sample

    @property
    def i_saltelli(self):
        i_sample, i_saltelli, A_mask = self.saltelli_layout
        return i_saltelli

    @property
    def A_mask(self):
        i_sample, i_saltelli, A_mask = self.saltelli_layout
        return A_mask

    def second_order_indices(self, algo, i):
        try:
            return algo.getSecondOrderIndices(i)
        except TypeError:
            return np.eye(self.n_groups)

    def compute_sobol_xr(
            self,
            da: xr.DataArray,

            n_boot: int = 0,
            boot_seed: int | None = None,
    ) -> xr.Dataset:
        """
        High-level Sobol computation from an xarray DataArray `da`
        with dims including 'IID' and 'QMC' (and possibly others).

        Steps:
          1. Optionally drop sim_time=0.
          2. Stack ('IID','QMC') → 'sample', all other dims → 'output'.
          3. Call low-level InputDesign.compute_sobol on 2D array.
          4. Optionally do bootstrap on the same design and return
             extra variables S1_boot_err, ST_boot_err, S2_boot_err.
        """
        # 1) Pre-process time if present
        if "sim_time" in da.dims:
            da = da.isel(sim_time=slice(1, None))  # skip t = 0

        # 2) Stack to (sample, output)
        conc_2d, _ = self._flatten_da_to_conc_2d(da)

        # 3) Low-level Sobol on 2D (sample, output)
        sobol_ds = self.compute_sobol(conc_2d.compute())
        # unstack 'output' MultiIndex back to original dims (+ 'aux')
        sobol_ds = sobol_ds.unstack("output")

        # 4) Optional bootstrap (done on original da)
        if n_boot > 0:
            boot_err_ds = self._bootstrap_sobol_errors_from_da(
                da=da,
                n_boot=n_boot,
                seed=boot_seed,
            )
            # merge error dataset into main one
            sobol_ds = xr.merge([sobol_ds, boot_err_ds])

        return sobol_ds

    def compute_sobol(
        self,
        output_array: np.ndarray,        # (S, n_outputs) model evals on mapped parameter rows   
    ) -> xr.Dataset:
        input_group_matrix = self.group_mat
        
        if isinstance(output_array, xr.DataArray):
            output_matrix = output_array.data  # dask-backed OK
            output_index = output_array.indexes["output"]
        else:
            output_matrix = np.asarray(output_array)
            output_index = np.arange(output_matrix.shape[1], dtype=int)

        # Ensure 2D outputs
        Y = np.atleast_2d(output_matrix)
        X = np.asarray(input_group_matrix)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Num rows mismatch: len(input_group_matrix) = {len(X)} and " \
            "len((output_matrix) = {len(Y)}")

        # OT expects Samples and the *base size* N used to build the Sobol design.
        # (The constructor is SaltelliSensitivityAlgorithm(inputDesign, outputDesign, N).)
        # See OT docs example. :contentReference[oaicite:0]{index=0}
        Xs = ot.Sample(X)
        Ys = ot.Sample(Y)
        base_size = int(self.n_samples)
        n_outputs = Y.shape[1]
        n_groups = X.shape[1]
        try:
            algo = ot.SaltelliSensitivityAlgorithm(Xs, Ys, base_size)
            algo.setConfidenceLevel(self.confidence_level)  # your class already exposes 95% CI as outputs            

            S1, ST, S2 = [], [], []
            for i in range(n_outputs):
                S1.append(algo.getFirstOrderIndices(i))
                ST.append(algo.getTotalOrderIndices(i))
                S2.append(self.second_order_indices(algo, i))
            
            S1 = np.stack(S1, axis=1)
            ST = np.stack(ST, axis=1)
            S2 = np.stack(S2, axis=2)
            #agg_S1 = algo.getAggregatedFirstOrderIndices()
            #agg_ST = algo.getAggregatedTotalOrderIndices()
            stack_ci = lambda CI: np.stack([CI.getLowerBound(), CI.getUpperBound()], axis=1)
            agg_S1_ci = stack_ci(algo.getFirstOrderIndicesInterval())
            agg_ST_ci = stack_ci(algo.getTotalOrderIndicesInterval())
        except TypeError as e:
            print("Warning: ", e)
            ST = S1 = np.zeros((n_groups, n_outputs))        
            S2 = np.zeros((n_groups, n_groups, n_outputs))
            agg_ST_ci = agg_S1_ci = np.zeros((n_groups, 2))
            

        coords = dict(
            group=np.array(self.groups, dtype=object),
            group2=np.array(self.groups, dtype=object),
            output=output_index,
            bound=np.array(["low", "high"], dtype=object),
        )
        data_vars = dict(
            S1=(("group", "output"), S1),
            ST=(("group", "output"), ST),
            S2=(("group", "group2", "output"), S2),
            #S1_agg=(("group",), agg_S1),
            #ST_agg=(("group",), agg_ST),
            S1_agg_ci=(("group", "bound"), agg_S1_ci),
            ST_agg_ci=(("group", "bound"), agg_ST_ci),
        )
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return ds

    def _flatten_da_to_conc_2d(
        self,
        da: xr.DataArray,
    ) -> tuple[xr.DataArray, list[str]]:
        """Stack (IID, QMC) → sample and everything else → output."""
        output_dims = set(da.dims).union({"aux"})
        output_dims -= {"IID", "QMC"}
        output_dims = sorted(output_dims)
        conc_2d = (
            da.expand_dims({"aux": [0]})
            .stack(sample=("IID", "QMC"), output=output_dims)
            .transpose("sample", "output")
        )
        return conc_2d, output_dims

    def _bootstrap_sobol_errors_from_da(
        self,
        da: xr.DataArray,
        n_boot: int,
        seed: int | None = None,
    ) -> xr.Dataset:
        """
        Internal helper:
        - da: original xarray with dims including 'IID' and 'QMC'
        - n_boot: number of bootstrap replicates
        - seed: RNG seed

        Returns a Dataset with bootstrap CIs:
          S1_boot_err(group, *output_dims, bound)
          ST_boot_err(group, *output_dims, bound)
          S2_boot_err(group, group2, *output_dims, bound)
        where *output_dims match those of compute_sobol_xr (e.g. aux, out, sim_time, ...).
        """
        assert isinstance(da, xr.DataArray)
        assert "IID" in da.dims and "QMC" in da.dims
        assert n_boot > 0

        rng = np.random.default_rng(seed)
        boot_coord = np.arange(n_boot, dtype=int)

        # --- bootstrap along IID dimension ---
        iid_size = da.sizes["IID"]
        idx_boot = rng.integers(0, iid_size, size=(n_boot, iid_size))
        da_boot = da.isel(IID=("boot", idx_boot)).assign_coords(boot=boot_coord)

        # --- flatten (sample, output) including boot in output dims ---
        conc_boot_2d, output_dims = self._flatten_da_to_conc_2d(da_boot)
        sobol_boot = self.compute_sobol(conc_boot_2d)
        sobol_boot = sobol_boot.unstack("output")

        # --- compute percentile CIs over 'boot' dimension ---
        alpha = 1.0 - self.confidence_level
        q_low = alpha / 2.0
        q_high = 1.0 - alpha / 2.0
        quantiles = [q_low, q_high]
        bound = np.array(["low", "high"], dtype=object)

        base_output_dims = [dim for dim in output_dims if dim != "boot"]

        S1 = sobol_boot["S1"]  # dims: ('group', 'boot', *output_dims)
        ST = sobol_boot["ST"]  # same
        S2 = sobol_boot["S2"]  # dims: ('group', 'group2', 'boot', *output_dims)

        # quantiles along 'boot' → dims ('group', 'quantile', *output_dims)
        S1_q = S1.quantile(quantiles, dim="boot")
        ST_q = ST.quantile(quantiles, dim="boot")
        S2_q = S2.quantile(quantiles, dim="boot")

        # rename quantile → bound
        S1_q = S1_q.rename(quantile="bound").assign_coords(bound=bound)
        ST_q = ST_q.rename(quantile="bound").assign_coords(bound=bound)
        S2_q = S2_q.rename(quantile="bound").assign_coords(bound=bound)

        # Reorder dims to match base S1/ST/S2 + 'bound'
        S1_err = S1_q.transpose("group", *base_output_dims, "bound")
        ST_err = ST_q.transpose("group", *base_output_dims, "bound")
        S2_err = S2_q.transpose("group", "group2", *base_output_dims, "bound")

        ds_err = xr.Dataset(
            data_vars=dict(
                S1_boot_err=S1_err,
                ST_boot_err=ST_err,
                S2_boot_err=S2_err,
            ),
        )
        return ds_err

# @attrs.define
# class SobolResultGroup:
#     """
#     Dictionaries maping group name to array of sobol indices
#     """
#     S1: np.ndarray                         # (n_outputs,)
#     ST: np.ndarray                         # (n_outputs,)
#     S2: Dict[str, np.ndarray]
#     agg_S1: float
#     agg_ST: float
#     agg_S1_ci: Tuple[float, float]
#     agg_ST_ci: Tuple[float, float]
#
#
# SobolResult = Dict[str, SobolResultGroup]




@attrs.define(frozen=False)
class SensitivityAnalysis:
   
    parameters: Dict[str, Parameter]
    sampler: Literal["sobol", "mc", "lhs"] = "sobol"
    compute_s2: bool = False
    n_samples: int = 0
    confidence_level: float = 0.95
    _experiment_design: Optional[Callable] = attrs.field(
        init=False,
        repr=False,
        )


    @staticmethod
    def from_cfg(sa_cfg):
        param_cfg = sa_cfg['parameters']
        # parameters = {name: Parameter.from_cfg(name, p_cfg) for name, p_cfg in param_cfg.items()}
        parameters = dict()
        for name, p_cfg in param_cfg.items():
            if p_cfg['distr'] == 'Population':
                # if not set, give a unique common group
                sub_group = p_cfg['group'] if 'group' in p_cfg else name
                for subname, sub_args in p_cfg['parameters'].items():
                    sub_p_cfg = p_cfg.copy()
                    sub_p_cfg['group'] = sub_group
                    sub_p_cfg['args'].update(sub_args)
                    _subname = name + '_' + subname
                    parameters[_subname] = Parameter.from_cfg(_subname, sub_p_cfg, name)
            else:
                parameters[name] = Parameter.from_cfg(name, p_cfg)

        return SensitivityAnalysis(
            parameters,
            sa_cfg.get('sampler', "sobol"),
            sa_cfg.get('second_order', False),
            sa_cfg['n_samples'],
            sa_cfg.get('err_est_confidence_level', 0.95))

    @property
    def groups(self) -> List[str]:
        """_summary_
        List of unique group names, order unrelated to the order of parameters.
        Returns:
        """
        return list({p.group for p in self.parameters.values()})

    
    def __attrs_post_init__(self):
        if not isinstance(self.n_samples, int) or self.n_samples < 0:
            raise ValueError("n_samples must be a non-negative integer")
        if not self.parameters:
            raise ValueError("parameters must be a non-empty dict")
        for k, p in self.parameters.items():
            if p.name != k:
                raise ValueError(f"Parameter key '{k}' must match Parameter.name '{p.name}'")
            if p.group is None:
                p.group = p.name

        # set sampler
        exp_design_functions= {
            'sobol': self._qmc_experiment, 
            'mc': self._mc_experiment, 
            'lhs': self._lhs_experiment} 
        try:
            self._experiment_design = exp_design_functions[self.sampler]
        except KeyError:
            raise KeyError(f"Unknown sampler '{self.sampler}'. Valid: {list(exp_design_functions.keys())}")
        
        # groups via comprehension (order preserved with dict.fromkeys)
        #self._groups: List[str] = list(dict.fromkeys([p.group for p in self.parameters.values()]))
        #self._group_to_col: Dict[str, int] = {g: j for j, g in enumerate(self._groups)}
        #self._param_names: List[str] = list(self.parameters.keys())

    def _qmc_experiment(self, distr) -> ot.WeightedExperiment:
        n_groups = distr.getDimension()
        seq = ot.SobolSequence(n_groups)
        restart_with_distr=False
        return ot.LowDiscrepancyExperiment(seq, distr, self.n_samples, restart_with_distr)

    def _mc_experiment(self, distr) -> ot.WeightedExperiment:
        return ot.MonteCarloExperiment(distr, self.n_samples)

    def _lhs_experiment(self, distr) -> ot.WeightedExperiment:
        always_shuffle = False
        random_shift = False
        return ot.LHSExperiment(distr, self.n_samples, always_shuffle, random_shift)
    
    def saltelli_design(self, distr):
        exp = self._experiment_design(distr)
        #assert isinstance(exp, ot.WeightedExperiment), type(exp)
        sobol_exp = ot.SobolIndicesExperiment(exp, self.compute_s2)
        inputs_design = sobol_exp.generate()
        return inputs_design


    # ----- sampling -----
    def sample(self, seed: int, n_samples: int=None) -> InputDesign:        
        if n_samples is None:
            n_samples = self.n_samples
        assert n_samples > 0
        ot.RandomGenerator.SetSeed(seed)
        group_distrs = ot.JointDistribution([ot.Uniform(0.0, 1.0)] * len(self.groups))
        group_samples = self.saltelli_design(group_distrs)
        raw_input_design_mat = np.array(group_samples)
        # map to parameter samples
        j_group = {g:j for j, g in enumerate(self.groups)}
        cols = [
            p.map_from_group(raw_input_design_mat[:, j_group[p.group]])
            for p in self.parameters.values()
        ]
        param_samples = np.column_stack(cols)

        param_names = list(self.parameters.keys())
        return InputDesign(self.groups, param_names, raw_input_design_mat, param_samples, self.confidence_level)
    
    @property
    def name_to_col(self):
        return {k:i for i, k in enumerate(self.parameters.keys())}

    def param_vec_to_dict(self, params: Union[np.ndarray, List[float]]):
        param_array = np.array(params)
        assert param_array.ndim == 1
        assert len(param_array) == len(self.parameters)
        return dict(zip(self.parameters.keys(), param_array))




def infer_saltelli_layout(input_design: np.ndarray, n_groups: int):
    """
    Infer Saltelli block structure for a design whose columns are parameters (M_p),
    but blocks were generated for n_groups = M_g.

    Supports two layouts:
      1) Contiguous blocks: rows [b*L : (b+1)*L) form a block.
      2) Transposed (SALib-style): rows grouped by within-block index first.

    Returns:
      block_index    : (N,) int block id per row
      saltelli_index : (N,) int position in block per row
      block_mask     : (L, M_p) bool; for each within-block row i, mask[i, j] is True
                       iff column j takes its value from A (False ⇒ from B).
    """
    X = np.asarray(input_design)
    if X.ndim != 2:
        raise ValueError("input_design must be a 2D array")
    N, Mp = X.shape
    if N == 0 or Mp == 0:
        raise ValueError("input_design must have positive shape in both dims")
    if not isinstance(n_groups, int) or n_groups <= 0:
        raise ValueError("n_groups must be a positive integer")

    # Candidate block lengths (first-order vs second-order)
    L1 = 2 + n_groups
    L2 = 2 + 2 * n_groups
    candidates = [L for L in (L1, L2) if N % L == 0]
    if not candidates:
        raise ValueError(
            f"N={N} not divisible by 2+M_g={L1} or 2+2*M_g={L2}; cannot infer Saltelli block size."
        )

    def looks_like_contiguous(Ltest: int) -> bool:
        # In a Saltelli block, each parameter column only takes A or B (≤2 unique values)
        if Ltest > N:
            return False
        block0 = X[:Ltest, :]
        uniq_per_col = [np.unique(block0[:, j]).size for j in range(block0.shape[1])]
        return (max(uniq_per_col) <= 2)

    # Choose block length and layout
    if len(candidates) == 1:
        L = candidates[0]
        contiguous = looks_like_contiguous(L)
    else:
        c0 = looks_like_contiguous(candidates[0])
        c1 = looks_like_contiguous(candidates[1])
        if c0 and not c1:
            L, contiguous = candidates[0], True
        elif c1 and not c0:
            L, contiguous = candidates[1], True
        else:
            # ambiguous; prefer first-order length, still infer layout
            L = min(candidates)
            contiguous = looks_like_contiguous(L)

    B = N // L  # number of blocks

    if contiguous:
        block_index    = np.arange(N, dtype=int) // L
        saltelli_index = np.arange(N, dtype=int) %  L
        rows_block0 = np.arange(L)
    else:
        block_index    = np.arange(N, dtype=int) %  B
        saltelli_index = np.arange(N, dtype=int) // B
        rows_block0 = np.arange(L) * B  # pick the first block’s representative rows

    block0 = X[rows_block0, :]  # (L, Mp)

    # Identify the A-row within the block: row most similar (by equal columns) to all rows
    equal_cols = (block0[:, None, :] == block0[None, :, :])   # (L, L, Mp)
    equality_counts = equal_cols.sum(axis=2).sum(axis=1)      # (L,)
    a_idx = int(np.argmax(equality_counts))

    # For each within-block row, a column equals A-row ⇒ that column comes from A
    block_mask = (block0 == block0[a_idx, :])                 # (L, Mp) boolean

    return block_index, saltelli_index, block_mask
