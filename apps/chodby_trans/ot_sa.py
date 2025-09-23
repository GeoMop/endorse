from __future__ import annotations

from typing import *
import sys
import numpy as np
import hashlib
import attrs
import openturns as ot

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



@attrs.define
class Parameter:
    name: str
    group: str
    distribution: ot.Distribution
    

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
    def from_cfg(name, cfg):
        
        distr = Parameter.distr_factory(
            cfg.get('distr', 'Uniform'),
            cfg.get('args', []))
        group = cfg.get('group', name)
        return Parameter(name, group, distr)

    def __attrs_post_init__(self):
        if self.group is None:
            self.group = self.name

    @property
    def param_hash(self):
        """
        Hash param name.
        masked to significand bits.
        """
        return hashlib.blake2b(self.name.encode("utf-8"), digest_size=8).digest()
 

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


@attrs.define
class SobolResultGroup:
    """
    Dictionaries maping group name to array of sobol indices
    """
    S1: np.ndarray                         # (n_outputs,)
    ST: np.ndarray                         # (n_outputs,)
    S2: Dict[str, np.ndarray]                      
    agg_S1: float
    agg_ST: float
    agg_S1_ci: Tuple[float, float]
    agg_ST_ci: Tuple[float, float]
 
 
SobolResult = Dict[str, SobolResultGroup]

@attrs.define(frozen=False)
class SensitivityAnalysis:
    n_samples: int
    parameters: Dict[str, Parameter]
    sampler: Literal["sobol", "mc"] = "sobol"
    compute_s2: bool = False
    confidence_level: float = 0.95
    _experiment_design: Optional[Callable] = attrs.field(
        init=False,
        repr=False,
        )


    @staticmethod
    def from_cfg(sa_cfg):
        param_cfg = sa_cfg['parameters']
        parameters = {name: Parameter.from_cfg(name, p_cfg) for name, p_cfg in param_cfg.items()}
        return SensitivityAnalysis(
            sa_cfg['n_samples'],
            parameters,
            sa_cfg.get('sampler', "sobol"),
            sa_cfg.get('second_order', False),
            sa_cfg.get('confidence_level', 0.95))

    @property
    def groups(self) -> List[str]:
        """_summary_
        List of unique group names, order unrelated to the order of parameters.
        Returns:
        """
        return list({p.group for p in self.parameters.values()})

    
    def __attrs_post_init__(self):
        if not isinstance(self.n_samples, int) or self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
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
    def sample(self, seed: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:        
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
        return raw_input_design_mat, param_samples
        
    def second_order_indices(self, algo, i):
        try:
            return algo.getSecondOrderIndices(i)
        except TypeError:
            return np.eye(len(self.groups))

    def compute_sobol(
        self,
        input_group_matrix: np.ndarray,   # (S, n_groups) from saltelli_design(...)
        output_matrix: np.ndarray,        # (S, n_outputs) model evals on mapped parameter rows
    ) -> SobolResult:
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

        algo = ot.SaltelliSensitivityAlgorithm(Xs, Ys, base_size)
        algo.setConfidenceLevel(self.confidence_level)  # your class already exposes 95% CI as outputs

        n_outputs = Y.shape[1]
        S1, ST, S2 = [], [], []
        for i in range(n_outputs):
            S1.append(algo.getFirstOrderIndices(i))
            ST.append(algo.getTotalOrderIndices(i))
            S2.append(self.second_order_indices(algo, i))
        
        S1 = np.stack(S1, axis=1)
        ST = np.stack(ST, axis=1)
        S2 = np.stack(S2, axis=2)
        agg_S1 = algo.getAggregatedFirstOrderIndices()
        agg_ST = algo.getAggregatedTotalOrderIndices()
        stack_ci = lambda CI: np.stack([CI.getLowerBound(), CI.getUpperBound()], axis=1)
        agg_S1_ci = stack_ci(algo.getFirstOrderIndicesInterval())
        agg_ST_ci = stack_ci(algo.getTotalOrderIndicesInterval())
        return {
            group: SobolResultGroup(
                S1[ig], ST[ig], dict(zip(self.groups, S2[ig])), 
                agg_S1[ig], agg_ST[ig], 
                agg_S1_ci[ig], agg_ST_ci[ig] 
            )
            for ig, group in enumerate(self.groups)
        }    
