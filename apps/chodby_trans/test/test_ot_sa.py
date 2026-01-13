# test_sensitivity.py
import numpy as np
import openturns as ot
import xarray as xr
import pytest
from chodby_trans import ot_sa as sa  # e.g., `import sensitivity as sa`


def test_parameter_from_cfg_and_mapping():
    # from_cfg should default group to the parameter name when not provided
    cfg = {"distr": "LogNormal", "args": {'mean_log': 0.0, 'std_log':0.25}}  # mu, sigma
    p = sa.Parameter.from_cfg("k1", cfg)

    assert isinstance(p, sa.Parameter)
    assert p.name == "k1"
    assert p.group == "k1"  # default group to name
    assert isinstance(p.distribution, ot.LogNormal)

    # Mapping should be deterministic for same input, and (very likely) differ for different names (hash key)
    u = np.linspace(0.0, 1.0, 256, endpoint=False)  # [0,1)
    x1 = p.map_from_group(u)
    x2 = p.map_from_group(u)
    assert np.allclose(x1, x2), "Mapping must be deterministic for the same parameter."

    # Same distribution but different parameter name -> different hash key -> different scrambling (very likely)
    p2 = sa.Parameter.from_cfg("k2", cfg)  # different name
    y = p2.map_from_group(u)
    # It's possible (but extremely unlikely) to be identical; allow a probabilistic check
    assert not np.allclose(x1, y), "Different parameter names should produce different scrambles with high probability."

    # LogNormal values must be positive and finite
    assert np.isfinite(x1).all()
    assert (x1 > 0).all()


import numpy as np
import pandas as pd
import pytest

# assumes your package imports like in the original test
# import sa


def _base_sa_cfg(n_samples: int, sampler: str, second_order: bool = True) -> dict:
    return {
        "n_samples": n_samples,
        "sampler": sampler,          # {"QMC","MC","LHC"}
        "second_order": second_order,
        "confidence_level": 0.95,
        "parameters": {
            "k1": {"distr": "LogNormal", "args": [0.0, 0.25], "group": "g1"},
            "k2": {"distr": "Uniform",   "args": [0.1, 0.5],  "group": "g1"},
            "S":  {"distr": "Normal",    "args": [0.0, 1.0]},   # own group "S"
        },
    }


def _run_sa(sa_cfg: dict, model_fn, seed: int = 2024):
    sa_obj = sa.SensitivityAnalysis.from_cfg(sa_cfg)

    in_design = sa_obj.sample(seed=seed, n_samples=sa_obj.n_samples)
    Xg = in_design.group_mat
    Xp = in_design.param_mat

    param_names = list(sa_obj.parameters.keys())
    df = pd.DataFrame(Xp, columns=param_names)

    Y = model_fn(df)
    assert isinstance(Y, np.ndarray)
    assert Y.ndim == 2 and Y.shape[0] == Xp.shape[0]

    result = in_design.compute_sobol(Y)
    return result


def _get_s2_pair(result: dict, g_left="g1", g_right="S"):
    """
    Return S2(g1,S) as a vector of shape (n_outputs,).
    Adapt if your project stores S2 differently.
    """
    # Try group-level storage on g_left
    obj = result[g_left]
    if hasattr(obj, "S2"):
        s2 = getattr(obj, "S2")
        if isinstance(s2, dict) and g_right in s2:
            v = np.asarray(s2[g_right], dtype=float)
            assert v.ndim == 1
            return v
        if isinstance(s2, (list, tuple, np.ndarray)):
            v = np.asarray(s2, dtype=float)
            if v.ndim == 1:
                return v

    # Try group-level storage on g_right
    obj = result[g_right]
    if hasattr(obj, "S2"):
        s2 = getattr(obj, "S2")
        if isinstance(s2, dict) and g_left in s2:
            v = np.asarray(s2[g_left], dtype=float)
            assert v.ndim == 1
            return v
        if isinstance(s2, (list, tuple, np.ndarray)):
            v = np.asarray(s2, dtype=float)
            if v.ndim == 1:
                return v

    raise AssertionError(
        "Could not locate S2(g1,S) vector. Please adapt _get_s2_pair() "
        "to your SobolResultGroup structure."
    )


def _extract_metrics(result: xr.Dataset) -> dict:
    """
    Extract all requested estimates at once:
      - group-level S1 vectors for g1 and S
      - group-level ST vectors for g1 and S
      - single interaction S2(g1,S) vector
    """
    g1 = result["g1"]
    Sg = result["S"]

    S1_g1 = np.asarray(g1.S1, dtype=float)
    S1_S = np.asarray(Sg.S1, dtype=float)
    ST_g1 = np.asarray(g1.ST, dtype=float)
    ST_S = np.asarray(Sg.ST, dtype=float)
    S2_g1S = _get_s2_pair(result, "g1", "S")

    # basic shape consistency (same n_outputs)
    n_out = S1_g1.shape[0]
    assert S1_S.shape == (n_out,)
    assert ST_g1.shape == (n_out,)
    assert ST_S.shape == (n_out,)
    assert S2_g1S.shape == (n_out,)

    return {
        "S1_g1": S1_g1,
        "S1_S": S1_S,
        "ST_g1": ST_g1,
        "ST_S": ST_S,
        "S2_g1S": S2_g1S,
    }


def _assert_generic_sobol_invariants(M: dict, eps: float = 0.20):
    """
    Model-independent checks on all outputs at once.
    """
    # ST >= S1 (allow tiny numerical tolerance)
    assert np.all(M["ST_g1"] + 1e-12 >= M["S1_g1"])
    assert np.all(M["ST_S"]  + 1e-12 >= M["S1_S"])

    # finite
    for k, v in M.items():
        assert np.all(np.isfinite(v)), f"{k} has non-finite values: {v}"

    # not wildly out of range (allow some estimator noise)
    for k in ("S1_g1", "S1_S", "ST_g1", "ST_S", "S2_g1S"):
        v = M[k]
        assert np.all(v >= -eps), f"{k} too negative: {v}"
        assert np.all(v <= 1.0 + eps), f"{k} too large: {v}"


def _assert_sampler_agreement(metrics_by_sampler: dict, atol: float):
    """
    Compare samplers for each vector metric at once by bounding the max range
    across samplers (elementwise) and requiring it stays below atol.
    """
    keys = next(iter(metrics_by_sampler.values())).keys()
    for k in keys:
        stack = np.stack([metrics_by_sampler[s][k] for s in metrics_by_sampler.keys()], axis=0)
        # elementwise range across samplers
        rng = np.max(stack, axis=0) - np.min(stack, axis=0)
        assert np.all(rng <= atol), f"Sampler disagreement too large for {k}: range={rng} (atol={atol})"


# ----------------------------- Models ----------------------------------------

def model_add(df: pd.DataFrame) -> np.ndarray:
    """
    Two outputs:
      y1 = k1 + k2     (depends only on group g1)
      y2 = S           (depends only on group S)
    """
    y1 = df["k1"].to_numpy() + df["k2"].to_numpy()
    y2 = df["S"].to_numpy()
    return np.column_stack([y1, y2])


def expectations_add(M: dict):
    """
    Assert both outputs at once.
    Output 0: depends only on g1
    Output 1: depends only on S
    """
    # output 0 (y1)
    assert M["S1_g1"][0] > 0.8
    assert M["ST_g1"][0] > 0.8
    assert M["S1_S"][0]  < 0.1
    assert M["ST_S"][0]  < 0.1
    assert abs(M["S2_g1S"][0]) < 0.1

    # output 1 (y2)
    assert M["S1_S"][1]  > 0.8
    assert M["ST_S"][1]  > 0.8
    assert M["S1_g1"][1] < 0.1
    assert M["ST_g1"][1] < 0.1
    assert abs(M["S2_g1S"][1]) < 0.1


def model_mul_add(df: pd.DataFrame) -> np.ndarray:
    """
    Two outputs:
      y1 = S*k1 + k2   (depends on both groups with interaction)
      y2 = S           (depends only on group S)
    """
    y1 = df["S"].to_numpy() * df["k1"].to_numpy() + df["k2"].to_numpy()
    y2 = df["S"].to_numpy()
    return np.column_stack([y1, y2])


def expectations_mul_add(M: dict):
    """
    Assert both outputs at once.
    Output 0: interaction should be non-trivial
    Output 1: depends only on S
    """
    # output 0 (y1)
    assert M["S2_g1S"][0] > 0.02
    assert M["ST_g1"][0] > 0.1
    assert M["ST_S"][0]  > 0.1

    # output 1 (y2)
    assert M["S1_S"][1]  > 0.8
    assert M["ST_S"][1]  > 0.8
    assert M["S1_g1"][1] < 0.1
    assert M["ST_g1"][1] < 0.1
    assert abs(M["S2_g1S"][1]) < 0.1


# ----------------------------- Single parametrized test ----------------------

@pytest.mark.parametrize(
    "model_fn, expectations_fn",
    [
        (model_add, expectations_add),
        (model_mul_add, expectations_mul_add),
    ],
)
def test_sa_end_to_end_models_compare_samplers(model_fn, expectations_fn):
    """
    Single test:
      - parametrized by model
      - loops over samplers (QMC/MC/LHC)
      - for each sampler:
          a) compute ALL outputs at once
          b) run model-independent invariants + model-specific expectations (for all outputs)
      - then compare estimates across samplers for the same model (all outputs at once)
    """
    samplers = ["QMC", "MC", "LHC"]
    n_samples = 512
    seed = 2024
    atol_compare = 0.25  # tighten/loosen as needed

    metrics_by_sampler = {}

    for sampler in samplers:
        sa_cfg = _base_sa_cfg(n_samples=n_samples, sampler=sampler, second_order=True)
        res = _run_sa(sa_cfg, model_fn, seed=seed)

        assert set(res.keys()) == {'S1', 'S1_agg_ci', 'S2', 'ST', 'ST_agg_ci'}
        assert set(res.group.values) == {'S', 'g1'}

        M = _extract_metrics(res)

        # a) model-independent invariants
        _assert_generic_sobol_invariants(M, eps=0.20)

        # b) model-specific expectations (both outputs at once)
        expectations_fn(M)

        metrics_by_sampler[sampler] = M

    # Compare samplers for the common model (all outputs at once)
    _assert_sampler_agreement(metrics_by_sampler, atol=atol_compare)


def test_sa_from_cfg_and_sampling():
    sa_cfg = {
        "n_samples": 256,
        "sampler": "QMC",          # or "mc" / "lhs" depending on your implementation
        "second_order": False,
        "confidence_level": 0.95,
        "parameters": {
            # two in the same group "g1"
            "k1": {"distr": "LogNormal", 
                   "args": dict(g_mean=1.0, ci95_factor=1.65), 
                   "group": "g1"},
            "k2": {"distr": "Uniform",  "args": [0.1, 0.5],   "group": "g1"},
            # one in its own group (defaults to "S" if group omitted)
            "S":  {"distr": "Normal",   "args": [0.0, 1.0]},
        },
    }

    sa_obj = sa.SensitivityAnalysis.from_cfg(sa_cfg)
    assert isinstance(sa_obj, sa.SensitivityAnalysis)
    assert sa_obj.n_samples == 256
    assert isinstance(sa_obj.parameters["k1"].distribution, ot.LogNormal)
    assert isinstance(sa_obj.parameters["k2"].distribution, ot.Uniform)
    assert isinstance(sa_obj.parameters["S"].distribution, ot.Normal)

    # sampling
    in_design= sa_obj.sample(seed=123, n_samples=sa_obj.n_samples)
    Xg = in_design.group_mat
    Xp = in_design.param_mat

    # unique groups should be {"g1", "S"}
    uniq_groups = sorted(set(sa_obj.groups))
    assert set(uniq_groups) == {"g1", "S"}

    # shapes: rows = Saltelli design size; cols match groups/params
    assert Xg.shape[1] == len(uniq_groups)
    assert Xp.shape[1] == len(sa_obj.parameters)

    # uniforms in [0,1)
    assert (Xg >= 0.0).all() and (Xg < 1.0).all()
    # mapped parameters finite
    assert np.isfinite(Xp).all()


def test_sa_end_to_end_3params_2outputs():
    """
    End-to-end: 3 params, 2 outputs.
    - k1 (LogNormal) and k2 (Uniform) share group 'g1'
    - S  (Normal)    in its own group 'S'
    Model:
        y1 = k1 + k2            (depends only on group 'g1')
        y2 = S                  (depends only on group 'S')
    Expectations (per-output S1 at group level):
        - For y1: S1['g1'] ~ 1, S1['S'] ~ 0
        - For y2: S1['S'] ~ 1, S1['g1'] ~ 0
    Aggregated indices are variance-weighted across outputs, so we only check per-output arrays.
    """
    sa_cfg = {
        "n_samples": 512,
        "sampler": "QMC",
        "second_order": False,
        "confidence_level": 0.95,
        "parameters": {
            "k1": {"distr": "LogNormal", 
                   "args": [0.0, 0.25], 
                   "group": "g1"},
            "k2": {"distr": "Uniform",  "args": [0.1, 0.5],   "group": "g1"},
            "S":  {"distr": "Normal",   "args": [0.0, 1.0]},  # own group "S"
        },
    }
    sa_obj = sa.SensitivityAnalysis.from_cfg(sa_cfg)

    # design + mapping
    in_design= sa_obj.sample(seed=2024, n_samples=sa_obj.n_samples)
    Xg = in_design.group_mat
    Xp = in_design.param_mat

    # build outputs: y1 = k1 + k2, y2 = S
    name_to_col = {name: i for i, name in enumerate(sa_obj.parameters.keys())}
    y1 = Xp[:, name_to_col["k1"]] + Xp[:, name_to_col["k2"]]
    y2 = Xp[:, name_to_col["S"]]
    Y = np.column_stack([y1, y2])

    # compute Sobol’ (group-level) and broadcast to groups in the result
    result = sa_obj.compute_sobol(Xg, Y)

    # result is Dict[group_name, SobolResultGroup]
    assert set(result.keys()) == {"g1", "S"}

    g1 = result["g1"]
    Sg = result["S"]

    # shapes
    assert g1.S1.shape == (2,)
    assert g1.ST.shape == (2,)
    assert Sg.S1.shape == (2,)
    assert Sg.ST.shape == (2,)

    # per-output expectations (use relaxed tolerances due to MC/QMC randomness & hashing)
    # y1 depends only on group g1
    assert g1.S1[0] > 0.8
    assert Sg.S1[0] < 0.1

    # y2 depends only on group S
    assert Sg.S1[1] > 0.8
    assert g1.S1[1] < 0.1

    # aggregated indices are in [0,1]
    assert 0.0 <= g1.agg_S1 <= 1.0
    assert 0.0 <= g1.agg_ST <= 1.0
    assert 0.0 <= Sg.agg_S1 <= 1.0
    assert 0.0 <= Sg.agg_ST <= 1.0

    # CI tuple sanity
    lo1, hi1 = g1.agg_S1_ci
    loT, hiT = g1.agg_ST_ci
    assert lo1 <= hi1 and -0.1 <= lo1 <= 0.2 and 0.1 <= hi1 <= 0.5
    assert loT <= hiT and -0.1 <= loT <= 0.2 and 0.1 <= hiT <= 0.5




def test_compute_sobol_xr_with_bootstrap():
    """
    Test the high-level InputDesign.compute_sobol_xr with bootstrapping.
    We build a simple 2-group, 1-output model and check that:
      - S1_boot_err / ST_boot_err are present,
      - OT point estimates S1/ST lie inside bootstrap CIs (per group).
    """
    sa_cfg = {
        "n_samples": 256,
        "sampler": "LHS",
        "second_order": True,
        "confidence_level": 0.95,
        "parameters": {
            "k1": {"distr": "LogNormal",
                   "args": [0.0, 0.25],
                   "group": "g1"},
            "k2": {"distr": "Uniform",
                   "args": [0.1, 10],
                   "group": "g1"},
            "S":  {"distr": "Normal",
                   "args": [0.0, 1.0]},  # own group "S"
            "T": {"distr": "Normal",
                  "args": [0.0, 1.0]},  # own group "T"
        },
    }
    sa_obj = sa.SensitivityAnalysis.from_cfg(sa_cfg)


    # design + mapping
    in_design = sa_obj.sample(seed=2025, n_samples=sa_obj.n_samples)
    Xp = in_design.param_mat

    # Single-output model: y = k1 + k2 + 0*S  (only group g1 matters)
    name_to_col = {name: i for i, name in enumerate(sa_obj.parameters.keys())}
    y = 0.01*(0.5 + Xp[:, name_to_col["S"]]) * (0.2 + Xp[:, name_to_col["T"]]) + Xp[:, name_to_col["k1"]] + Xp[:, name_to_col["k2"]]
    # little dependence on 'S'

    Y = y.reshape(-1, 1)

    # Build xarray DataArray with dims ('IID','QMC','out') from flat sample index
    N = Y.shape[0]
    i_sample = np.asarray(in_design.i_sample)
    i_saltelli = np.asarray(in_design.i_saltelli)
    assert i_sample.shape == (N,) and i_saltelli.shape == (N,)

    Y_da = xr.DataArray(
        Y,
        dims=("sample", "out"),
        coords={
            "sample": np.arange(N),
            "IID": ("sample", i_sample),
            "QMC": ("sample", i_saltelli),
            "out": np.arange(Y.shape[1]),
        },
    )
    da = Y_da.set_index(sample=("IID", "QMC")).unstack("sample")  # dims ('IID','QMC','out')

    # High-level Sobol with bootstrap
    ds = in_design.compute_sobol_xr(da, n_boot=64, boot_seed=41)

    # Check that bootstrap variables exist
    assert "S1_boot_err" in ds.data_vars
    assert "ST_boot_err" in ds.data_vars
    assert "S2_boot_err" in ds.data_vars

    # For this test: dims 'group','aux','out' (S1/ST) and 'group','aux','out','bound' (bootstrap)
    groups = list(ds.coords["group"].values)
    g1_idx = groups.index("g1")
    S_idx = groups.index("S")

    # Base OT estimates (single output: aux=0, out=0)
    S1_base = ds["S1"].sel(aux=0).isel(out=0).values  # (group,)
    ST_base = ds["ST"].sel(aux=0).isel(out=0).values  # (group,)

    # Bootstrap CIs
    S1_low  = ds["S1_boot_err"].sel(aux=0, bound="low").isel(out=0).values
    S1_high = ds["S1_boot_err"].sel(aux=0, bound="high").isel(out=0).values
    ST_low  = ds["ST_boot_err"].sel(aux=0, bound="low").isel(out=0).values
    ST_high = ds["ST_boot_err"].sel(aux=0, bound="high").isel(out=0).values

    # Check OT point estimates lie inside bootstrap CIs (per group)
    print("S1 bounds: ", S1_low, "\n<=\n", S1_base, "\n<=\n",  S1_high)
    print("S1 OT CI: ", ds['S1_agg_ci'].values)
    assert np.all(S1_low <= S1_base)
    assert np.all(S1_base <= S1_high)
    print("ST bounds: ", ST_low, "\n<=\n", ST_base, "\n<=\n",  ST_high)
    print("ST OT CI: ", ds['ST_agg_ci'].values)
    assert np.all(ST_low <= ST_base)
    assert np.all(ST_base <= ST_high)

    # Sanity: length of intervals is positive and < 1
    assert np.all((S1_high - S1_low) > 0.0)
    assert np.all((S1_high - S1_low) < 1.0)
    assert np.all((ST_high - ST_low) > 0.0)
    assert np.all((ST_high - ST_low) < 1.0)

    # And the model structure: only group g1 should matter for this output
    assert S1_base[g1_idx] > 0.8
    assert S1_base[S_idx] < 0.1

def _first_order_block_params(Ap, Bp, group_of_param, n_groups):
    """
    Build a single first-order Saltelli block at PARAMETER level (Mp columns),
    for a design generated at GROUP level (n_groups).
      row 0: A (all params from A)
      row 1: B (all params from B)
      row 2+: for each group g, swap that group: params in group g from B, others from A
    """
    Ap = np.asarray(Ap, float)  # (Mp,)
    Bp = np.asarray(Bp, float)  # (Mp,)
    rows = [Ap, Bp]
    for g in range(n_groups):
        r = Ap.copy()
        mask_g = (group_of_param == g)
        r[mask_g] = Bp[mask_g]
        rows.append(r)
    return np.vstack(rows)  # shape (2 + n_groups, Mp)

@pytest.mark.skip
def test_infer_saltelli_layout_2blocks_3groups_4params():
    """
    2 blocks (B=2), n_groups=3, Mp=4 parameters with grouping:
      group 0 -> params [0,1]
      group 1 -> param  [2]
      group 2 -> param  [3]
    We verify both contiguous and transposed row layouts.
    """
    Mp = 4
    n_groups = 3
    L = 2 + n_groups  # 5 rows per block for first-order Saltelli

    # parameter -> group mapping
    group_of_param = np.array([0, 0, 1, 2], dtype=int)
    # sanity: group sizes
    size_g0 = int(np.sum(group_of_param == 0))  # 2
    size_g1 = int(np.sum(group_of_param == 1))  # 1
    size_g2 = int(np.sum(group_of_param == 2))  # 1
    assert (size_g0, size_g1, size_g2) == (2, 1, 1)

    # Block 0 A/B parameter-level values
    A0p = np.array([0.10, 0.20, 0.30, 0.40])
    B0p = np.array([0.90, 0.80, 0.70, 0.60])
    block0 = _first_order_block_params(A0p, B0p, group_of_param, n_groups)  # (5,4)

    # Block 1 A/B parameter-level values (different from block 0)
    A1p = np.array([0.11, 0.22, 0.33, 0.44])
    B1p = np.array([0.91, 0.82, 0.73, 0.64])
    block1 = _first_order_block_params(A1p, B1p, group_of_param, n_groups)  # (5,4)

    # Layout 1: contiguous blocks (OpenTURNS-like)
    X_contig = np.vstack([block0, block1])  # (10,4)

    # Layout 2: transposed (SALib-style): rows grouped by within-block index
    # rows: [block0[i], block1[i]] for i in 0..L-1
    X_trans = np.vstack([np.vstack([block0[i], block1[i]]) for i in range(L)])  # (10,4)

    for X in (X_contig, X_trans):
        N = X.shape[0]
        block_idx, saltelli_idx, mask = sa.get_saltelli_layout_ot(X, n_groups=n_groups)

        # shapes
        assert block_idx.shape == (N,)
        assert saltelli_idx.shape == (N,)
        assert mask.shape == (L, Mp)

        # inferred L and number of blocks
        L_inferred = int(saltelli_idx.max()) + 1
        assert L_inferred == L
        B = N // L_inferred
        assert N == B * L_inferred

        # counts per block / per within-block position
        for b in range(B):
            assert np.sum(block_idx == b) == L_inferred
        for i in range(L_inferred):
            assert np.sum(saltelli_idx == i) == B

        # ---- mask semantics in a first-order block ----
        # Expect one row all-True (A), one row all-False (B),
        # and one swap-row per group with True count = Mp - |group|
        true_counts = mask.sum(axis=1)
        a_rows = np.where(true_counts == Mp)[0]
        b_rows = np.where(true_counts == 0)[0]
        rows_swap_g0 = np.where(true_counts == Mp - size_g0)[0]  # -> Mp-2 = 2
        rows_swap_g1 = np.where(true_counts == Mp - size_g1)[0]  # -> Mp-1 = 3
        # both g1 and g2 swaps have Mp-1 True; there should be 2 such rows total
        assert a_rows.size == 1 and b_rows.size == 1
        assert rows_swap_g0.size == 1
        assert (true_counts == Mp - 1).sum() == 2

        a_idx = int(a_rows[0])
        b_idx0 = int(b_rows[0])

        # Which columns are swapped (False) in those rows?
        # For the 2-True row: should be exactly the two params in group 0
        false_cols_g0 = set(np.where(~mask[rows_swap_g0[0]])[0].tolist())
        expected_g0_false = set(np.where(group_of_param == 0)[0].tolist())
        assert false_cols_g0 == expected_g0_false

        # Among the two 3-True rows: each should have exactly one False column,
        # and together they should cover the two singletons (groups 1 and 2).
        rows_3true = np.where(true_counts == Mp - 1)[0]
        assert len(rows_3true) == 2
        false_sets = [set(np.where(~mask[r])[0].tolist()) for r in rows_3true]
        assert all(len(s) == 1 for s in false_sets)
        union_false = set().union(*false_sets)
        expected_singletons = set(np.where(group_of_param != 0)[0].tolist())
        assert union_false == expected_singletons  # should be {2, 3}

        # ---- Consistency: each in-block row is a mix of that block's A/B rows per mask ----
        for b in range(B):
            # locate A and B rows for block b
            rA = np.where((block_idx == b) & (saltelli_idx == a_idx))[0][0]
            rB = np.where((block_idx == b) & (saltelli_idx == b_idx0))[0][0]
            Arow = X[rA]
            Brow = X[rB]
            for i in range(L_inferred):
                r = np.where((block_idx == b) & (saltelli_idx == i))[0][0]
                mixed = np.where(mask[i], Arow, Brow)
                assert np.allclose(X[r], mixed), (
                    f"Inconsistent mix for block {b}, pos {i}: "
                    f"got {X[r]}, expected {mixed}"
                )

Seed = sa.Seed  # alias for brevity in tests
def test_seed_helpers_and_reproducibility():
    # +0.0 and -0.0 must map to the same uint64
    s_pos0 = Seed.get_int64(0.0)
    s_neg0 = Seed.get_int64(-0.0)
    assert isinstance(s_pos0, np.uint64)
    assert s_pos0 == s_neg0

    # Determinism
    s_a = Seed.get_int64(0.123456789)
    assert s_a == Seed.get_int64(0.123456789)

    # Different inputs (very likely) produce different 64-bit values
    s_b = Seed.get_int64(0.987654321)
    assert s_a != s_b

    # SeedSequence reproducibility
    ss1 = Seed.get_seedsequence(0.42)
    ss2 = Seed.get_seedsequence(0.42)
    rng1 = np.random.default_rng(ss1)
    rng2 = np.random.default_rng(ss2)
    x1 = rng1.integers(0, 2**31, size=8)
    x2 = rng2.integers(0, 2**31, size=8)
    assert np.array_equal(x1, x2)

    # Spawned children should yield distinct streams (with overwhelming probability)
    kids = ss1.spawn(2)
    r0 = np.random.default_rng(kids[0]).integers(0, 2**31, size=8)
    r1 = np.random.default_rng(kids[1]).integers(0, 2**31, size=8)
    assert not np.array_equal(r0, r1)
    assert not np.array_equal(r0, x1)


def test_seed_distribution_behaves_like_uniform():
    """
    The Seed class delegates to ot.Uniform(0,1). Check basic behavior:
    - samples are in [0,1)
    - sample mean is close to 0.5
    """
    ot.RandomGenerator.SetSeed(12345)
    dist = Seed()

    N = 10000
    samp = np.asarray(dist.getSample(N)).ravel()
    assert (samp >= 0.0).all() and (samp < 1.0).all()

    # Mean ~ 0.5 for Uniform(0,1); tolerance relaxed to avoid flakiness
    mean = float(samp.mean())
    assert abs(mean - 0.5) < 0.02, f"mean={mean} too far from 0.5"


def test_seed_mapping_uniformity_on_close_pairs():
    """
    Generate many very-close float pairs u and nextafter(u, 1.0) from Seed(),
    map each to uint64 via Seed.get_int64, and check the mapping is
    roughly uniform:
      - histogram across high bits is near-uniform (chi-by-eye bound)
      - a few individual bit positions are ~50% ones
    """
    ot.RandomGenerator.SetSeed(24680)
    dist = Seed()

    # Draw base uniforms
    N_pairs = 20000  # 20k pairs -> 40k uint64s (keeps the test fast & stable)
    u = np.asarray(dist.getSample(N_pairs)).ravel().astype(np.float64)

    # Make very close neighbors on the right
    u_next = np.nextafter(u, np.float64(1.0))

    # Map both to uint64 using the hashing helper
    seeds_a = np.array([Seed.get_int64(x) for x in u], dtype=np.uint64)
    seeds_b = np.array([Seed.get_int64(x) for x in u_next], dtype=np.uint64)
    seeds_all = np.concatenate([seeds_a, seeds_b]).astype(np.uint64)

    # Sanity: neighbors should (overwhelmingly likely) hash to different values
    # (A cryptographic hash should avalanche on tiny input changes.)
    diff_ratio = np.mean(seeds_a != seeds_b)
    assert diff_ratio > 0.999, f"Too many collisions for close pairs: ratio={diff_ratio}"

    # --- Uniformity check via histogram over high bits ---
    # Use the top k bits to form bins; with k=11 -> 2048 bins
    k = 11
    bins = 1 << k
    high = (seeds_all >> np.uint64(64 - k)).astype(np.uint64)
    counts = np.bincount(high.astype(np.int64), minlength=bins)
    total = seeds_all.size
    expected = total / bins
    std = np.sqrt(expected)

    # Max deviation shouldn't exceed ~6σ (very generous, keeps flakiness low)
    max_dev = np.max(np.abs(counts - expected))
    assert max_dev <= 6 * std, f"Histogram deviates too much: max_dev={max_dev:.2f}, 6σ={6*std:.2f}"

    # --- Bit balance sanity: a few specific bit positions near 50% ones ---
    def bit_ratio(arr: np.ndarray, bit: int) -> float:
        return float(((arr >> np.uint64(bit)) & np.uint64(1)).mean())

    # Check LSB, a middle bit, and the MSB
    ratios = {
        "bit0": bit_ratio(seeds_all, 0),
        "bit31": bit_ratio(seeds_all, 31),
        "bit63": bit_ratio(seeds_all, 63),
    }
    for name, r in ratios.items():
        assert 0.45 <= r <= 0.55, f"{name} ones ratio out of range: {r:.3f}"