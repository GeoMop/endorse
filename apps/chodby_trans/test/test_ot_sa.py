# test_sensitivity.py
import numpy as np
import openturns as ot
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


def test_sa_from_cfg_and_sampling():
    sa_cfg = {
        "n_samples": 256,
        "sampler": "sobol",          # or "mc" / "lhs" depending on your implementation
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
    Xg, Xp = sa_obj.sample(seed=123, n_samples=sa_obj.n_samples)
    Xg = np.asarray(Xg)
    Xp = np.asarray(Xp)

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
        "sampler": "sobol",
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
    Xg, Xp = sa_obj.sample(seed=2024, n_samples=sa_obj.n_samples)
    Xg = np.asarray(Xg)
    Xp = np.asarray(Xp)

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
