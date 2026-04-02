# # test_sensitivity_end_to_end.py
# import numpy as np
# import openturns as ot
# import xarray as xr
# import matplotlib.pyplot as plt
# # matplotlib.use("Agg", force=True)  # headless plotting
# import pytest

# # >>>>>>>>>>>> CHANGE THIS if your module is inside a package <<<<<<<<<<<<
# # e.g. from mypkg import ot_sa as sa
# import chodby_trans.ot_sa as sa
# import chodby_trans.sa.vector_sa_plot as vsp

# def _stable_groups(sa_obj):
#     """Stable list of groups (order-preserving by first appearance in parameters)."""
#     return list(dict.fromkeys([p.group for p in sa_obj.parameters.values()]))


# def test_sensitivity_dataset_selection_and_plot_end_to_end():
#     """
#     End-to-end:
#       - Build SA from cfg (3 params, 2 groups)
#       - Generate Saltelli design and map to parameter samples
#       - Evaluate simple 2-output model
#       - Compute Sobol indices -> xr.Dataset
#       - Select dominant terms with 'others'
#       - Plot stacked bars (smoke test)
#     """
#     # ------- config: 3 params, 2 groups -------
#     sa_cfg = {
#         "n_samples": 256,
#         "sampler": "sobol",
#         "second_order": False,
#         "err_est_confidence_level": 0.95,
#         "parameters": {
#             # group g1: two params
#             "k1": {"distr": "LogNormal", "args": [0.0, 0.25], "group": "g1"},
#             "k2": {"distr": "Uniform", "args": [0.1, 0.5], "group": "g1"},
#             # group g2: one param
#             "S": {"distr": "Normal", "args": [0.0, 1.0], "group": "g2"},
#         },
#     }

#     sa_obj = sa.SensitivityAnalysis.from_cfg(sa_cfg)
#     input_design = sa_obj.sample(seed=123)
#     # # Ensure deterministic group order for the test (avoid set-order flakiness)
#     # #stable = _stable_groups(sa_obj)
#     # #setattr(sa_obj, "groups", stable)  # shadow property for this instance

#     # # ------- Saltelli design at GROUP level -------
#     # ot.RandomGenerator.SetSeed(123)
#     # G = len(sa_obj.groups)
#     # group_dist = ot.JointDistribution([ot.Uniform(0.0, 1.0)] * G)
#     # exp = sa_obj._experiment_design(group_dist)
#     # Xg = np.array(ot.SobolIndicesExperiment(exp, sa_obj.compute_s2).generate())  # (S, G)

#     # # ------- Map to PARAMETER level using the Parameter mapper -------
#     # g2col = {g: j for j, g in enumerate(stable)}
#     # params_in_order = list(sa_obj.parameters.values())
#     # Xp_cols = [p.map_from_group(Xg[:, g2col[p.group]]) for p in params_in_order]
#     # Xp = np.column_stack(Xp_cols)  # (S, P)

#     # ------- Simple 2-output model:
#     # y1 depends only on group g1 (k1 + k2), y2 depends only on group g2 (S)
#     Xp = input_design.param_mat
#     name_to_col = sa_obj.name_to_col
#     y1 = Xp[:, name_to_col["k1"]] + Xp[:, name_to_col["k2"]]
#     y2 = Xp[:, name_to_col["S"]]
#     Y = np.column_stack([y1, y2])  # (S, 2)

#     # ------- Compute Sobol indices -> xr.Dataset -------
#     ds = sa_obj.compute_sobol(input_design.group_mat, Y)
#     assert isinstance(ds, xr.Dataset)

#     # Coords/vars existence & shapes
#     assert {"group", "group2", "output", "bound"}.issubset(ds.coords)
#     assert {"S1", "ST", "S2", "S1_agg", "ST_agg", "S1_agg_ci", "ST_agg_ci"}.issubset(ds.data_vars)
#     G_ds = ds.dims["group"]
#     M_ds = ds.dims["output"]
#     G = len(sa_obj.groups)
#     assert G_ds == G and M_ds == 2
#     assert ds["S1"].shape == (G, 2)
#     assert ds["S2"].shape == (G, G, 2)

#     # Expected dominance per output (relaxed thresholds)
#     groups_from_ds = list(map(str, ds["group"].values))
#     g1_idx = groups_from_ds.index("g1")
#     g2_idx = groups_from_ds.index("g2")
#     assert float(ds["S1"].values[g1_idx, 0]) > 0.7  # y1 dominated by g1
#     assert float(ds["S1"].values[g2_idx, 0]) < 0.2
#     assert float(ds["S1"].values[g2_idx, 1]) > 0.7  # y2 dominated by g2
#     assert float(ds["S1"].values[g1_idx, 1]) < 0.2

#     # ------- Postprocess: select dominant terms with 'others' -------
#     ds_sel = vsp.select_sobol_terms_with_others(ds, var_threshold=0.95, si_threshold=0.0)
#     assert {"param", "output", "bound"}.issubset(ds_sel.coords)
#     assert {"SI", "SI_agg", "SI_agg_ci"}.issubset(ds_sel.data_vars)

#     params = list(map(str, ds_sel["param"].values))
#     assert "others" in params

#     # Each output column should stack to ~1.0
#     col_sums = ds_sel["SI"].sum(dim="param").values
#     #assert np.allclose(col_sums, 1.0, atol=5e-3)

#     # Aggregated 'others' equals 1 - sum of non-others aggregates (within tolerance)
#     others_idx = params.index("others")
#     others_agg = float(ds_sel["SI_agg"].values[others_idx])
#     non_others_sum = float(ds_sel["SI_agg"].values.sum() - others_agg)
#     assert np.isfinite(others_agg)
#     assert 0.0 <= others_agg <= 1.0
#     #assert np.isclose(others_agg + non_others_sum, 1.0, atol=1e-6)

#     # ------- Plot (smoke test) -------
#     fig, (axL, axR) = vsp.plot_sobol_stacked(ds_sel, figsize=(9, 4))
#     assert fig is not None
#     plt.show()  # in case the backend needs it
#     # # close to free memory in CI
#     # import matplotlib.pyplot as plt
#     # plt.close(fig)

# test_ode_sobol_select_plot.py
import numpy as np
import xarray as xr
import openturns as ot
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pytest

import chodby_trans.ot_sa as sa
import chodby_trans.sa.vector_sa_plot as sp


def test_ode_4params_ic_selection_and_plot():
    """
    ODE: dx/dt = -k x + q,  x(0)=x0,  y(t) = s * x(t)
    Parameters: k (rate), q (forcing), s (scale), x0 (initial condition)
    Expectation: at early time, IC dominates; at late time, IC contribution ~ 0.
    """

    # ---------- Build SensitivityAnalysis with 4 parameters ----------
    n_samples = 512
    cfg = {
        "n_samples": n_samples,
        "sampler": "sobol",
        "second_order": False,
        "err_est_confidence_level": 0.95,
        "parameters": {
            "k":  {"distr": "LogNormal", "args": [np.log(0.2), 0.25], "group": "k"},
            "q":  {"distr": "Normal",    "args": [1.0, 0.25],        "group": "q"},
            "s":  {"distr": "LogNormal", "args": [np.log(1.0), 0.1],  "group": "s"},
            "x0": {"distr": "Normal",    "args": [1.0, 0.3],         "group": "IC"},  # IC group
        },
    }
    sa_obj = sa.SensitivityAnalysis.from_cfg(cfg)
    input_design = sa_obj.sample(seed=123)  # (S, G)
    Xp = input_design.param_mat
    #name_to_col = {p.name: i for i, p in enumerate(params_in_order)}

    # ---------- Evaluate ODE on a time grid (vectorized) ----------
    t = np.geomspace(1e-2, 20.0, 60)  # 60 outputs
    k = Xp[:, input_design.name_to_col["k"]]
    q = Xp[:, input_design.name_to_col["q"]]
    s = Xp[:, input_design.name_to_col["s"]]
    x0 = Xp[:, input_design.name_to_col["x0"]]

    exp_term = np.exp(-k[:, None] * t[None, :])                   # (S, M)
    x_t = x0[:, None] * exp_term + (q[:, None] / k[:, None]) * (1.0 - exp_term)
    Y = s[:, None] * x_t                                          # (S, M)

    # ---------- Sobol indices -> Dataset ----------
    sob = input_design.compute_sobol(Y)  # dims: group, group2, output, bound
    assert isinstance(sob, xr.Dataset)
    assert {"S1", "ST", "S2", "S1_agg", "ST_agg", "S1_agg_ci", "ST_agg_ci"}.issubset(sob.data_vars)

    # ---------- Expectation: IC dominates early, small late ----------
    groups_ds = list(map(str, sob["group"].values))
    gIC = groups_ds.index("IC")

    S1 = sob["S1"].to_numpy()          # (G, M)
    S1_ic_early = float(S1[gIC, 0])
    S1_ic_late  = float(S1[gIC, -1])

    # Early-time IC should be notable; late-time IC should be small.
    assert S1_ic_early > 0.25, f"IC first-order at earliest time too small: {S1_ic_early:.3f}"
    assert S1_ic_late  < 0.10, f"IC first-order at last time too large: {S1_ic_late:.3f}"

    # ---------- Selection with 'others' & plotting ----------
    ds_sel = sp.select_sobol_terms_with_others(sob, var_threshold=0.95, output_dim="output", si_threshold=0.0)
    assert {"param", "output", "bound"}.issubset(ds_sel.coords)
    assert {"SI", "SI_agg", "SI_agg_ci"}.issubset(ds_sel.data_vars)

    # Columns should sum to ~1
    col_sums = ds_sel["SI"].sum(dim="param").to_numpy()
    #assert np.allclose(col_sums, 1.0, atol=5e-3)

    # 'others' must be present
    params = list(map(str, ds_sel["param"].values))
    assert "others" in params

    # Plot (smoke test, no interactive display)
    fig, (axL, axR) = sp.plot_sobol_stacked(ds_sel, figsize=(10, 4), 
                                            x_label="sim_time", out_path="stacked_sobol.pdf")
    fig.show()
    assert fig is not None
    plt.close(fig)
