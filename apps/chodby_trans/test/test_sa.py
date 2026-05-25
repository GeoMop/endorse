# test_sobol_result.py
import numpy as np
import numpy.testing as npt
import pickle
import pytest

from chodby_trans import sa

def test_sobol_result():
    # --- synthetic inputs
    names = ["X1", "X2", "X3"]
    S1 = [0.3, 0.1, 0.6]
    lo = [0.25, 0.05, 0.55]
    hi = [0.35, 0.15, 0.65]
    ST = [0.5, 0.2, 0.8]

    # ---------- make_results_dict
    # packed = SobolResult.make_results_dict(name=names, S1=S1, lo=lo, hi=hi, ST=ST)
    # assert set(packed.keys()) == {"name", "S1", "lo", "hi", "ST"}
    # assert packed["name"].dtype.kind in ("U", "S")  # string dtype
    # assert all(packed[k].dtype == np.float64 for k in ("S1", "lo", "hi", "ST"))
    # assert packed["name"].shape == (3,)
    # for k in ("S1", "lo", "hi", "ST"):
    #     assert packed[k].shape == (3,)

    # ---------- from_kwargs + basic properties
    res = sa.SobolResult(names, S1, lo, hi, ST)
    assert len(res) == 3
    npt.assert_array_equal(res.name, np.array(names, dtype=str))
    npt.assert_allclose(res.S1, np.array(S1, dtype=np.float64))
    npt.assert_allclose(res.lo, np.array(lo, dtype=np.float64))
    npt.assert_allclose(res.hi, np.array(hi, dtype=np.float64))
    npt.assert_allclose(res.ST, np.array(ST, dtype=np.float64))

    # ---------- items order & contents
    items = list(res.items)
    assert [k for k, _ in items] == ["name", "S1", "lo", "hi", "ST"]
    assert items[1][0] == "S1" and items[1][1].shape == (3,)

    # ---------- val_items excludes 'name'
    val_items = list(res.val_items)
    assert [k for k, _ in val_items] == ["S1", "lo", "hi", "ST"]

    # ---------- take
    sub = res.param_subset([2, 0])  # reorder & subset
    assert len(sub) == 2
    npt.assert_array_equal(sub.name, np.array(["X3", "X1"], dtype=str))
    npt.assert_allclose(sub.S1, [0.6, 0.3])

    # ---------- reorder (same as take with a full permutation)
    perm = [1, 2, 0]
    reo = res.reorder(perm)
    npt.assert_array_equal(reo.name, np.array([names[i] for i in perm], dtype=str))
    npt.assert_allclose(reo.S1, np.array([S1[i] for i in perm], dtype=np.float64))

    # ---------- as_dict round-trip
    d = res.as_dict()
    assert set(d.keys()) == {"name", "S1", "lo", "hi", "ST"}
    npt.assert_array_equal(d["name"], res.name)
    npt.assert_allclose(d["S1"], res.S1)
    npt.assert_allclose(d["lo"], res.lo)
    npt.assert_allclose(d["hi"], res.hi)
    npt.assert_allclose(d["ST"], res.ST)

    # ---------- pickle round-trip
    blob = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    res2 = pickle.loads(blob)

    # equal fields
    npt.assert_array_equal(res2.name, res.name)
    npt.assert_allclose(res2.S1, res.S1)
    npt.assert_allclose(res2.lo, res.lo)
    npt.assert_allclose(res2.hi, res.hi)
    npt.assert_allclose(res2.ST, res.ST)

def test_prepare_function_and_plot(smart_tmp_path):
    # -----------------------
    # Make a mock SobolResult
    # -----------------------
    # Intentionally unsorted; function should sort by S1 desc.
    names = ["X1", "X2", "X3", "X4", "X5"]
    S1 = np.array([0.25, 0.50, 0.05, 0.15, 0.05])    # sorted → [0.50, 0.25, 0.15, 0.05, 0.05]
    lo = S1 - 0.02                                   # mock lower CI bound
    hi = S1 + 0.02                                   # mock upper CI bound
    ST = S1 + np.array([0.20, 0.10, 0.05, 0.05, 0.02])  # some extra totals

    res = sa.SobolResult(names, S1, lo, hi, ST)

    # -----------------------
    # Prepare with threshold
    # -----------------------
    thr = 0.89
    stacked = sa.get_sensitive_params(res, threshold=thr)

    # After sorting by S1 desc, cum S1 = [0.50, 0.75, 0.90, 0.95, 1.00]
    # First index where cum > 0.89 is i=2 → kept 3 entries, drop last 2 → residual
    assert len(stacked) == 4  # 3 kept + 1 residual
    assert stacked.name[-1] == "residual"

    # Kept are the top-3 by S1: [0.50 (X2), 0.25 (X1), 0.15 (X4)]
    kept_names_expected = ["X2", "X1", "X4"]
    npt.assert_array_equal(stacked.name[:-1], np.array(kept_names_expected, dtype=str))

    # Residual is sum over dropped indices: the smallest two S1 are 0.05 (X3) and 0.05 (X5) => 0.10
    dropped_S1_sum = 0.05 + 0.05
    assert pytest.approx(stacked.S1[-1], rel=0, abs=1e-12) == dropped_S1_sum

    # The stacked S1 total equals original S1 total
    assert pytest.approx(stacked.S1.sum(), rel=0, abs=1e-12) == pytest.approx(res.S1.sum())

    # Do the same check for lo/hi/ST (residual = sum over dropped)
    order = np.argsort(-res.S1)
    S1_sorted = res.S1[order]
    lo_sorted = res.lo[order]
    hi_sorted = res.hi[order]
    ST_sorted = res.ST[order]
    cum = np.cumsum(S1_sorted)
    kept_len = np.where(cum > thr)[0][0] + 1
    dropped_slice = slice(kept_len, None)

    resid_lo_expected = lo_sorted[dropped_slice].sum()
    resid_hi_expected = hi_sorted[dropped_slice].sum()
    resid_ST_expected = ST_sorted[dropped_slice].sum()

    assert pytest.approx(stacked.lo[-1], rel=0, abs=1e-12) == resid_lo_expected
    assert pytest.approx(stacked.hi[-1], rel=0, abs=1e-12) == resid_hi_expected
    assert pytest.approx(stacked.ST[-1], rel=0, abs=1e-12) == resid_ST_expected

    # -----------------------
    # Smoke-test the single plot (no file, no show)
    # -----------------------
    sa.sobol_plot(stacked, base_width=0.6, max_extra_ratio=1.5, ci_color="k", show=False)

    # -----------------------
    # Second plot: two results, write to file
    # -----------------------
    # Make a slightly different second result
    S1_b = np.array([0.30, 0.45, 0.05, 0.10, 0.10])
    lo_b = S1_b - 0.02
    hi_b = S1_b + 0.02
    ST_b = S1_b + np.array([0.15, 0.12, 0.05, 0.04, 0.03])
    res_b = sa.SobolResult(names, S1_b, lo_b, hi_b, ST_b)
    stacked_b = sa.get_sensitive_params(res_b, threshold=thr)

    out = smart_tmp_path / "two_stacks.pdf"
    sa.sobol_plot([stacked, stacked_b], x=["case A", "case B"], fname=str(out), show=False)

    assert out.exists() and out.stat().st_size > 0