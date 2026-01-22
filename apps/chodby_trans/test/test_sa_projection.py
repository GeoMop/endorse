# test_sobol_projection.py
#
# Pytest unit tests for "project each bootstrap draw, then form CIs".
#
# Assumptions:
#   - You have a function `project_sobol_draws(ds_boot)` that takes an xarray.Dataset
#     with variables:
#         S1(group, boot, sim_time)
#         ST(group, boot, sim_time)
#         S2(group, group2, boot, sim_time)
#     and returns a Dataset with the same variables projected so that Sobol constraints
#     hold for EVERY (boot, sim_time) slice.
#
# SciPy constraint machinery reference:
#   scipy.optimize.minimize(..., method="trust-constr") supports Bounds + LinearConstraint.
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html  (trust-constr + constraints)
# xarray quantile reference:
#   https://docs.xarray.dev/en/stable/generated/xarray.DataArray.quantile.html

from __future__ import annotations

import numpy as np
import xarray as xr
import pytest

# Change this import to wherever you put the code
# from yourpkg.sobol_projection import project_sobol_draws, project_one
from yourpkg.sobol_projection import project_sobol_draws, project_one  # noqa: F401


def _upper_pairs(d: int):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def _assert_sobol_constraints_hold(
    S1: np.ndarray,  # (d,)
    ST: np.ndarray,  # (d,)
    S2: np.ndarray,  # (d,d)
    eps: float = 1e-10,
):
    d = S1.size
    assert ST.shape == (d,)
    assert S2.shape == (d, d)

    # box bounds (allow tiny numerical tolerances)
    assert np.nanmin(S1) >= -eps
    assert np.nanmin(ST) >= -eps
    assert np.nanmin(S2) >= -eps
    assert np.nanmax(S1) <= 1.0 + eps
    assert np.nanmax(ST) <= 1.0 + eps
    assert np.nanmax(S2) <= 1.0 + eps

    # ST_i >= S1_i
    assert np.all(ST + eps >= S1)

    # enforce diagonal is 0 (your convention; if you keep diag undefined, adapt)
    assert np.allclose(np.diag(S2), 0.0, atol=1e-12)

    # per-input: S1_i + sum_{j!=i} S2_ij <= ST_i
    offdiag = S2.copy()
    np.fill_diagonal(offdiag, 0.0)
    per_i = S1 + offdiag.sum(axis=1)
    assert np.all(per_i <= ST + 1e-8)

    # global: sum S1 + sum_{i<j} S2_ij <= 1
    s2_upper = sum(S2[i, j] for i, j in _upper_pairs(d))
    assert float(S1.sum() + s2_upper) <= 1.0 + 1e-8


def _make_intentionally_infeasible_boot_dataset(
    d: int = 4,
    n_boot: int = 200,
    n_time: int = 18,
    seed: int = 123,
) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    groups = np.array([f"g{i}" for i in range(d)], dtype=object)
    sim_time = np.arange(n_time, dtype=int)
    boot = np.arange(n_boot, dtype=int)

    # Start with "wild" values that violate constraints (negatives, >1, ST < S1, etc.)
    S1 = rng.normal(loc=0.2, scale=0.8, size=(d, n_boot, n_time))
    ST = rng.normal(loc=0.1, scale=0.8, size=(d, n_boot, n_time))

    # Force some blatant violations: ST < S1 frequently
    ST -= 0.8 * np.abs(rng.normal(size=ST.shape))

    # S2: allow large positive/negative to violate positivity and sum constraints
    S2 = rng.normal(loc=0.3, scale=0.9, size=(d, d, n_boot, n_time))
    # Symmetrize, zero diagonal
    for i in range(d):
        S2[i, i, :, :] = 0.0
        for j in range(i + 1, d):
            S2[j, i, :, :] = S2[i, j, :, :]

    ds = xr.Dataset(
        data_vars=dict(
            S1=(("group", "boot", "sim_time"), S1),
            ST=(("group", "boot", "sim_time"), ST),
            S2=(("group", "group2", "boot", "sim_time"), S2),
        ),
        coords=dict(
            group=groups,
            group2=groups,
            boot=boot,
            sim_time=sim_time,
        ),
    )
    return ds


@pytest.mark.parametrize("d,n_boot,n_time", [(3, 120, 18), (5, 80, 18)])
def test_project_sobol_draws_enforces_constraints_for_every_boot_and_time(d, n_boot, n_time):
    ds_bad = _make_intentionally_infeasible_boot_dataset(d=d, n_boot=n_boot, n_time=n_time, seed=2026)

    ds_proj = project_sobol_draws(ds_bad)

    # shape invariants
    assert ds_proj["S1"].shape == ds_bad["S1"].shape
    assert ds_proj["ST"].shape == ds_bad["ST"].shape
    assert ds_proj["S2"].shape == ds_bad["S2"].shape

    # check every (boot, sim_time) slice is feasible
    S1p = ds_proj["S1"].to_numpy()
    STp = ds_proj["ST"].to_numpy()
    S2p = ds_proj["S2"].to_numpy()

    for b in range(n_boot):
        for t in range(n_time):
            _assert_sobol_constraints_hold(
                S1=S1p[:, b, t],
                ST=STp[:, b, t],
                S2=S2p[:, :, b, t],
            )


def test_projected_point_estimate_and_projected_ci_endpoints_are_feasible():
    """
    If you want the CI endpoints themselves to respect constraints, you must
    project them as vectors (componentwise quantiles do not preserve coupled constraints).
    """
    d, n_boot, n_time = 4, 200, 18
    ds_bad = _make_intentionally_infeasible_boot_dataset(d=d, n_boot=n_boot, n_time=n_time, seed=7)
    ds_proj = project_sobol_draws(ds_bad)

    # Point estimate (mean over boot), then project per time
    S1_mean = ds_bad["S1"].mean("boot")
    ST_mean = ds_bad["ST"].mean("boot")
    S2_mean = ds_bad["S2"].mean("boot")

    # Project the mean slice-by-slice in time
    S1p = np.empty((d, n_time))
    STp = np.empty((d, n_time))
    S2p = np.empty((d, d, n_time))
    for t in range(n_time):
        s1_t = S1_mean.isel(sim_time=t).to_numpy()
        st_t = ST_mean.isel(sim_time=t).to_numpy()
        s2_t = S2_mean.isel(sim_time=t).to_numpy()
        s1p_t, stp_t, s2p_t = project_one(s1_t, st_t, s2_t)  # your single-slice projector
        S1p[:, t] = s1p_t
        STp[:, t] = stp_t
        S2p[:, :, t] = s2p_t
        _assert_sobol_constraints_hold(s1p_t, stp_t, s2p_t)

    # CI endpoints from projected draws (componentwise quantiles)
    qlo, qhi = 0.025, 0.975
    S1_lo = ds_proj["S1"].quantile(qlo, dim="boot").to_numpy()
    S1_hi = ds_proj["S1"].quantile(qhi, dim="boot").to_numpy()
    ST_lo = ds_proj["ST"].quantile(qlo, dim="boot").to_numpy()
    ST_hi = ds_proj["ST"].quantile(qhi, dim="boot").to_numpy()
    S2_lo = ds_proj["S2"].quantile(qlo, dim="boot").to_numpy()
    S2_hi = ds_proj["S2"].quantile(qhi, dim="boot").to_numpy()

    # IMPORTANT: quantile endpoints are componentwise; coupled constraints (like global sums)
    # are not guaranteed. If you want endpoints feasible, project the *endpoint vectors*.
    for t in range(n_time):
        s1_lo_p, st_lo_p, s2_lo_p = project_one(S1_lo[:, t], ST_lo[:, t], S2_lo[:, :, t])
        s1_hi_p, st_hi_p, s2_hi_p = project_one(S1_hi[:, t], ST_hi[:, t], S2_hi[:, :, t])

        _assert_sobol_constraints_hold(s1_lo_p, st_lo_p, s2_lo_p)
        _assert_sobol_constraints_hold(s1_hi_p, st_hi_p, s2_hi_p)

        # Basic interval sanity: projected low <= projected high componentwise
        assert np.all(s1_lo_p <= s1_hi_p + 1e-10)
        assert np.all(st_lo_p <= st_hi_p + 1e-10)
        # S2: only check upper triangle to avoid duplicate comparisons
        for i, j in _upper_pairs(d):
            assert s2_lo_p[i, j] <= s2_hi_p[i, j] + 1e-10
