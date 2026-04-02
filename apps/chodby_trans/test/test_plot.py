import numpy as np
import chodby_trans.plots as plots


def test_maxmin_cover_for_sorted():

    # 1) Simple evenly spaced case
    vals = np.linspace(0.0, 9.0, 10)  # [0,1,2,...,9], N=10
    M = 4
    idx = plots.maxmin_cover_for_sorted(vals, M)

    # Basic properties
    assert idx.ndim == 1
    assert len(idx) == M
    assert np.all(idx[:-1] < idx[1:])  # sorted
    assert idx[0] == 0
    assert idx[-1] == len(vals) - 1

    # For equal gaps, the largest-gap selection should roughly spread them
    # We don't enforce exact positions but ensure they're spaced out
    gaps = np.diff(idx)
    assert gaps.min() >= 1
    assert gaps.max() <= len(vals) - 1

    # 2) Clustered case: dense on left, sparse on right
    vals = np.concatenate([
        np.linspace(0.0, 1.0, 50, endpoint=False),  # dense cluster
        np.linspace(1.0, 10.0, 10)                  # sparse tail
    ])
    N = len(vals)
    M = 6
    idx = plots.maxmin_cover_for_sorted(vals, M)

    assert len(idx) == M
    assert idx[0] == 0
    assert idx[-1] == N - 1

    # Check that at least one interior point is in the sparse region
    # (i.e., large gaps near the right tail are recognized)
    interior_vals = vals[idx[1:-1]]
    assert np.any(interior_vals > 5.0)

    # 3) Random monotone increasing values: sanity checks
    rng = np.random.default_rng(42)
    base = np.sort(rng.random(100))   # strictly increasing [0,1]
    vals = base ** 2                  # some non-linear spacing
    M = 10
    idx = plots.maxmin_cover_for_sorted(vals, M)

    assert len(idx) == M
    assert idx[0] == 0
    assert idx[-1] == len(vals) - 1
    assert np.all(idx[:-1] < idx[1:])

    # Compare largest gap between selected points vs. naive uniform subsample:
    # largest-gaps method should not be *worse* in terms of max gap.
    naive_idx = np.round(
        np.linspace(0, len(vals) - 1, M)
    ).astype(int)
    naive_idx[0], naive_idx[-1] = 0, len(vals) - 1

    max_gap_selected = np.max(np.diff(vals[idx]))
    max_gap_naive = np.max(np.diff(vals[naive_idx]))

    assert max_gap_selected <= max_gap_naive + 1e-12  # small tolerance

    # 4) Assertion checks for invalid M (optional)
    # These should raise AssertionError due to 1 < M < N
    try:
        plots.maxmin_cover_for_sorted(vals, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError for M=1")

    try:
        plots.maxmin_cover_for_sorted(vals, len(vals))
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError for M=N")

    print("All tests for plots.maxmin_cover_for_sorted passed.")