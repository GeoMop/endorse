from typing import *
import numpy as np
import matplotlib.pyplot as plt
from . import SobolResult


def get_sensitive_params(res: SobolResult, threshold: float = 0.8) -> SobolResult:
    """
    Sort by S1 (desc), keep entries until cumulative S1 > threshold (inclusive),
    fold the remainder into a single 'residual' by summing S1/lo/hi/ST over dropped.
    Returns a new SobolResult with kept entries + residual appended.
    """
    # sort by S1 descending
    order = np.argsort(-res.S1)
    res_sorted = res.reorder(order)

    # find cut length: include the item that makes cum > threshold
    cum = np.cumsum(res_sorted.S1)
    where = np.where(cum > threshold)[0]
    kept_len = (where[0] + 1) if where.size else len(cum)

    kept = res_sorted.param_subset(range(kept_len))
    dropped_idx = range(kept_len, len(res_sorted))

    if kept_len < len(res_sorted):
        dropped = res_sorted.param_subset(dropped_idx)
        # sum over dropped
        resid_name = np.array(["residual"])
        resid_S1 = np.array([dropped.S1.sum()])
        resid_lo = np.array([dropped.lo.sum()])
        resid_hi = np.array([dropped.hi.sum()])
        resid_ST = np.array([dropped.ST.sum()])

        # concatenate kept + residual
        stacked = SobolResult(
            name=np.concatenate([kept.name, resid_name]),
            S1=np.concatenate([kept.S1, resid_S1]),
            lo=np.concatenate([kept.lo, resid_lo]),
            hi=np.concatenate([kept.hi, resid_hi]),
            ST=np.concatenate([kept.ST, resid_ST]),
        )
        return stacked
    else:
        # nothing dropped → no residual
        return kept




def _lighten_color(rgb, factor=0.5):
    """Blend towards white by 'factor' (0..1)."""
    r, g, b, *rest = rgb
    r2 = 1 - factor*(1 - r)
    g2 = 1 - factor*(1 - g)
    b2 = 1 - factor*(1 - b)
    return (r2, g2, b2, *(rest[:1] if rest else []))


def estimate_centers_widths(
    results: Sequence[SobolResult],
    *,
    base_width: float = 0.6,
    max_extra_ratio: float = 1.5,
    gap: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate x-centers and total widths (main + max ragged) for each stacked bar.

    Returns
    -------
    centers : (m,) array
    widths  : (m,) array    # base_width + max ragged extension for each bar
    span    : float         # total data span = rightmost edge - leftmost edge
    """
    m = len(results)
    if m == 0:
        return np.array([]), np.array([]), 0.0

    eps = 1e-12
    widths = np.zeros(m, dtype=float)

    # Per-bar estimated total width
    for i, res in enumerate(results):
        S1, ST = res.S1, res.ST
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(S1 > eps, (ST - S1) / S1, 0.0)
        ratios = np.clip(ratios, 0.0, max_extra_ratio)
        widths[i] = base_width + base_width * (ratios.max() if ratios.size else 0.0)

    # Centers: left→right with gaps between total widths
    centers = np.zeros(m, dtype=float)
    for i in range(1, m):
        centers[i] = centers[i-1] + 0.5*widths[i-1] + gap + 0.5*widths[i]

    # Total span occupied by all bars (for xlim/figure sizing)
    left_edge = centers[0] - widths[0]/2.0
    right_edge = centers[-1] + widths[-1]/2.0
    span = right_edge - left_edge

    return centers, widths, span


def _plot_one_stacked_at(
    ax: plt.Axes,
    stacked: SobolResult,
    x_center: float,
    base_width: float,
    max_extra_ratio: float,
    ci_color: str,
) -> None:
    """Render one stacked bar (with CI tick & ragged width) at x_center."""
    names, S1, lo, ST = stacked.name, stacked.S1, stacked.lo, stacked.ST
    k = len(stacked)
    cmap = plt.get_cmap("tab20")

    colors = [cmap(i % 20) for i in range(k)]
    ragged_colors = [_lighten_color(colors[i], factor=0.65) for i in range(k)]

    bottom = 0.0
    eps = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(S1 > eps, (ST - S1) / S1, 0.0)
    ratios = np.clip(ratios, 0.0, max_extra_ratio)

    for i in range(k):
        h = float(max(0.0, S1[i]))
        if h <= 0:
            continue

        # main slice
        ax.bar(x_center, h, width=base_width, bottom=bottom,
               color=colors[i], edgecolor="none")

        # 90% CI LOWER branch at the slice top
        y_top = bottom + h
        ci_low = max(0.0, h - float(lo[i]))  # downward branch only
        ax.errorbar(
            x_center, y_top,
            yerr=np.array([[ci_low], [0.0]]),
            fmt="none", ecolor=ci_color, elinewidth=1.2, capsize=3, capthick=1.2
        )

        # ragged right: glued rectangle showing (ST - S1)/S1
        extra_w = base_width * ratios[i]
        if extra_w > 0:
            ax.bar(
                x_center + base_width/2.0 + extra_w/2.0, h, width=extra_w,
                bottom=bottom, color=ragged_colors[i], edgecolor="none"
            )

        bottom += h

    # legend (once per bar; caller sets labels once overall if desired)
    handles = [plt.Rectangle((0, 0), 1, 1, color=plt.get_cmap("tab20")(i % 20)) for i in range(k)]
    ax.legend(handles, list(names), loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


def sobol_plot(
    stacked: Union[SobolResult, Sequence[SobolResult]],
    *,
    x: Optional[Sequence[str]] = None,
    base_width: float = 0.6,
    max_extra_ratio: float = 1.5,
    ci_color: str = "k",
    ax: Optional[plt.Axes] = None,
    fname: Optional[str] = None,
    show: bool = False,
    gap: float = 0.2,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw one or more SobolResult stacks on a SINGLE axis.

    - 'stacked' may be a single SobolResult or a sequence; internally normalized to a list.
    - 'x' provides x-tick labels (defaults to "1..m").
    - Figure size is chosen from the **estimated total span** before creating the figure
      (unless an 'ax' is provided).
    - Saves to 'fname' if given; returns (fig, ax).
    """
    # Normalize input to a list
    results = list(stacked) if isinstance(stacked, (list, tuple)) else [stacked]
    m = len(results)

    # Estimate centers/widths/span BEFORE creating the figure
    centers, widths, span = estimate_centers_widths(
        results, base_width=base_width, max_extra_ratio=max_extra_ratio, gap=gap
    )

    # Create fig/ax (use span to choose width) unless an axis is provided
    if ax is None:
        # Heuristic: 6.0 in minimum; scale with span so labels remain readable
        fig_width = max(6.0, 2.2 * span / base_width)  # 2.2 is a readable scaling factor
        fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    else:
        fig = ax.figure

    # Draw each bar
    for xc, res in zip(centers, results):
        _plot_one_stacked_at(ax, res, xc, base_width, max_extra_ratio, ci_color)

    # Y limits based on the tallest stack among results
    max_s1_sum = max(float(r.S1.sum()) for r in results) if results else 1.0
    ax.set_ylim(0.0, max(1.0, max_s1_sum) * 1.02)

    # X limits to snug-fit all bars (+ small margins)
    if m > 0:
        left = centers[0] - widths[0]/2.0 - 0.1
        right = centers[-1] + widths[-1]/2.0 + 0.1
        ax.set_xlim(left, right)

    # X ticks / labels
    if x is None:
        labels = [str(i+1) for i in range(m)]
    else:
        if len(x) != m:
            raise ValueError(f"len(x) must equal number of results ({m}), got {len(x)}")
        labels = list(x)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Σ S1 (first-order mass)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    if fname:
        fig.savefig(fname, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


