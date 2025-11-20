from typing import *
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

matplotlib.use("Agg")
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages



def plot_conc_timeseries(
    ds_stat: xr.Dataset,
    var_name: str,
    *,
    figsize=(11, 5),
):
    """
    Figure 1: time-series of conc_q99_XYZ statistics from ds_stat:
      - shaded band: q025..q975
      - mean curve
      - mean ± std (dashed)
      - median (dotted)

    Expects in ds_stat:
      sim_time, conc_q99_XYZ_q025, conc_q99_XYZ_q500, conc_q99_XYZ_q975, conc_q99_XYZ_mean, conc_q99_XYZ_std
    """
    x_label: str = "Simulation time",
    y_label: str = f"{var_name}_q99_XYZ",
    logx: bool = False
    logy: bool = False

    t = np.asarray(ds_stat["sim_time"].values)
    q025 = np.asarray(ds_stat[f"{var_name}_q99_XYZ_q025"].values, dtype=float)
    q500 = np.asarray(ds_stat[f"{var_name}_q99_XYZ_q500"].values, dtype=float)
    q975 = np.asarray(ds_stat[f"{var_name}_q99_XYZ_q975"].values, dtype=float)
    mean = np.asarray(ds_stat[f"{var_name}_q99_XYZ_mean"].values, dtype=float)
    std  = np.asarray(ds_stat[f"{var_name}_q99_XYZ_std"].values, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    print(f"Plotting conc_q99_XYZ timeseries t[{t.shape}]: ", q025, q975)
    ax.fill_between(t, q025, q975, alpha=0.2, label="q[0.025, 0.975]")
    ax.plot(t, q025, alpha=1, label="q[0.025]")
    ax.plot(t, q975, alpha=1, label="q[0.975]")
    ax.plot(t, mean, lw=1.75, label="mean")
    #ax.plot(t, mean - std, lw=1.0, ls="--", label="mean ± std")
    ax.plot(t, mean + std, lw=1.0, ls="--")
    ax.plot(t, q500, lw=1.0, ls=":", label="median (q0.5)")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_xscale("log")
    if not var_name.startswith("log"):
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_sobol_time_and_agg(
    sobol_time: xr.Dataset,   # select_sobol(...) for conc_q99_XYZ (time-dependent)
    sobol_agg: xr.Dataset,    # select_sobol(...) for conc_q99 (time-independent)
    var_name: str,
    *,
    figsize=(12, 5),
    si_ci_level: float = 0.90,
):
    """
    Figure 2:
      - Left: stacked-area SIs vs time (from sobol_time["SI"])
      - Right: stacked aggregated bar (from sobol_agg["SI"], with SI_ci lower-only error)
        and S1/ST markers for S1 entries (labels without '×' and not 'others').

    Assumes:
      sobol_time: coords ('group','sim_time'); vars 'SI' (and optionally 'ST')
      sobol_agg : coords ('group','bound'); vars 'SI','SI_ci' and optionally 'ST'
    """
    x_label: str = "Simulation time",
    sobol_agg = sobol_agg.squeeze('aux', drop=True)  # remove 'aux' dim 
    # Sort by aggregated SI descending (controls both left order and right stacking)
    labels = sobol_agg["group"].values.astype(str)
    SI_agg = np.asarray(sobol_agg["SI"].values, dtype=float)           # (P,)
    order = np.argsort(-SI_agg)
    labels_s = labels[order]
    SI_agg_s = SI_agg[order]
    SI_ci_s = np.asarray(sobol_agg["SI_ci"].values, dtype=float)[order, :]  # (P, 2) [low, high]

    ST_agg_src = sobol_agg.get("ST")
    ST_agg_s = (np.asarray(ST_agg_src.values, dtype=float)[order]
                if isinstance(ST_agg_src, xr.DataArray) else np.zeros_like(SI_agg_s))

    # Time-dependent SI arranged in the same order
    sobol_time = sobol_time.squeeze('aux', drop=True)  # remove 'aux' dim 
    t = np.asarray(sobol_time["sim_time"].values)
    SI_t = np.asarray(sobol_time["SI"].sel(group=labels_s).values, dtype=float)  # (P, T)

    # Colors (special-case "others" as gray)
    cmap = plt.get_cmap("tab20")
    colors = [(0.6, 0.6, 0.6, 1.0) if lab == "others" else cmap(i % 20)
              for i, lab in enumerate(labels_s)]

    # Layout: left stacked area, right thin aggregated bar
    right_plot_frac = 0.14
    legend_space_frac = 0.16
    left_frac = max(1e-3, 1.0 - right_plot_frac - legend_space_frac)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[left_frac, right_plot_frac], wspace=0.25)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # Left: stacked area over time
    axL.stackplot(t, SI_t, colors=colors, labels=list(labels_s), linewidth=0.5)
    axL.set_ylim(0.0, 1.0)
    axL.set_xlim(t.min(), t.max())
    axL.set_ylabel(f"Sobol index of {var_name}")
    axL.set_xlabel(x_label)
    axL.grid(axis="y", alpha=0.2)

    # Right: stacked aggregated bar at x=0
    x0, width = 0.0, 0.6
    bottom = 0.0
    ci_proxy = Line2D([], [], color="k", linestyle="-", label=f"CI ({int(round(si_ci_level*100))}%)")

    for i in range(len(labels_s)):
        axR.bar([x0], [SI_agg_s[i]], bottom=[bottom], width=width, color=colors[i])

        # Lower-only error bar from top of segment down to 'low' bound
        low_i = float(SI_ci_s[i, 0])
        top_i = bottom + SI_agg_s[i]
        if 0.0 < low_i < SI_agg_s[i]:
            axR.errorbar([x0], [top_i], yerr=[[top_i - low_i], [0.0]],
                         fmt="none", ecolor="k", elinewidth=1.2, capsize=3, capthick=1.2)

        # S1/ST markers for S1 labels (no '×' and not 'others')
        lab = str(labels_s[i])
        if ("×" not in lab) and (lab != "others") and (ST_agg_s[i] > 0.0):
            x_tri = x0 - width/2.0 + SI_agg_s[i] * width  # ▲ at S1_agg
            x_star = x0 - width/2.0 + ST_agg_s[i] * width # ★ at ST_agg
            axR.plot([x_tri], [bottom], marker="^", markersize=6, color="k", linestyle="none")
            axR.plot([x_star], [bottom], marker="*", markersize=7, color="k", linestyle="none")

        bottom += SI_agg_s[i]

    axR.set_ylim(0.0, 1.0)
    axR.set_xlim(x0 - 1.0, x0 + 1.0)
    axR.set_xticks([x0])
    axR.set_xticklabels(["aggregated"])
    axR.set_xlabel("")
    axR.grid(axis="y", alpha=0.2)

    # Legend on the right outside the axes (single column)
    proxies = [Patch(facecolor=colors[i], edgecolor="none", label=str(labels_s[i]))
               for i in range(len(labels_s))]
    proxies.append(ci_proxy)
    labels_legend = list(labels_s) + [ci_proxy.get_label()]
    fig.legend(handles=proxies, labels=labels_legend,
               loc="center left", bbox_to_anchor=(0.98, 0.5), ncol=1, frameon=False,
               title="Parameter / Interaction")

    fig.tight_layout()
    return fig, (axL, axR)


def save_conc_and_si_pdf(
    ds_stat: xr.Dataset,
    sobol_time: xr.Dataset,
    sobol_agg: xr.Dataset,
    var_name: str,
    *,
    figsize=(11, 5),
    si_ci_level: float = 0.90,
    out_pdf_path: str | Path = "conc_and_si.pdf",
):
    """
    Convenience: render the two figures and save each on its own page of a single PDF.
    """

    with PdfPages(out_pdf_path) as pdf:
        fig1, _ = plot_conc_timeseries(ds_stat, var_name, figsize=figsize)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)
        if sobol_time is not None and sobol_agg is not None:
            fig2, _ = plot_sobol_time_and_agg(sobol_time, sobol_agg, var_name,
                                            figsize=figsize, si_ci_level=si_ci_level)
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)







def raw_conc_plot(ds: xr.Dataset, sample_dim: Set[str], space_dim: Set[str], time_dim: str,
                  output_pdf: Path| str = 'conc_ecdf.pdf', eps: float = 1e-12) -> None:
    """
    For each time step, compute  ECDF/quantile function conc_quantiles(space_dim_rank)
    for each sample and sort the samples by  integral of conc between quantiles (top - 10) to top.
    top quantile is 1 - ECDF(0).
    Resulting 2D map of conc quantiles is plotted as color map with values given by log(conc)
    negative values trimmed to zero.
    The top quantile will be marked by red line, viridis color map used
    """
    matplotlib.use("Agg")
    if "conc" not in ds:
        raise ValueError("Dataset must contain variable 'conc'.")
    conc = ds["conc"]

    s_dims = tuple(sample_dim)
    x_dims = tuple(space_dim)
    for d in s_dims + x_dims + (time_dim,):
        if d not in conc.dims:
            raise ValueError(f"Dimension '{d}' not found in conc.")

    # Arrange dims, stack to (time, sample, space)
    c = conc.transpose(time_dim, *s_dims, *x_dims)
    c2 = c.stack(sample=s_dims, space=x_dims)  # (time, sample, space)

    # Rechunk to make time-slicing cheap when plotting (if dask-backed)
    if hasattr(c2.data, "chunks"):
        c2 = c2.chunk({"space": -1})
    # if hasattr(c2.data, "chunks"):
    #     try:
    #         c2 = c2.chunk({time_dim: 1})
    #     except Exception:
    #         pass

    # 1) Quantile curves via sort along 'space' (vectorized & dask-friendly)
    q_sorted = xr.apply_ufunc(
        np.sort, c2,
        input_core_dims=[["space"]],
        output_core_dims=[["space"]],
        vectorize=True,
        dask="parallelized",
    )


    # 2) p_top per (time, sample): fraction of strictly positive values
    p_minus = (c2 <= 0.0).mean(dim="space")  # F(0)
    p_zero = p_minus
    # p_top = 1.0 - p_minus  # 1 - F(0)
    #p_sort =

    # 3) Index range for last 10% below p_top (vectorized)
    n_space = q_sorted.sizes["space"]
    k_top = n_space  # (time, sample)
    p_low = 0.9
    k_low = np.floor(p_low * n_space).clip(min=0, max=n_space - 1)  # (time, sample)
    kv = k_top.values
    print(f"top quantile range: [{np.min(kv)}, {np.max(kv)}]")

    # Broadcast mask for [k_low, k_top] along 'space'
    idx_space = xr.DataArray(np.arange(n_space), dims=["space"])
    mask = (idx_space >= k_low) & (idx_space <= k_top)  # (time, sample, space)

    # 4) Tail score ~ sum over last-10% quantiles (uniform ECDF spacing)
    tail_score = q_sorted.where(mask, 0.0).sum(dim="space") / n_space  # (time, sample)

    # 5) Order samples per time by descending tail score (vectorized)
    ts_np = tail_score.compute().values  # shape (T, S)
    order_np = np.argsort(-ts_np, axis=1)  # descending per time row

    order = xr.DataArray(
        order_np,
        coords={time_dim: tail_score[time_dim], "sample": tail_score["sample"]},
        dims=(time_dim, "sample"),
    )
    #order = tail_score.argsort(dim="sample", ascending=False)  # (time, sample)


    # 6) Transform to plot values once (vectorized) and persist
    plot_vals = xr.apply_ufunc(
        np.log, xr.ufuncs.maximum(q_sorted, 0.0) + eps,
        dask="parallelized",
    )

    # Persist heavy intermediates
    try:
        plot_vals = plot_vals.persist()
        order = order.persist()
    except Exception:
        pass

    # 7) Per-time color limits (independent)
    vmin_t = plot_vals.min(dim=("sample", "space"))
    vmax_t = plot_vals.max(dim=("sample", "space"))
    vmin_vals = vmin_t.compute().values
    vmax_vals = vmax_t.compute().values
    times = plot_vals.coords[time_dim].values

    # 8) Save all time slices into a single PDF, two plots per page
    with PdfPages(output_pdf) as pdf:
        n_time = len(times)
        n_pages = int(np.ceil(n_time / 2))

        ti = 0  # time index
        for page in range(n_pages):
            # Create a figure with two vertical panels
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), constrained_layout=True)

            for row in range(2):
                if ti >= n_time:
                    # Hide unused panel on the last page (odd number of times)
                    axes[row].axis("off")
                    continue

                tval = times[ti]
                pv_t = plot_vals.sel({time_dim: tval})  # (sample, space)
                ord_t = order.sel({time_dim: tval}).values  # plain NumPy array of ints
                pv_t_s = pv_t.isel(sample=ord_t)  # positional indexing, no coord conflict

                p_top_t = p_zero.sel({time_dim: tval})  # (sample,)
                p_top_sorted = p_top_t.isel(sample=ord_t).compute().values

                arr = pv_t_s.compute().values  # bring this slice to NumPy
                vmin = max(float(vmin_vals[ti]), -25)
                vmax = float(vmax_vals[ti])
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin = vmax - 1.0

                ax = axes[row]
                im = ax.imshow(
                    arr, aspect="auto", origin="lower", cmap="viridis",
                    vmin=vmin, vmax=vmax, extent=[0.0, 1.0, 0, arr.shape[0]]
                )
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("log(max(conc,0)+eps)")

                # Red vertical line at the "top" reference (normalized x=1)
                y = np.arange(arr.shape[0]) + 0.5
                ax.fill_betweenx(p_top_sorted, y, color="red", linewidth=0.5, )

                ax.set_xlabel("space quantile p")
                ax.set_ylabel("sample (sorted by tail integral)")
                ax.set_title(f"Quantile map at {time_dim}={tval}")

                ti += 1

            pdf.savefig(fig)
            plt.close(fig)
    print("cond ECDF plot DONE")




def maxmin_cover_for_sorted(vals: np.ndarray, M: int) -> np.ndarray:
    """
    Given a sorted 1D array vals of length N and an integer M with
    1 < M < N, select M indices by:
      - always including 0 and N-1
      - adding the lower bounds of the M-2 largest gaps between
        consecutive values, excluding the first interval (since 0 is in).

    Parameters
    ----------
    vals : np.ndarray
        1D sorted array (ascending).
    M : int
        Number of points to select, with 1 < M < N.

    Returns
    -------
    selected : np.ndarray of int
        Sorted array of selected indices into vals (length M).
    """
    vals = np.asarray(vals)
    N = vals.size
    if N == 0:
        return np.array([], dtype=int)
    if N == 1:
        return np.array([0], dtype=int)
    M = min(M, N)
    assert vals.ndim == 1

    assert 1 < M <= N


    # 1) consecutive gaps
    diffs = np.diff(vals)            # shape (N-1,)

    # 2) we exclude the first interval (between 0 and 1),
    #    since index 0 is always selected
    inner_needed = M - 2
    inner_diffs = diffs[1:]          # intervals starting at indices 1..N-2

    # 3) indices of the (M-2) largest inner gaps
    part_idx = np.argpartition(inner_diffs, -inner_needed)[-inner_needed:]
    inner_indices = part_idx + 1     # shift because we skipped diffs[0]

    # 4) add first and last points
    selected = np.concatenate(([0], inner_indices, [N - 1]))
    selected.sort()

    # with the asserts above, length should be exactly M
    return selected


def outliers_filter(scores_t: xr.DataArray,
                    i_eval,
                    q: float = 0.99) -> np.ndarray:
    """
    Filter and flag outliers in a 1D array of scores_t.

    Mask codes:
      4 → scores_t == -inf
      3 → scores_t < 0
      2 → scores_t == 0
      1 → MAD-based outlier on log(scores_t)
      0 → OK
    """
    if scores_t.ndim != 1:
        raise ValueError("scores_t must be 1D over a MultiIndex dimension.")

    scores = np.asarray(scores_t.values)
    N = scores.size
    mask = np.zeros(N, dtype=int)

    # MultiIndex as array of tuples, shape (N,)
    dim = scores_t.dims[0]
    mi = scores_t.indexes[dim]

    # Local helper: print list of triplets (i, *mi[i]) for given indices
    def print_triplets(label: str, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            return
        # mi[indices] is an array of tuples → stack to 2D
        i_eval_vals = np.asarray(i_eval.values[indices])

        level_vals = np.stack(mi.values[indices])           # (k, n_levels)
        triplets = np.column_stack((i_eval_vals, level_vals))  # (k, 1+n_levels)
        print(label, "#=", len(indices))
        print(" i_eval | " + " | ".join(mi.names))
        print(triplets)

    # 4) scores_t == -inf
    idx_neg_inf = np.isneginf(scores)
    ii = np.where(idx_neg_inf)[0]
    print_triplets("scores_t == -inf:", ii)
    mask[idx_neg_inf] = 4

    # 3) scores_t < 0
    idx_neg = (scores < 0) & ~idx_neg_inf
    ii = np.where(idx_neg)[0]
    print_triplets("scores_t < 0:", ii)
    mask[idx_neg] = 3

    # 2) scores_t == 0
    idx_zero = (scores == 0) & ~(idx_neg_inf | idx_neg)
    ii = np.where(idx_zero)[0]
    print_triplets("scores_t == 0:", ii)
    mask[idx_zero] = 2

    # 1) MAD on log(scores), threshold derived from q
    remaining = (mask == 0) & np.isfinite(scores) & (scores > 0)

    if np.any(remaining):
        vals = scores[remaining]
        log_vals = np.log(vals)

        median = np.median(log_vals)
        abs_dev = np.abs(log_vals - median)
        mad = np.median(abs_dev)

        if mad > 0:
            # two-sided q → tail probability
            tail = (1.0 - q) / 2.0
            z = stats.norm.ppf(1.0 - tail)
            mad_z_threshold = z / 0.6745

            mad_z = 0.6745 * (log_vals - median) / mad
            outliers = np.abs(mad_z) > mad_z_threshold

            rem_idx = np.where(remaining)[0]
            out_idx = rem_idx[outliers]

            print_triplets("MAD-outlier:", out_idx)
            mask[out_idx] = 1

    return mask


def conc_tail_ecdf_plot(
    ds: xr.Dataset,
    sample_dim: set[str],
    space_dim: set[str],
    time_dim: str,
    output_pdf: str | Path = "conc_tail_ecdf.pdf",
    p_tail: float = 0.10,       # top 10%
    frac_samples: float = 1.0   # fraction of samples considered for plotting
) -> None:

    if "conc" not in ds:
        raise ValueError("Dataset must contain variable 'conc'.")
    # conc = ds["conc"]
    # i_eval = ds['i_eval']

    s_dims = tuple(sample_dim)
    x_dims = tuple(space_dim)
    for d in s_dims + x_dims + (time_dim,):
        if d not in ds.dims:
            raise ValueError(f"Dimension '{d}' not found in conc.")

    # Move dims → (time, sample, space)
    c_t = ds["conc"].transpose(time_dim, *s_dims, *x_dims)
    conc = c_t.stack(sample=s_dims, space=x_dims)
    i_eval = ds['i_eval'].stack(sample=s_dims)

    if hasattr(conc.data, "chunks"):
        conc = conc.chunk({"space": -1})

    # 1. Quantile / ECDF via sorting
    c_sorted = xr.apply_ufunc(
        np.sort,
        conc,
        input_core_dims=[["space"]],
        output_core_dims=[["space"]],
        vectorize=True,
        dask="parallelized",
    )

    n_space = c_sorted.sizes["space"]
    idx_space = np.arange(n_space)
    p_all = (idx_space + 0.5) / n_space  # ECDF probabilities
    p_all_da = xr.DataArray(p_all, dims=["space"])

    # 2. Tail mask = top p_tail
    p_low = 1.0 - p_tail
    tail_mask = p_all_da >= p_low

    c_space_tail = c_sorted.sel(space=tail_mask)
    p_tail_da = p_all_da.sel(space=tail_mask)

    q_tail_score = 0.99
    q_space_tail = 1.0 - (1.0 - q_tail_score) / p_tail
    #print("q_space_tail", q_space_tail)
    assert 0.0 <= q_tail_score <= 1.0

    # 3. tail_score = median of tail
    tail_score = c_space_tail.quantile(q_space_tail, dim="space")
    tail_score = tail_score.where(np.isfinite(tail_score), -np.inf).compute()

    # tail_score_np = tail_score.compute().values
    # tail_score_np = np.where(np.isnan(tail_score_np), -np.inf, tail_score_np)

    times = c_sorted.coords[time_dim].values
    n_time = len(times)
    n_samples = c_sorted.sizes["sample"]
    n_keep = max(1, int(np.ceil(frac_samples * n_samples)))

    # max lines per time step
    max_lines = 50
    assert n_keep > max_lines

    y_tail = p_tail_da.values  # ECDF y-values for tail

    with PdfPages(output_pdf) as pdf:
        for ti, tval in enumerate(times):
            print("Processing time: ", ti)
            conc_tail_t = c_space_tail.sel({time_dim: tval})
            scores_t = tail_score[ti, :].compute()  # shape (n_samples,)
            scores_t_np = scores_t.values
            outliers = outliers_filter(scores_t, i_eval, q=0.99)
            valid_idx = np.where(outliers <= 1)[0]
            scores_t_valid = scores_t_np[valid_idx]
            outlier_valid = (outliers[valid_idx] == 1)

            # indices *within* scores_t_valid of the top n_keep values
            top_idx = valid_idx[np.argsort(scores_t_valid)[-n_keep:]]
            top_scores = scores_t_np[top_idx]

            # top_scores are still sorted
            # 3) Use maxmin_cover_for_sorted to select at most max_lines
            sel_local = maxmin_cover_for_sorted(np.log(top_scores), max_lines)
            #print(scores_t_np[top_idx])
            #print(scores_t_np[top_idx[sel_local]])

            #keep_scores = top_scores[sel_local]

            norm = mcolors.Normalize(vmin=0, vmax=len(top_idx))
            cmap = plt.cm.get_cmap("viridis")
            # ---------------------------------------------------

            fig, ax = plt.subplots(figsize=(12, 8))

            # Build list of vertices for each line, just like repeated plot(x_tail, y_tail)
            print("  build segments")
            top_samples = top_idx[sel_local]
            x_tail = conc_tail_t.isel(sample=top_samples).values
            # (n_top_samples, n_tail_points)
            Y2 = np.broadcast_to(y_tail, x_tail.shape)  # from (n_tail_points,)
            segments = np.stack((x_tail, Y2), axis=-1)

            # Colors per line from scores
            line_colors = cmap(norm(sel_local))

            # One LineCollection instead of many plot() calls
            lc = LineCollection(
                segments, colors=line_colors,
                linewidths=1.0, alpha=0.8,
            )
            print("  add lines")
            ax.add_collection(lc)

            print("  scatter")
            # Single scatter call for all markers
            X = scores_t_np[top_samples]
            Y = np.full_like(X, q_tail_score)
            top_i_eval = i_eval.isel(sample=top_samples).values
            if not np.all(np.isfinite(X)):
                print(    "Some keep_scores are not finite: ", len(np.isfinite(X)))
            dot_size = 10
            ax.scatter(
                X, Y,
                c='red', s=dot_size, zorder=3,
            )
            dot_diam = np.sqrt(dot_size)
            # Add tiny index labels on top of each point
            for i, (x, y, idx) in enumerate(zip(X, Y, top_i_eval)):
                ax.annotate(
                    text =str(idx),
                    xy = (x,y),
                    xytext=(0, -1.5 - (i % 7)),
                    textcoords='offset fontsize',
                    fontsize=2,  # adjust to match your dot size
                    ha='center',
                    va='center',
                    color='black',  # contrasts with the red dot
                    zorder=4,
                    clip_on=True,
                )
            ax.set_xscale("log")

            # --- Colorbar ---
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            # ----------------

            cbar.set_label("tail_score = median of top 10%")
            ax.set_xlabel("concentration")
            ax.set_ylabel("ECDF")
            ax.set_ylim(p_low, 1.0)
            ax.set_title(
                f"Tail ECDFs at {time_dim}={tval}\n"
                f"Top {100*frac_samples:.1f}% samples; tail = top {100*p_tail:.1f}%"
            )

            pdf.savefig(fig)
            plt.close(fig)

    print("Tail ECDF plot DONE:", output_pdf)