from typing import *
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

#matplotlib.use("Agg")
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages


matplotlib.rcParams['hatch.linewidth'] = 6

def save_close_fig(fig: plt.Figure, pdf: PdfPages, fname):
    """
    Helper function for saving a figure - both individually and into PdfPages.
    :param fig: matplotlib.pyplot.Figure object
    :param pdf: PdfPages (path already known)
    :param fname: filepath to individual figure file
    :return:
    """
    fig.savefig(fname=fname, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)




def plot_conc_timeseries_distribution1(
    ds: xr.Dataset,
    *,
    Q: float = 0.05,
    n_slices: int = 7,
    hist_bins: tuple[int, int] = (220, 60),  # (time_bins, y_bins)
    max_extreme_lines: int = 20,
    figsize: tuple[float, float] = (16, 12),
    time_dim: str = "sim_time",
    sample_dims: tuple[str, str] = ("QMC", "IID"),
    # top-slices rendering controls (screen-space sizing, independent of log-x)
    slice_width_in: float = 0.75,  # inches
    slice_alpha: float = 0.25,
    slice_bins_y: int = 90,
):
    """
    Fixed vars in ds:
      - ds['log10_conc_q99']         : per-sample max over time (dims: QMC,IID) -> used for extremes via Q,1-Q
      - ds['log10_conc_q99_XYZ']     : time dependent values (dims: sim_time,QMC,IID) -> plotted as distribution

    Output:
      - bottom ax: 2D histogram (remaining samples) + extreme trajectories (out-of-Q)
      - top ax:   per-slice inset histograms (remaining samples), widths independent of log time scale
                 + quantile lines: Q, 0.25, 0.5, 0.75, 1-Q
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # --- asserts
    assert time_dim in ds, f"missing coord/dim {time_dim}"
    assert "log10_conc_q99" in ds, "missing variable 'log10_conc_q99'"
    assert "log10_conc_q99_XYZ" in ds, "missing variable 'log10_conc_q99_XYZ'"
    assert 0.0 < Q < 0.5, "Q must be in (0, 0.5)"
    assert all(d in ds["log10_conc_q99"].dims for d in sample_dims), ds["log10_conc_q99"].dims
    assert all(d in ds["log10_conc_q99_XYZ"].dims for d in (time_dim, *sample_dims)), ds["log10_conc_q99_XYZ"].dims

    # --- time (ky), robust for log-x (no zeros)
    t_raw = np.asarray(ds[time_dim].values, dtype=float) / 1000.0
    Nt = t_raw.size
    assert Nt > 0, "empty time axis"

    t_pos = t_raw[t_raw > 0]
    assert t_pos.size > 0, "time axis has no positive values; cannot use log x-scale"
    t_min_pos = float(t_pos.min())
    t = np.where(t_raw > 0, t_raw, t_min_pos * 0.999)

    # --- flatten samples consistently for both arrays
    da_max = ds["log10_conc_q99"].values.ravel()              # (Ns,)
    da_td  = ds["log10_conc_q99_XYZ"].values.reshape(-1, Nt)  # (Ns, Nt)
    Ns = da_max.shape[0]

    # --- extremes: apply Q on per-sample max-over-time
    finite_max = np.isfinite(da_max)
    vmax = da_max[finite_max]
    assert vmax.size > 0, "no finite values in log10_conc_q99"

    thr_low = float(np.quantile(vmax, Q))
    thr_high = float(np.quantile(vmax, 1.0 - Q))

    bottom_samples = np.flatnonzero(finite_max & (da_max <= thr_low))[:max_extreme_lines]
    top_samples    = np.flatnonzero(finite_max & (da_max >= thr_high))[:max_extreme_lines]

    extreme_mask = np.zeros(Ns, dtype=bool)
    extreme_mask[bottom_samples] = True
    extreme_mask[top_samples] = True
    rem_samples = np.flatnonzero(~extreme_mask)

    da_rem = da_td[rem_samples, :] if rem_samples.size else da_td[:0, :]

    # --- quantiles (computed on ALL samples)
    q_lo  = np.nanquantile(da_td, Q, axis=0)
    q_25  = np.nanquantile(da_td, 0.25, axis=0)
    q_50  = np.nanquantile(da_td, 0.50, axis=0)
    q_75  = np.nanquantile(da_td, 0.75, axis=0)
    q_hi  = np.nanquantile(da_td, 1.0 - Q, axis=0)

    # --- y-range for bins
    y_all = da_td.ravel()
    y_all = y_all[np.isfinite(y_all)]
    assert y_all.size > 0, "no finite data in log10_conc_q99_XYZ"
    y_min, y_max = np.percentile(y_all, [0.5, 99.5])
    if (not np.isfinite(y_min)) or (not np.isfinite(y_max)) or (y_min == y_max):
        y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))

    # --- figure (TOP and BOTTOM same size)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, sharey=False, height_ratios=[1, 1]
    )

    # =======================
    # Bottom: 2D histogram (remaining samples) + extremes
    # =======================

    # out-of-Q samples (extremes)
    for s in bottom_samples:
        ax_bot.plot(t, da_td[s, :], lw=0.7, alpha=0.7, color="red", zorder=2)
    for s in top_samples:
        ax_bot.plot(t, da_td[s, :], lw=0.7, alpha=0.7, color="blue", zorder=3)

    if rem_samples.size:
        vals = np.asarray(da_rem, dtype=float)  # (Nrem, Nt)
        x_flat = np.repeat(t, vals.shape[0])
        y_flat = vals.T.reshape(-1)

        m = np.isfinite(y_flat) & np.isfinite(x_flat)
        x_flat, y_flat = x_flat[m], y_flat[m]

        t_min = float(np.min(t[t > 0]))
        t_max = float(np.max(t))
        x_edges = np.geomspace(t_min, t_max, hist_bins[0] + 1)
        y_edges = np.linspace(y_min, y_max, hist_bins[1] + 1)

        H, xe, ye = np.histogram2d(x_flat, y_flat, bins=[x_edges, y_edges])

        vmin = 1 if np.any(H > 0) else None
        ax_bot.pcolormesh(
            xe, ye, H.T,
            shading="auto",
            norm=LogNorm(vmin=vmin) if vmin is not None else None,
            zorder=5
        )

    fig.suptitle("Log10(conc), 99% quantile over evaluation boundary planes")
    ax_bot.set_xscale("log")
    ax_bot.tick_params(axis="x", which="both", pad=12)
    ax_bot.set_xlabel("Time from 50y pulse (ky)")
    ax_bot.set_ylabel("Log10(conc)")
    # ax_bot.set_ylim(y_min, y_max)
    ax_bot.set_ylim(-10, -1)
    ax_bot.grid(alpha=0.25)

    # =======================
    # Top: inset hist slices + quantile lines
    # =======================
    ax_top.set_ylabel("Log10(conc)")
    ax_top.grid(alpha=0.2)

    # quantile lines in TOP
    ax_top.plot(t, q_lo, lw=1.3, alpha=0.8, label=f"q[{Q:g}]")
    ax_top.plot(t, q_25, lw=1.3, alpha=0.8, label="q[0.25]")
    ax_top.plot(t, q_50, lw=1.5, ls=":", label="q[0.5]")
    ax_top.plot(t, q_75, lw=1.3, alpha=0.8, label="q[0.75]")
    ax_top.plot(t, q_hi, lw=1.3, alpha=0.8, label=f"q[{1.0 - Q:g}]")
    ax_top.legend(loc="lower right", framealpha=0.5, facecolor="white")

    if rem_samples.size and Nt > 1 and n_slices > 0:
        slice_times = np.geomspace(float(np.min(t[t > 0])), float(np.max(t)), n_slices)
        idxs = np.unique(np.clip(np.searchsorted(t, slice_times), 0, Nt - 1))

        y_edges_v = np.linspace(y_min, y_max, slice_bins_y + 1)
        y_centers = 0.5 * (y_edges_v[:-1] + y_edges_v[1:])

        for i in idxs:
            x0 = float(t[i])

            y_slice = da_rem[:, i]
            y_slice = y_slice[np.isfinite(y_slice)]
            if y_slice.size == 0:
                continue

            counts, _ = np.histogram(y_slice, bins=y_edges_v)
            if counts.max() == 0:
                continue
            dens = counts / counts.max()

            ax_h = inset_axes(
                ax_top,
                width=slice_width_in,     # inches (constant on screen)
                height="100%",
                loc="lower left",
                bbox_to_anchor=(x0, y_min, 0.0, y_max - y_min),
                bbox_transform=ax_top.transData,
                borderpad=0.0,
            )

            ax_h.set_facecolor("none")
            for sp in ax_h.spines.values():
                sp.set_visible(False)
            ax_h.set_xticks([])
            ax_h.set_yticks([])
            ax_h.set_xlim(0.0, 1.05)
            ax_h.set_ylim(y_min, y_max)

            ax_h.fill_betweenx(y_centers, 0.0, dens, alpha=slice_alpha, linewidth=0.0)
            ax_h.plot(dens, y_centers, lw=0.8, alpha=0.9)
            ax_h.axvline(0.0, lw=0.6, alpha=0.25)

    fig.tight_layout()
    #plt.show()
    return fig

def plot_sobol_time_and_agg(
    sobol_time: xr.Dataset,   # select_sobol(...) for conc_q99_XYZ (time-dependent)
    sobol_agg: xr.Dataset,    # select_sobol(...) for conc_q99 (time-independent)
    var_name: str,
    *,
    figsize=(14, 5),
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
    x_label = "Simulation time (1000 y)"
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
    if 'aux' in sobol_time.dims:
        sobol_time = sobol_time.squeeze('aux', drop=True)  # remove 'aux' dim
    t = np.asarray(sobol_time["sim_time"].values) / 1000
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
    gs = fig.add_gridspec(1, 2, width_ratios=[left_frac, right_plot_frac], wspace=0.4)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # Left: stacked area over time
    axL.stackplot(t, SI_t, colors=colors, labels=list(labels_s), linewidth=0.5)
    #axL.set_ylim(0.0, 1.0)
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

    #axR.set_ylim(0.0, 1.0)
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


def plot_sobol_time_and_agg_boot(
    sobol_time: xr.Dataset,
    df_si: xr.Dataset,
    si_conc: xr.Dataset,
    var_name: str,
    *,
    figsize=(14, 6),
    si_ci_level: float = 0.90,
):
    # ---- squeeze aux if present
    sobol_time = sobol_time.squeeze("aux", drop=True) if "aux" in sobol_time.dims else sobol_time

    labels_s = df_si["label"].values[df_si["selected"].values]
    labels_g = si_conc["group"].values

    # ---- time-dependent SI in same order
    t = np.asarray(sobol_time["sim_time"].values) / 1000.0
    SI_t = np.asarray(sobol_time["SI"].sel(group=labels_s).values, dtype=float)  # (P, T)

    # ---- colors (others gray)
    cmap = plt.get_cmap("tab20")
    # colors = [(0.6, 0.6, 0.6, 1.0) if lab == "others" else cmap(i % 20)
    #           for i, lab in enumerate(labels_s)]
    colors = [cmap(i % 20) for i, lab in enumerate(labels_g)]

    # ---- layout: left time plot + right two axes (SI and ST)
    right_plot_frac = 0.4   # wider now because it's 2 axes
    legend_space_frac = 0.1
    left_frac = max(1e-3, 1.0 - right_plot_frac - legend_space_frac)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[left_frac, right_plot_frac], wspace=0.25)

    axL = fig.add_subplot(gs[0, 0])

    # right side split into two axes
    gsR = gs[0, 1].subgridspec(1, 2, width_ratios=[1., 1.], wspace=0.25)
    axSI = fig.add_subplot(gsR[0, 0])
    axST = fig.add_subplot(gsR[0, 1], sharey=axSI)

    # ---- Left: stacked area
    polys = axL.stackplot(t, SI_t, colors=colors, labels=list(labels_s), linewidth=0.5)

    for poly, lab, i in zip(polys, labels_s, range(len(labels_s))):
        def lighten(color, amount=0.65):
            """Blend color towards white by 'amount' (0..1)."""
            rgb = np.array(mcolors.to_rgb(color))
            return tuple((1 - amount) * rgb + amount * np.ones(3))

        base, label, pi, pj = df_si.iloc[i][['base', 'label', 'idx_0', 'idx_1']]

        if lab == "others":
            poly.set_facecolor((0.6, 0.6, 0.6, 1.0))
            poly.set_edgecolor((0.6, 0.6, 0.6, 1.0))
            continue

        if base == "S1":
            poly.set_facecolor(colors[pi])
            poly.set_edgecolor("none")  # cleaner for solid fills
            poly.set_hatch(None)

        elif base == "S2":
            # option A: light fill + striped hatch (usually best-looking)
            # poly.set_facecolor(lighten(colors[pi], 0.75))
            poly.set_facecolor(colors[pi])
            poly.set_edgecolor(colors[pj])  # hatch color comes from edgecolor
            poly.set_linewidth(0.0)  # avoids visible polygon outline
            poly.set_hatch("|")  # stripes
            # matplotlib.rcParams['hatch.linewidth'] = 6 # set the "width" at the top
            # if i%2 == 0:
            #     poly.set_hatch("\\")
            # else:
            #     poly.set_hatch("/")

            # if you want denser stripes: "////" or "xxx" etc.

    axL.set_xlim(t.min(), t.max())
    axL.set_xscale("log")
    axL.set_ylabel(f"Sobol index of {var_name}")
    axL.set_xlabel("Simulation time (1000 y)", labelpad=5)
    axL.grid(axis="y", alpha=0.2)

    # =========================
    # Right-A: SI "separate columns, connected corners"
    # =========================
    P = len(labels_s)
    x = np.arange(P)
    width = 0.9

    cum = 0.0
    # draw stacked-but-separated look:
    # each segment i is a bar at x=i with bottom=cum, height=SI_i
    # plus a thin "connector" line from previous top to next bottom (step)
    for i in range(P):
        # si = float(df_si["si"][i])
        base, label, pi, pj = df_si.iloc[i][['base', 'label', 'idx_0', 'idx_1']]
        indices = (pi,) if base == 'S1' else (pi, pj)
        si = (si_conc[base].values[indices])[0]
        si_err = (si_conc[f"{base}_boot_err"].values[indices])[0]
        assert si_err.shape == (2,), f"Unexpected shape for {base}_boot_err: {si_err.shape}"

        bottom_i = cum
        top_i = bottom_i + si

        # segment bar at its own column
        if base == "S1":
            axSI.bar(x[i], si, bottom=bottom_i, width=width, color=colors[pi], edgecolor="none")
        else:
            cont = axSI.bar(
                x[i], si, bottom=bottom_i, width=width,
                color=colors[pi],
                edgecolor=colors[pj],  # hatch color
                linewidth=0.0
            )
            cont.patches[0].set_hatch("/")  # or "////" etc.

        # connector: vertical at right edge of previous column to show the step,
        # and horizontal across to the next column (draw after i>0)
        if i > 0:
            # previous column right edge and current column left edge
            prev_right = x[i-1] + width/2
            curr_left  = x[i]   - width/2
            prev_top   = bottom_i  # because bottom_i == cumulative top after i-1

            # vertical rise on prev_right from prev_top - si_prev? no, we want step corner:
            # draw a horizontal from prev_right to curr_left at y=bottom_i
            axSI.plot([prev_right, curr_left], [bottom_i, bottom_i], color="k", lw=0.8, alpha=0.7)

        # CI error: lower-only from top_i down to low bound (or use full CI if you prefer)
        # low_i = float(df_si["si_err_lower"][i])
        # high_i = float(df_si["si_err_upper"][i])
        low_i = float(si_err[0])
        high_i = float(si_err[1])
        # if np.isfinite(low_i) and (0.0 < low_i < top_i):
        if np.isfinite(low_i) and (low_i < si < high_i):
            axSI.errorbar(
                x[i], top_i,
                # yerr=[[si - low_i], [0.0]],  # lower part only CI
                yerr=[[si - low_i], [high_i - si]],  # asymmetric CI around estimate
                fmt="none", ecolor="k", elinewidth=0.8, capsize=3, capthick=1.2
            )

        cum = top_i

    axSI.set_xticks(x)
    axSI.set_xticklabels([""] * P)
    axSI.set_xlim(-0.6, P - 0.4)
    axSI.set_xlabel("SI\n(stacked by rank)", labelpad=5)
    axSI.grid(axis="y", alpha=0.2)

    # =========================
    # Right-B: ST axis
    # =========================
    # Option 1: bars (simple + readable)
    for i in range(P):
        base, label, pi, pj = df_si.iloc[i][['base', 'label', 'idx_0', 'idx_1']]
        if base == 'S1':
            idx = (pi,)
            st = (si_conc['ST'].values[idx, 0])[0]
            st_err = (si_conc[f"ST_boot_err"].values[idx])[0]
            low_i = float(st_err[0])
            high_i = float(st_err[1])
            if np.isfinite(st):
                axST.bar(pi, st, width=0.9, color=colors[pi], edgecolor="none")
            if np.isfinite(low_i) and (low_i < st < high_i):
                axST.errorbar(
                    pi, st,
                    # yerr=[[si - low_i], [0.0]],  # lower part only CI
                    yerr=[[st - low_i], [high_i - st]],  # asymmetric CI around estimate
                    fmt="none", ecolor="k", elinewidth=0.8, capsize=3, capthick=1.2
                )

    axST.set_xticks(x)
    axST.set_xticklabels([""] * len(labels_g))
    axST.set_xlim(-0.6, len(labels_g) - 0.4)
    axST.set_xlabel("ST", labelpad=5)
    axST.grid(axis="y", alpha=0.2)

    # Hide repeated y tick labels on ST panel
    plt.setp(axST.get_yticklabels(), visible=False)
    axST.tick_params(axis="y", length=0)

    # ---- legend outside (as you had)
    ci_proxy = Line2D([], [], color="k", linestyle="-", label=f"CI ({int(round(si_ci_level*100))}%)")
    proxies = [Patch(facecolor=colors[i], edgecolor="none", label=group) for i, group in enumerate(labels_g)]
    proxies.append(ci_proxy)
    labels_legend = list(labels_g) + [ci_proxy.get_label()]
    fig.legend(
        handles=proxies, labels=labels_legend,
        loc="center left", bbox_to_anchor=(0.9, 0.5),
        ncol=1, frameon=False, title="Parameter group"
    )

    fig.tight_layout()
    return fig, (axL, axSI, axST)


def plot_sobol_time_and_agg_split(
    sobol_time: xr.Dataset,
    sobol_agg: xr.Dataset,
    var_name: str,
    *,
    figsize=(14, 6),
    si_ci_level: float = 0.90,
):
    x_label = "Simulation time (1000 y)"

    # ---- squeeze aux if present
    sobol_agg = sobol_agg.squeeze("aux", drop=True) if "aux" in sobol_agg.dims else sobol_agg
    sobol_time = sobol_time.squeeze("aux", drop=True) if "aux" in sobol_time.dims else sobol_time

    # ---- sort by aggregated SI desc
    labels = sobol_agg["group"].values.astype(str)
    SI_agg = np.asarray(sobol_agg["SI"].values, dtype=float)  # (P,)
    order = np.argsort(-SI_agg)

    labels_s = labels[order]
    SI_agg_s = SI_agg[order]
    SI_ci_s = np.asarray(sobol_agg["SI_ci"].values, dtype=float)[order, :]  # (P, 2) low/high

    ST_agg_src = sobol_agg.get("ST")
    ST_agg_s = (
        np.asarray(ST_agg_src.values, dtype=float)[order]
        if isinstance(ST_agg_src, xr.DataArray)
        else np.full_like(SI_agg_s, np.nan)
    )

    # ---- time-dependent SI in same order
    t = np.asarray(sobol_time["sim_time"].values) / 1000.0
    SI_t = np.asarray(sobol_time["SI"].sel(group=labels_s).values, dtype=float)  # (P, T)

    # ---- colors (others gray)
    cmap = plt.get_cmap("tab20")
    colors = [(0.6, 0.6, 0.6, 1.0) if lab == "others" else cmap(i % 20)
              for i, lab in enumerate(labels_s)]

    # ---- layout: left time plot + right two axes (SI and ST)
    right_plot_frac = 0.26   # wider now because it's 2 axes
    legend_space_frac = 0.16
    left_frac = max(1e-3, 1.0 - right_plot_frac - legend_space_frac)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[left_frac, right_plot_frac], wspace=0.4)

    axL = fig.add_subplot(gs[0, 0])

    # right side split into two axes
    gsR = gs[0, 1].subgridspec(1, 2, width_ratios=[1., 1.], wspace=0.25)
    axSI = fig.add_subplot(gsR[0, 0])
    axST = fig.add_subplot(gsR[0, 1], sharey=axSI)

    # ---- Left: stacked area
    axL.stackplot(t, SI_t, colors=colors, labels=list(labels_s), linewidth=0.5)
    axL.set_xlim(t.min(), t.max())
    axL.set_xscale("log")
    axL.set_ylabel(f"Sobol index of {var_name}")
    axL.set_xlabel(x_label, labelpad=5)
    axL.grid(axis="y", alpha=0.2)

    # =========================
    # Right-A: SI "separate columns, connected corners"
    # =========================
    P = len(labels_s)
    x = np.arange(P)
    width = 0.9

    cum = 0.0
    # draw stacked-but-separated look:
    # each segment i is a bar at x=i with bottom=cum, height=SI_i
    # plus a thin "connector" line from previous top to next bottom (step)
    for i in range(P):
        si = float(SI_agg_s[i])
        bottom_i = cum
        top_i = bottom_i + si

        # segment bar at its own column
        axSI.bar(x[i], si, bottom=bottom_i, width=width, color=colors[i], edgecolor="none")

        # connector: vertical at right edge of previous column to show the step,
        # and horizontal across to the next column (draw after i>0)
        if i > 0:
            # previous column right edge and current column left edge
            prev_right = x[i-1] + width/2
            curr_left  = x[i]   - width/2
            prev_top   = bottom_i  # because bottom_i == cumulative top after i-1

            # vertical rise on prev_right from prev_top - si_prev? no, we want step corner:
            # draw a horizontal from prev_right to curr_left at y=bottom_i
            axSI.plot([prev_right, curr_left], [bottom_i, bottom_i], color="k", lw=0.8, alpha=0.7)

        # CI error: lower-only from top_i down to low bound (or use full CI if you prefer)
        low_i = float(SI_ci_s[i, 0])
        high_i = float(SI_ci_s[i, 1])
        # if np.isfinite(low_i) and (0.0 < low_i < top_i):
        if np.isfinite(low_i) and (low_i < si < high_i):
            axSI.errorbar(
                x[i], top_i,
                yerr=[[si - low_i], [0.0]],  # lower part only CI
                # yerr=[[si - low_i], [high_i - si]],  # asymmetric CI around estimate
                fmt="none", ecolor="k", elinewidth=0.8, capsize=3, capthick=1.2
            )

        cum = top_i

    axSI.set_xticks(x)
    axSI.set_xticklabels([""] * P)
    axSI.set_xlim(-0.6, P - 0.4)
    axSI.set_xlabel("SI\n(stacked by rank)", labelpad=5)
    axSI.grid(axis="y", alpha=0.2)

    # =========================
    # Right-B: ST axis
    # =========================
    # Option 1: bars (simple + readable)
    for i in range(P):
        st = float(ST_agg_s[i]) if np.isfinite(ST_agg_s[i]) else np.nan
        if np.isfinite(st):
            axST.bar(x[i], st, width=0.9, color=colors[i], edgecolor="none")

    axST.set_xticks(x)
    axST.set_xticklabels([""] * P)
    axST.set_xlim(-0.6, P - 0.4)
    axST.set_xlabel("ST", labelpad=5)
    axST.grid(axis="y", alpha=0.2)

    # Hide repeated y tick labels on ST panel
    plt.setp(axST.get_yticklabels(), visible=False)
    axST.tick_params(axis="y", length=0)

    # ---- legend outside (as you had)
    ci_proxy = Line2D([], [], color="k", linestyle="-", label=f"CI ({int(round(si_ci_level*100))}%)")
    proxies = [Patch(facecolor=colors[i], edgecolor="none", label=str(labels_s[i])) for i in range(P)]
    proxies.append(ci_proxy)
    labels_legend = list(labels_s) + [ci_proxy.get_label()]
    fig.legend(
        handles=proxies, labels=labels_legend,
        loc="center left", bbox_to_anchor=(0.98, 0.5),
        ncol=1, frameon=False, title="Parameter / Interaction"
    )

    fig.tight_layout()
    return fig, (axL, axSI, axST)

def save_conc_and_si_pdf(
    ds_stat: xr.Dataset,
    sobol_time: xr.Dataset,
    sobol_agg: xr.Dataset,
    df_si: xr.Dataset,
    si_conc: xr.Dataset,
    var_name: str,
    title: str,
    *,
    figsize=(11, 5),
    si_ci_level: float = 0.90,
    si_table = None,
    out_pdf_path: str | Path = "conc_and_si.pdf",
):
    """
    Convenience: render the two figures and save each on its own page of a single PDF.
    """
    # split the figures for publication
    subdir = out_pdf_path.parents[0] / "pub" / out_pdf_path.stem
    subdir.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf_path) as pdf:
        fig1 = plot_conc_timeseries_distribution1(ds_stat)

        save_close_fig(fig1, pdf, subdir / "hist_conc_over_time.pdf")

        if sobol_time is not None and sobol_agg is not None:
            # fig2, _ = plot_sobol_time_and_agg_split(sobol_time, sobol_agg, var_name,
            #                                 figsize=figsize, si_ci_level=si_ci_level)
            fig2, _ = plot_sobol_time_and_agg_boot(sobol_time, df_si, si_conc, var_name,
                                                   #figsize=figsize,
                                                   si_ci_level=si_ci_level)

            save_close_fig(fig2, pdf, subdir / "sobol_agg_over_time.pdf")

        if si_table is not None:
            fig3, ax = plt.subplots(figsize=figsize)
            ax.axis("off")

            # Title
            ax.text(
                0.5, 0.95,
                f"Sobol indices – {var_name}\n{title}",
                ha="center", va="top",
                fontsize=14, fontweight="bold",
            )

            table_text = "\n".join(si_table)
            # Table (monospace so columns stay aligned)
            ax.text(
                0.5, 0.80,
                table_text,
                ha="center", va="top",
                family="monospace",
                fontsize=10,
            )

            save_close_fig(fig3, pdf, subdir / "sobol_agg_table.pdf")







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

    subdir = output_pdf.parents[0] / "pub" / output_pdf.stem
    subdir.mkdir(parents=True, exist_ok=True)

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

            save_close_fig(fig, pdf, subdir / f"conc_ecdf_{ti}.pdf")

    print("Tail ECDF plot DONE:", output_pdf)


def plot_qmc_iid_mask_heatmap(mask_rc: xr.DataArray,
                              mask_big: xr.DataArray,
                              mask_small: xr.DataArray,
                              n_computed_samples: int,
                              title="QMC × IID mask heatmap (exclusive: rc → big → small)",
                              figsize=(12, 6),
                              output_dir=None) -> Tuple[plt.Figure, plt.Axes, xr.DataArray]:
    """
    Exclusive categorical heatmap over (IID,QMC):
      0=OK, 1=return_code, 2=too_big, 3=too_small
    Precedence: return_code -> big -> small.
    """
    assert mask_rc.dims == ("IID", "QMC")
    assert mask_big.dims == ("IID", "QMC")
    assert mask_small.dims == ("IID", "QMC")

    mask_rc, mask_big, mask_small = xr.align(mask_rc, mask_big, mask_small, join="exact")

    mask_rc = mask_rc.compute()
    mask_big = mask_big.compute()
    mask_small = mask_small.compute()

    cat = xr.zeros_like(mask_rc, dtype=np.uint8)

    # numpy-style boolean assignment (masks already exclusive)
    c = cat.data
    c[mask_rc.data] = 1
    c[mask_big.data] = 2
    c[mask_small.data] = 3
    arr = cat.values  # shape (IID,QMC), values in {0,1,2,3}

    # zeros_per_qmc = (cat == 0).sum("IID")
    # zeros_per_iid = (cat == 0).sum("QMC")
    # n_all_zero_iid = (zeros_per_iid == cat.sizes["QMC"]).sum().item()
    all_zero_iid = (cat == 0).all("QMC")  # dims: (IID,) boolean
    n_all_zero_iid = all_zero_iid.sum().item()  # Python int
    print(f"N valid isaltelli samples: {n_all_zero_iid} of {n_computed_samples}")

    colors = ["white", "#d62728", "#00ff0e", "#1f77b4"]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr.T, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3)

    ax.set_title(f"{title}\nN valid Saltelli samples #{n_all_zero_iid}/{n_computed_samples}")
    ax.set_xlabel("QMC")
    ax.set_ylabel("IID")

    import matplotlib.patches as mpatches
    labels = [
        f"OK (#{np.sum(arr==0)})",
        f"return_code<0 (#{np.sum(arr==1)})",
        f"conc_max>hi (#{np.sum(arr==2)})",
        f"conc_min<lo (#{np.sum(arr==3)})",
    ]
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=c, label=l, edgecolor="black")
            for c, l in
            zip(colors, labels)
        ],
        loc="upper right",
#        frameon=True,
    )

    fig.tight_layout()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "qmc_iid_mask_heatmap.pdf")
    else:
        plt.show()

    return fig, ax, cat