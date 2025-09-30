from typing import List, Tuple, Union
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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
