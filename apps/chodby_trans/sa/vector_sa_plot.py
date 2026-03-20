import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

def select_sobol_terms_with_others(
    ds: xr.Dataset,
    var_threshold: float,
    output_dim: str = "output",
    si_threshold: float = 0.0,

) -> xr.Dataset:
    """
    Vectorized selection of dominant S1 and S2 terms across outputs.
    Adds an 'others' row so that each output column sums to 1.0.

    Inputs
    ------
    ds  : Dataset with coords group, group2, output, bound and vars:
          S1(group, output), S2(group, group2, output),
          S1_agg(group), S1_agg_ci(group, bound)
          (S2 is always present; identity on diag if not computed.)
    var_threshold : cumulative sum threshold applied per-output
    si_threshold  : minimum SI to consider before cum-sum

    Returns
    -------
    Dataset with coords:
      param (labels like 'g' or 'g×h', plus 'others'),
      output, bound
    and vars:
      SI(param, output), SI_agg(param), SI_agg_ci(param, bound)
      (S2 and 'others' get CI=[0,0])

    TODO:
    0. Assume ds with: S1, S2, ST, S1_agg, S2_agg, ST_agg, S1_agg_ci, ST_agg_ci
    1. merge S1, S2; S1_agg, S2_agg; extend CI and ST and ST_agg -> new DS
    2. ranking by SI_agg -> select mask
    3. assemble selected + 'others' -> new DS
    """
    groups = ds["group"].values
    outputs = ds[output_dim].values
    bound = ds.get("bound", xr.DataArray(["low", "high"], dims=("bound",))).values

    # Merge S1 and S2 (upper triangle i<j)
    S1 = ds["S1"].values                # (G, M)
    G, M = S1.shape
    I, J = np.triu_indices(G, k=1)
    S2_pairs = ds["S2"].values[I, J, :]              # (P, M) with P = G*(G-1)/2
    SI_all = np.vstack([S1, S2_pairs])  # (T, M) T=G+P
    g_labels = groups.astype(str)
    labels_all = [("S1", g, (i,)) for i, g in enumerate(g_labels)]
    labels_all.extend(
         [("S2", f"{g_labels[i]}×{g_labels[j]}", (i, j)) for i, j in zip(I, J)]
    )
    
    # Aggregates
    #S1_agg =                     # (G,)
    S1_agg_ci = ds["S1_agg_ci"].values              # (G,2)
    S2_agg_pairs = S2.mean(axis=2)[I, J]            # (P,)
    zeros_ci_pairs = np.zeros((len(I), 2), dtype=float)

    SI_agg_all = np.concatenate([
        ds["S1_agg"].values, S2_agg_pairs])          # (T,)
    SI_agg_ci_all = np.vstack([S1_agg_ci, zeros_ci_pairs])       # (T,2)
    ST_agg_all = np.concatenate([
         ds["ST_agg"].values,  zeros_ci_pairs[:, 0]])

    # Per-output ranking → cum-sum → cut
    order = np.argsort(-SI_all, axis=0)                           # (T,M) descending
    sorted_vals = np.take_along_axis(SI_all, order, axis=0)       # (T,M)
    valid_mask = (sorted_vals >= si_threshold)                    # (T,M)
    
    cs = (sorted_vals * valid_mask).cumsum(axis=0)                # (T,M)

    #reached =                                # (T,M)
    any_reached = (cs >= var_threshold).any(axis=0)                             # (M,)
    first_pos = np.argmax(reached, axis=0)                        # (M,)
    valid_counts = valid_mask.sum(axis=0)                         # (M,)
    cut = np.where(any_reached, first_pos + 1, valid_counts)      # (M,)

    ranks = np.arange(SI_all.shape[0])[:, None]
    take_sorted = (ranks < cut[None, :]) & valid_mask             # (T,M)
    rows, cols = np.where(take_sorted)
    selected_terms = np.unique(order[rows, cols])                 # (K,)


 
    # Assemble selected
    SI_sel = SI_all[selected_terms, :]            # (K,M)
    SI_agg_sel = SI_agg_all[selected_terms]       # (K,)
    SI_agg_ci_sel = SI_agg_ci_all[selected_terms, :]  # (K,2)
    params_sel = labels_all[selected_terms]       # (K,)

    # 'others' so each column sums to 1.0
    # This equals (sum of all dropped S1/S2) + (residual higher-order), clipped at 0
    others = np.clip(1.0 - SI_sel.sum(axis=0), 0.0, 1.0)          # (M,)
    # Aggregated 'others' as 1 - sum of selected aggregates, clipped
    others_agg = float(np.clip(1.0 - SI_agg_sel.sum(), 0.0, 1.0))
    others_ci = np.array([0.0, 0.0], dtype=float)

    # Append 'others'
    SI = np.vstack([SI_sel, others[None, :]])                     # (K+1, M)
    SI_agg = np.concatenate([SI_agg_sel, [others_agg]])           # (K+1,)
    SI_agg_ci = np.vstack([SI_agg_ci_sel, others_ci[None, :]])    # (K+1, 2)
    ST_agg = ST_agg_all[selected_terms]
    params = np.concatenate([params_sel, np.array(["others"], dtype=object)])

    return xr.Dataset(
        data_vars=dict(
            SI=(("param", output_dim), SI),
            SI_agg=(("param",), SI_agg),
            SI_agg_ci=(("param", "bound"), SI_agg_ci),
            ST_agg=(("param"), ST_agg)
        ),
        coords=dict(
            param=np.array(params, dtype=object),
            output=outputs,
            bound=bound,
        ),
    )



def _distinct_colors(n: int, labels) -> list:
    """More distinct colors than a single tab10/tab20. 'others' gets gray."""
    pools = []
    for name in ("tab20", "tab20b", "tab20c"):
        cmap = mpl.cm.get_cmap(name)
        if hasattr(cmap, "colors"):
            pools.extend(list(cmap.colors))
        else:
            pools.extend([cmap(i / 20.0) for i in range(20)])
    if n <= len(pools):
        cols = pools[:n]
    else:
        # fallback: evenly spaced in HSV
        cols = [mpl.colors.hsv_to_rgb((i / n, 0.65, 0.9)) for i in range(n)]
    # force 'others' to gray if present
    for i, lab in enumerate(labels):
        if str(lab) == "others":
            cols[i] = (0.7, 0.7, 0.7, 1.0)
    return cols


def plot_sobol_stacked(
    ds_sel: xr.Dataset,
    *,
    x_label: str,
    figsize=(12, 5),
    ci_level: float = 0.95,
    out_path: str | None = None,
):
    """
    Stacked-area SIs by output (left) and stacked aggregated SIs (right).
    - params sorted by SI_agg desc (largest at bottom)
    - legend to the right (single column); legend_x and legend_space_frac control placement
    - right plot shows half error bars (lower only) with legend entry 'CI(<level>)'
    - right plot: for S1 terms (labels without '×' and not 'others'), draw ▲ for S1_agg and ★ for
      ST_agg at the bottom of the segment; x scaled so 1.0 hits the right edge of the bar.

    st_agg_by_group: mapping {group_name -> ST_agg value in [0,1]} or DataArray with coord 'group'
    """
    right_plot_frac: float = 0.12,     # ~ 1/10 figure width for right plot
    legend_space_frac: float = 0.12,   # ~ 1/10 figure width reserved for legend
    legend_x: float = 0.92,            # legend anchor x in figure coords


    params = ds_sel["param"].values.astype(object)
    outputs = ds_sel["output"].values
    SI = ds_sel["SI"].values                  # (P, M)
    SI_agg = ds_sel["SI_agg"].values          # (P,)
    SI_agg_ci = ds_sel["SI_agg_ci"].values    # (P, 2) [low, high]
    ST_agg = ds_sel["ST_agg"].values

    P, M = SI.shape

    # sort by aggregate descending
    order = np.argsort(-SI_agg)
    params_s = params[order]
    SI_s = SI[order, :]
    SI_agg_s = SI_agg[order]
    SI_agg_ci_s = SI_agg_ci[order, :]

    # colors (others -> gray)
    colors = _distinct_colors(P, params_s)

    # ---- layout: left plot, right thin plot; leave space on right for legend
    left_frac = max(1e-3, 1.0 - right_plot_frac - legend_space_frac)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[left_frac, right_plot_frac],
        wspace=0.25
    )
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)

    # ---------- LEFT: stacked area (supports non-uniform outputs)
    # stackplot expects list/rows per layer, bottom-most first
    polys = axL.stackplot(outputs, SI_s, colors=colors, labels=[str(p) for p in params_s], linewidth=0.5)
    axL.set_ylim(0.0, 1.0)
    axL.set_ylabel("Sobol index")
    axL.set_xlabel(x_label)
    axL.grid(axis="y", alpha=0.2)

    # ---------- RIGHT: stacked aggregated bar at x=0
    x0 = 0.0
    width = 0.6  # visual width only
    bottom = 0.0

    # proxy for CI legend entry
    ci_proxy = Line2D([], [], color="k", linestyle="-", label=f"CI ({int(round(ci_level*100))}%)")

    for i in range(P):
        # stacked segment
        axR.bar(
            [x0], [SI_agg_s[i]],
            bottom=[bottom],
            width=width,
            color=colors[i],
        )

        # half error bar (lower only), drawn as a vertical errorbar ending at the top
        low_i = float(SI_agg_ci_s[i, 0])
        top_i = bottom + SI_agg_s[i]
        # if low_i is valid (< segment height), draw from top down to top - (top_seg - low_abs)
        if low_i > 0.0 and low_i < SI_agg_s[i]:
            lo = top_i - low_i     # amount below the top
            axR.errorbar(
                [x0], [top_i],
                yerr=[[lo], [0.0]],  # lower error only
                fmt="none",
                ecolor="k",
                elinewidth=1.2,
                capsize=3,
                capthick=1.2,
            )

        # S1/ST markers for S1 terms only (labels without '×' and not 'others')
        if ST_agg[i] > 0.0:
            # Have ST information (i.e. S1 index)
            #label_i = str(params_s[i])
            # Triangle (▲) for S1_agg of this term
            x_tri = x0 - width/2.0 + SI_agg_s[i] * width
            x_star = x0 - width/2.0 + ST_agg[i] * width            
            axR.plot([x_tri], [bottom], marker="^", markersize=6, color="k", linestyle="none")
            axR.plot([x_star], [bottom], marker="*", markersize=7, color="k", linestyle="none")

        bottom += SI_agg_s[i]

    axR.set_ylim(0.0, 1.0)
    axR.set_xlim(x0 - 1.0, x0 + 1.0)
    axR.set_xticks([x0])
    axR.set_xticklabels(["aggregated"])
    axR.set_xlabel("")

    axR.grid(axis="y", alpha=0.2)

    # ---------- Legend: to the right, single column ----------
    legend_labels = [str(p) for p in params_s]
    proxies = [mpl.patches.Patch(facecolor=colors[i], edgecolor="none", label=legend_labels[i]) for i in range(P)]
    # add CI proxy as well
    proxies.append(ci_proxy)
    labels = legend_labels + [ci_proxy.get_label()]

    fig.legend(
        handles=proxies,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(legend_x, 0.5),
        ncol=1,
        frameon=False,
        title="Parameter / Interaction",
    )

    # leave horizontal space on right for legend
    fig.tight_layout(rect=(0, 0, 1.0 - legend_space_frac, 1))

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig, (axL, axR)