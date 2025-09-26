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
    """
    groups = ds["group"].values
    outputs = ds[output_dim].values
    bound = ds.get("bound", xr.DataArray(["low", "high"], dims=("bound",))).values

    # Merge S1 and S2 (upper triangle i<j)
    S1 = ds["S1"].values                # (G, M)
    S2 = ds["S2"].values                # (G, G, M)
    G, M = S1.shape
    I, J = np.triu_indices(G, k=1)
    S2_pairs = S2[I, J, :]              # (P, M) with P = G*(G-1)/2

    SI_all = np.vstack([S1, S2_pairs])  # (T, M) T=G+P

    # Labels
    gstr = groups.astype(str)
    s2_labels = np.char.add(np.char.add(gstr[I], "×"), gstr[J])
    labels_all = np.concatenate([gstr, s2_labels])  # (T,)

    # Aggregates
    S1_agg = ds["S1_agg"].values                    # (G,)
    S1_agg_ci = ds["S1_agg_ci"].values              # (G,2)
    S2_agg_pairs = S2.mean(axis=2)[I, J]            # (P,)
    zeros_ci_pairs = np.zeros((len(I), 2), dtype=float)

    SI_agg_all = np.concatenate([S1_agg, S2_agg_pairs])          # (T,)
    SI_agg_ci_all = np.vstack([S1_agg_ci, zeros_ci_pairs])       # (T,2)

    # Per-output ranking → cum-sum → cut
    order = np.argsort(-SI_all, axis=0)                           # (T,M) descending
    sorted_vals = np.take_along_axis(SI_all, order, axis=0)       # (T,M)
    valid_mask = (sorted_vals >= si_threshold)                    # (T,M)
    cs = (sorted_vals * valid_mask).cumsum(axis=0)                # (T,M)

    reached = (cs >= var_threshold)                               # (T,M)
    any_reached = reached.any(axis=0)                             # (M,)
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
    params = np.concatenate([params_sel, np.array(["others"], dtype=object)])

    return xr.Dataset(
        data_vars=dict(
            SI=(("param", output_dim), SI),
            SI_agg=(("param",), SI_agg),
            SI_agg_ci=(("param", "bound"), SI_agg_ci),
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


def plot_sobol_stacked(ds_sel: xr.Dataset, *, figsize=(12, 5), 
                       x_label, out_path=None):
    """
    Stacked SIs by output (left) and stacked aggregated SIs (right).
    - params sorted by SI_agg desc (largest at bottom)
    - legend to the right, single column, shared for both panels
    - 'others' colored gray
    - CI on the right: mark lower bound with a horizontal tick per stacked segment
    """
    params = ds_sel["param"].values.astype(object)
    outputs = ds_sel["output"].values
    bound = ds_sel["bound"].values
    SI = ds_sel["SI"].values           # (P, M)
    SI_agg = ds_sel["SI_agg"].values   # (P,)
    SI_agg_ci = ds_sel["SI_agg_ci"].values  # (P, 2) [low, high]

    P, M = SI.shape

    # sort by aggregate descending
    order = np.argsort(-SI_agg)
    params_s = params[order]
    SI_s = SI[order, :]
    SI_agg_s = SI_agg[order]
    SI_agg_ci_s = SI_agg_ci[order, :]

    # colors
    colors = _distinct_colors(P, params_s)

    # figure with shared y; leave room on right for legend
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=figsize, sharey=True, #gridspec_kw={"wspace": 0.25}
    )

    # ---------- LEFT: stacked per-output ----------
    bottoms = np.zeros(M, dtype=float)
    handles = []
    for i in range(P):
        h = axL.bar(
            outputs,
            SI_s[i, :],
            bottom=bottoms,
            color=colors[i],
            label=str(params_s[i]),
        )
        handles.append(h)
        bottoms += SI_s[i, :]

    axL.set_ylim(0.0, 1.0)
    axL.set_ylabel("Most important S1 and S2 indices")
    axL.set_xlabel(x_label)
    
    # ---------- RIGHT: stacked aggregated (one bar at x=0) ----------
    xrng = np.array([0.0])
    width = 0.1
    bottom = 0.0
    for i in range(P):
        axR.bar(
            xrng,
            [SI_agg_s[i]],
            bottom=[bottom],
            width=width,
            color=colors[i],
            #edgecolor="white",
            #linewidth=0.5,
        )
        # lower-only CI tick at cumulative lower bound
        low_i = float(SI_agg_ci_s[i, 0])  # lower bound for this term
        y_tick = bottom + low_i
        axR.hlines(
            y_tick,
            xrng[0] - width * 0.45,
            xrng[0] + width * 0.45,
            colors="k",
            #linewidth=1.2,
        )
        bottom += SI_agg_s[i]

    axR.set_xlabel("var weighted mean SI")

    # ---------- Legend to the right, single column ----------
    # Build legend entries (one per param, top-to-bottom matches stack order)
    #legend_labels = [str(p) for p in params_s]
    # Use one handle per layer (from any bar call); create proxy artists of correct colors
    #proxies = [mpl.patches.Patch(facecolor=colors[i], edgecolor="white", label=legend_labels[i]) for i in range(P)]
    fig.legend(
        #handles=proxies,
        #labels=legend_labels,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        ncol=1,
        frameon=False,
        title="Parameter,\nInteraction",
    )

    # tighten layout but keep right margin for legend
    fig.tight_layout(rect=(0, 0, 0.85, 1))  # leave 15% for legend
    if out_path:
        fig.savefig(out_path)
    return fig, (axL, axR)