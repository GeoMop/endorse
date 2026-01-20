from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from chodby_trans import job

plt.rcParams.update({
    "font.size": 18,        # base size
    # "axes.titlesize": 16,
    # "axes.labelsize": 14,
    # "xtick.labelsize": 12,
    # "ytick.labelsize": 12,
    # "legend.fontsize": 12,
    # "figure.titlesize": 18, # suptitle
})

def load_population(filename):
    file = job.input.dir_path / filename
    df = pd.read_csv(file)
    # points_vec = df.to_numpy().reshape(-1, 1)
    return df.to_numpy()



import numpy as np
import matplotlib.pyplot as plt

def plot_bivariate_samples(
    v1,
    v2,
    *,
    weights=None,
    ax=None,
    kind="scatter",            # "scatter" | "hexbin"
    s=6,
    alpha=0.25,
    bins=60,                   # for contours
    gridsize=60,               # for hexbin
    cmap="viridis",
    color=None,                # e.g. "k" for scatter
    xlabel="v1",
    ylabel="v2",
    title=None,                # axes title (optional)
    suptitle=None,             # figure title (optional)
    subtitle=None,             # figure subtitle (optional)
    subtitle_y=0.94,           # vertical position for subtitle in figure coords
    suptitle_y=0.985,          # vertical position for suptitle in figure coords
    tight_layout=True,
    tight_layout_rect=(0, 0, 1, 0.92),  # leave room for suptitle/subtitle
    equal_aspect=False,
    lims=None,                 # (xmin, xmax, ymin, ymax) or None
    show_mean=True,
    mean_kwargs=None,
    contour_levels=(0.5, 0.9, 0.95),
    draw_contours=False,
    contour_kwargs=None,
):
    """
    Plot bivariate Bayesian inversion samples (v1, v2) with optional figure title + subtitle.

    LaTeX-friendly labels: pass strings like r"$v_1$" etc.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    v1 = np.asarray(v1).ravel()
    v2 = np.asarray(v2).ravel()
    if v1.size != v2.size:
        raise ValueError("v1 and v2 must have the same length.")

    # Handle weights
    w = None
    if weights is not None:
        w = np.asarray(weights).ravel()
        if w.size != v1.size:
            raise ValueError("weights must have the same length as v1/v2.")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative.")
        w_sum = w.sum()
        if w_sum <= 0:
            raise ValueError("weights sum must be > 0.")
        w = w / w_sum

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Limits
    if lims is not None:
        xmin, xmax, ymin, ymax = lims
    else:
        xmin, xmax = np.nanmin(v1), np.nanmax(v1)
        ymin, ymax = np.nanmin(v2), np.nanmax(v2)
    print(f"lims: {xmin, xmax, ymin, ymax}")

    # Main plot
    mappable = None
    if kind == "scatter":
        ax.scatter(
            v1, v2,
            s=s,
            alpha=alpha,
            c=(color if color is not None else None),
            cmap=cmap if color is None else None,
            linewidths=0,
        )
    elif kind == "hexbin":
        if w is None:
            mappable = ax.hexbin(v1, v2, gridsize=gridsize, cmap=cmap, mincnt=1)
        else:
            mappable = ax.hexbin(
                v1, v2,
                C=w,
                reduce_C_function=np.sum,
                gridsize=gridsize,
                cmap=cmap,
                mincnt=1,
            )
        fig.colorbar(mappable, ax=ax, label=("count" if w is None else "weight sum"))
    else:
        raise ValueError('kind must be "scatter" or "hexbin".')

    # Mean marker
    if show_mean:
        mean_kwargs = mean_kwargs or {}
        if w is None:
            mx, my = np.mean(v1), np.mean(v2)
        else:
            mx, my = np.sum(w * v1), np.sum(w * v2)
        ax.plot(mx, my, marker="o", markersize=4, mew=2, **mean_kwargs)

    # Optional approximate credible contours from a 2D histogram
    if draw_contours:
        contour_kwargs = contour_kwargs or {}
        H, xedges, yedges = np.histogram2d(
            v1, v2,
            bins=bins,
            range=[[xmin, xmax], [ymin, ymax]],
            weights=(w if w is not None else None),
            density=False,
        )
        H = H.astype(float)
        total = H.sum()
        if total > 0:
            P = H / total
            flat = P.ravel()
            order = np.argsort(flat)[::-1]
            cdf = np.cumsum(flat[order])

            thresholds = []
            for L in contour_levels:
                idx = np.searchsorted(cdf, L, side="left")
                idx = min(idx, len(order) - 1)
                thresholds.append(flat[order][idx])

            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xc, yc, indexing="xy")
            Z = P.T

            levels = sorted(thresholds)
            cs = ax.contour(X, Y, Z, levels=levels, **contour_kwargs)

            fmt = {}
            sorted_idx = np.argsort(thresholds)  # aligns with "levels"
            for lev, j in zip(cs.levels, sorted_idx):
                fmt[lev] = f"{contour_levels[j]:.0%}"
            ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=8)

    # Labels / titles
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # Figure title + subtitle
    if suptitle:
        fig.suptitle(suptitle, y=suptitle_y)
    if subtitle:
        fig.text(0.5, subtitle_y, subtitle, ha="center", va="top")

    # Layout (leave space for suptitle/subtitle)
    if tight_layout:
        if suptitle or subtitle:
            fig.tight_layout(rect=list(tight_layout_rect))
        else:
            fig.tight_layout()

    return ax




def main():
    for i in range(5):
        arr = load_population(f"fr_Bukov_bayes/P30_alpha_minus_pop{i+1}.csv")

        fig, ax = plt.subplots()
        # assuming you already have plot_bivariate_samples() from above
        plot_bivariate_samples(
            arr[:,0], arr[:,1],
            ax=ax,
            kind="scatter",
            s=14,  # point size
            alpha=0.15,  # transparency
            color="blue",
            show_mean=True,
            draw_contours=True,  # nice for Bayesian posteriors
            contour_levels=(0.5, 0.95),
            contour_kwargs={
                "colors": ["darkmagenta", "darkorange"],  # or ["k","k","k"] or any Matplotlib color
                "linewidths": 2.0,  # thicker lines
                "linestyles": "-",  # "--", ":", "-.", etc.
                "alpha": 0.9,
            },
            xlabel=r"$p_{30}$",
            ylabel=r"$\alpha$",
            # title=f"Fracture size posterior samples (10k) -- Population {i+1}",
            title=f"Population {i + 1}",
            # suptitle=f"Population {i+1}",
            # subtitle=r"N=10k draws • mean marked with circle • 50/95% contours",
            subtitle_y=0.88,
            suptitle_y=0.93,
            mean_kwargs={"color":"red"}
        )
        # plt.show()
        # exit(0)
        fig.savefig(job.output.dir_path / f"fr_pop_{i+1}.pdf")

if __name__ == '__main__':
    work_dir = Path("fr_Bukov_plots").absolute()
    script_path = Path(__file__).absolute()
    job.set_workdir(work_dir)
    shutil.copytree(script_path.parent / job.input.dir_path.name, job.input.dir_path, dirs_exist_ok=True)


    main()