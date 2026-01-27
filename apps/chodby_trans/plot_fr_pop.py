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

def load_fixed_population(filename):
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


########################################################################################################################

# ----------------------------
# Geometry helpers (ENU coords)
# x = East, y = North, z = Up
# Azimuth/trend measured clockwise from North.
# Plunge positive downward.
# Assumes strike is Right-Hand-Rule (dip direction = strike + 90).
# ----------------------------

def trend_plunge_to_vector(trend_deg, plunge_deg):
    t = np.deg2rad(trend_deg)
    p = np.deg2rad(plunge_deg)
    # components: East, North, Up
    e = np.cos(p) * np.sin(t)
    n = np.cos(p) * np.cos(t)
    u = -np.sin(p)
    v = np.stack([e, n, u], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def strike_dip_to_pole(strike_deg, dip_deg):
    # Pole trend points toward dip direction (RHR): strike + 90
    pole_trend = (strike_deg + 90.0) % 360.0
    pole_plunge = 90.0 - dip_deg
    return trend_plunge_to_vector(pole_trend, pole_plunge)

def vector_to_trend_plunge(v):
    v = np.asarray(v)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    e, n, u = v[..., 0], v[..., 1], v[..., 2]
    trend = (np.rad2deg(np.arctan2(e, n)) + 360.0) % 360.0
    plunge = np.rad2deg(np.arctan2(-u, np.sqrt(e**2 + n**2)))
    return trend, plunge

def pole_to_strike_dip(pole_vec):
    trend, plunge = vector_to_trend_plunge(pole_vec)
    dip = 90.0 - plunge
    strike = (trend - 90.0) % 360.0  # RHR strike
    return strike, dip

# ----------------------------
# von Mises–Fisher sampling on S^2 (Fisher in 3D)
# Uses Wood (1994)-style sampler.
# ----------------------------

def sample_vmf_3d(mu, kappa, n, rng=None):
    """
    Sample n unit vectors from vMF on S^2 with mean direction mu and concentration kappa.
    """
    rng = np.random.default_rng(rng)
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)

    if kappa <= 1e-12:
        # essentially uniform on sphere
        x = rng.normal(size=(n, 3))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    # Step 1: sample w = cos(theta)
    # For p=3, the distribution of w has a simple form:
    # w = 1 + (1/kappa) * log(u + (1-u)*exp(-2*kappa))
    u = rng.random(n)
    w = 1.0 + (1.0 / kappa) * np.log(u + (1.0 - u) * np.exp(-2.0 * kappa))

    # Step 2: sample a random unit vector orthogonal component
    phi = 2.0 * np.pi * rng.random(n)
    s = np.sqrt(1.0 - w**2)
    x = np.stack([s * np.cos(phi), s * np.sin(phi), w], axis=1)  # around +z

    # Step 3: rotate +z to mu
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(mu, z):
        return x
    if np.allclose(mu, -z):
        x[:, 2] *= -1
        return x

    v = np.cross(z, mu)
    c = np.dot(z, mu)
    s_rot = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s_rot**2))
    return x @ R.T

# ----------------------------
# Rose diagram (axial option for strike)
# ----------------------------

def set_rlabels_min_density(ax, angles_deg, *, bins=36, axial=False, pad=8):
    angles = np.asarray(angles_deg) % 360.0
    if axial:
        angles = angles % 180.0
        theta = np.deg2rad(2.0 * angles)
    else:
        theta = np.deg2rad(angles)

    counts, edges = np.histogram(theta, bins=bins, range=(0.0, 2*np.pi))
    centers = edges[:-1] + np.diff(edges)/2.0
    theta_min = centers[np.argmin(counts)]
    pos = (np.rad2deg(theta_min)) % 360.0

    ax.set_rlabel_position(pos)
    ax.tick_params(axis="y", pad=pad)
    return pos

def add_params_caption(ax, strike_deg, dip_deg, kappa, *, offset_pts=14, fontsize=11):
    """
    Places a caption centered under the polar axes.
    offset_pts controls distance below the axes (in points). Smaller = closer.
    """
    caption = rf"strike={strike_deg:.1f}°  dip={dip_deg:.1f}°  $\kappa$={kappa:.1f}"

    ax.annotate(
        caption,
        xy=(0.5, 0.0), xycoords="axes fraction",   # bottom-center of axes
        xytext=(0, -offset_pts), textcoords="offset points",
        ha="center", va="top",
        fontsize=fontsize,
        clip_on=False,
    )


def plot_rose_azimuth(
    angles_deg,
    *,
    bins=36,
    axial=False,          # True for strike (0–180 equivalence)
    ax=None,
    density=True,
    facecolor=None,
    edgecolor=None,
    linewidth=1.0,
    mean_line=True,
    mean_kwargs=None,
):
    """
    Rose diagram of azimuth angles (deg). If axial=True, treats theta == theta+180.
    """
    angles = np.asarray(angles_deg) % 360.0

    if axial:
        # map to [0,180) then double-angle to use circular histogram properly
        angles = angles % 180.0
        theta = np.deg2rad(2.0 * angles)
    else:
        theta = np.deg2rad(angles)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # histogram on [0, 2pi)
    counts, edges = np.histogram(theta, bins=bins, range=(0.0, 2.0 * np.pi), density=density)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2.0

    bars = ax.bar(
        centers, counts, width=widths, align="center",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth
    )

    # Polar formatting: geology convention often has 0° at North and clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(axis="x", pad=12)

    # Mean line
    if mean_line:
        mean_kwargs = mean_kwargs or {}
        # circular mean for directional; for axial we compute mean on doubled angles then halve
        if axial:
            m = np.angle(np.mean(np.exp(1j * theta)))
            mean_theta = (m % (2*np.pi)) / 2.0  # halve back to axial angle in radians
            # draw line both directions
            for add in [0.0, np.pi]:
                ax.plot([mean_theta + add, mean_theta + add], [0, counts.max() if counts.size else 1],
                        **{"lw": 2.0, **mean_kwargs})
        else:
            m = np.angle(np.mean(np.exp(1j * theta))) % (2*np.pi)
            ax.plot([m, m], [0, counts.max() if counts.size else 1],
                    **{"lw": 2.0, **mean_kwargs})

    return ax

# ----------------------------
# High-level: from (strike,dip,kappa) to rose plot
# ----------------------------

def fisher_fracture_rose(strike_deg, dip_deg, kappa, *, n=10000, bins=36, axial_strike=True, rng=None, ax=None):
    mu_pole = strike_dip_to_pole(strike_deg, dip_deg)
    poles = sample_vmf_3d(mu_pole, kappa, n=n, rng=rng)
    strikes, dips = pole_to_strike_dip(poles)

    ax = plot_rose_azimuth(
        strikes,
        ax=ax,
        bins=bins,
        axial=axial_strike,
        facecolor="tab:green",
        edgecolor="k",
        linewidth=0.8,
        mean_line=True,
        mean_kwargs={"color": "tab:blue"},
    )
    set_rlabels_min_density(ax, strikes, bins=36, axial=True, pad=10)
    add_params_caption(ax, strike_deg, dip_deg, kappa, fontsize=16, offset_pts=40)
    # ax.text(
    #     0.5, -0.18,
    #     rf"strike={strike_deg:.1f}°  dip={dip_deg:.1f}°  $\kappa$={kappa:.1f}",
    #     transform=ax.transAxes,
    #     ha="center", va="top",
    #     fontsize=16,
    #     clip_on=False,
    # )

    return ax, strikes, dips, poles



########################################################################################################################

def main():
    fish_par = load_fixed_population(f"fr_Bukov_bayes/fixed_params.csv")

    for i in range(5):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
        dip, strike, kappa = fish_par[i][1], fish_par[i][2], fish_par[i][3]
        ax, strikes, dips, poles = fisher_fracture_rose(strike, dip, kappa, n=10000, bins=36, axial_strike=True, ax=ax)
        # plt.show()
        ax.set_title(f"Population {i+1}", pad=10)
        # ax.set_title(rf"Strike rose (Fisher/vMF samples) $\kappa={kappa}$", pad=14)
        fig.tight_layout()
        fig.savefig(job.output.dir_path / f"fr_rose_{i + 1}.pdf")

        # continue

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