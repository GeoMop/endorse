from pathlib import Path
from typing import *
from arviz import InferenceData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.cbook import boxplot_stats
from scipy.integrate import quad
from scipy.optimize import brentq

from chodby_inv import input_data as inputs
from .idata_tools import read_idata_from_file
work_dir = inputs.work_dir


from math import sqrt, erf
from scipy.stats import t


def welch_t_from_stats(
    mean1: float, std1: float, n1: int,
    mean2: float, std2: float, n2: int,
) -> tuple[float, float, float]:
    """
    Welch's two-sample t-test (unequal variances) from summary stats.
    Returns (t_stat, df, p_value) for a two-sided test.
    """
    v1 = (std1**2) / n1
    v2 = (std2**2) / n2
    se = sqrt(v1 + v2)
    t_stat = (mean1 - mean2) / se

    # Welch–Satterthwaite degrees of freedom
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    p_val = 2 * t.sf(abs(t_stat), df)
    return t_stat, df, p_val

def _two_sided_normal_p_from_z(z: float) -> float:
    """
    Two-sided p-value for a standard normal test statistic z.
    Uses erf (no scipy needed).
    """
    # Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
    phi = 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))
    return 2.0 * (1.0 - phi)


def ztest_sigma_based(
    mean_inv: float, std_inv: float,
    mean_meas: float,
) -> tuple[float, float]:
    """
    Sigma-based Z test: treat inversion std as uncertainty of the *quantity*
    (NOT standard error of the mean). Measurement is treated as certain.

    z = (mean_meas - mean_inv) / std_inv
    Returns (z, two-sided p-value).
    """
    if std_inv == 0:
        z = float("inf") if mean_meas != mean_inv else 0.0
        p = 0.0 if mean_meas != mean_inv else 1.0
        return z, p

    z = (mean_meas - mean_inv) / std_inv
    p = _two_sided_normal_p_from_z(z)
    return z, p

def _fmt_p_one_sig_digit(p: float) -> str:
    """
    Scientific notation with 1 significant digit, e.g. 3e-2, 1e-4, 1e+0.
    """
    # Keep it stable for very small/large values; also avoid "-0"
    if p == 0.0:
        return "0e+0"
    return f"{p:.0e}"


def plot_flow_errorbars(
    csv_path: Path,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9, 10),
    n_inv: int = 1000,
    n_meas: int = 5,
    alpha: float = 0.05,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Read the CSV at `csv_path` and plot flow error bars:
      - 2024 unknown mean±std (blue,  label: '2024 inv')
      - 2024 known   mean±std (blue,  same label: '2024 inv')  [two realizations]
      - 2025 unknown mean±std (red,   label: '2025 inv')
      - 2025 known   mean±std (green, label: '2025 measured')

    One borehole-section per line, sorted by smallest->largest 2025 known mean.

    Additionally:
      - Welch t-test (unequal variances) between 2025 inv (unknown) and 2025 measured (known)
        using summary stats and sample sizes n_inv=1000 and n_meas=5 (defaults).
      - If p < alpha, append "(p=...)" to the y-label (scientific notation, 1 significant digit)
        and color that y-label red.

    Saves figure as: work_dir / csv_path.with_suffix(".pdf").name
    Returns (fig, ax).
    """


    # Read CSV; tolerate trailing "===" line or other junk lines starting with '='
    df = pd.read_csv(
        csv_path,
        comment="=",  # ignores lines starting with '=' (e.g., "===")
        skip_blank_lines=True,
    )

    required = [
        "borehole", "section",
        "flow_unknown_mean_2024", "flow_unknown_std_2024",
        "flow_unknown_mean_2025", "flow_unknown_std_2025",
        "flow_known_mean_2024", "flow_known_std_2024",
        "flow_known_mean_2025", "flow_known_std_2025",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric for numeric columns
    num_cols = [c for c in required if c not in ("borehole", "section")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing critical 2025 stats (needed for sorting + testing)
    df = df.dropna(subset=[
        "flow_known_mean_2025",
        "flow_unknown_mean_2025", "flow_unknown_std_2025",
    ]).reset_index(drop=True)

    # Sort by measured flow (2025 known mean)
    df = df.sort_values("flow_known_mean_2025", ascending=True).reset_index(drop=True)

    # Pair label
    df["pair"] = df["borehole"].astype(str) + "-" + df["section"].astype(str)

    # Sigma-based Z test: 2025 inv (unknown) vs 2025 measured (known)
    pvals = []
    reject = []
    for _, r in df.iterrows():
        _, p = ztest_sigma_based(
            mean_inv=float(r["flow_unknown_mean_2025"]),
            std_inv=float(r["flow_unknown_std_2025"]),
            mean_meas=float(r["flow_known_mean_2025"]),
        )
        pvals.append(p)
        reject.append(p < alpha)

    df["pval_2025_sigma_z"] = pvals
    df["reject_2025_sigma_z"] = reject

    # y positions
    y = list(range(len(df)))

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Small vertical offsets so 4 errorbars don't fully overlap on each line
    offsets = {
        "2024_unknown": -0.18,
        "2024_known": -0.06,
        "2025_unknown": 0.06,
        "2025_known": 0.18,
    }

    flow_unit=-6  # to plot in ml/s instead of m^3/s
    # Plot: 2024 inv (two realizations, both blue, same legend label once)
    ax.errorbar(
        df["flow_unknown_mean_2024"]-flow_unit, [i + offsets["2024_unknown"] for i in y],
        xerr=df["flow_unknown_std_2024"],
        fmt="o", capsize=3, color="blue", label="2024 inv"
    )
    ax.errorbar(
        df["flow_known_mean_2024"]-flow_unit, [i + offsets["2024_known"] for i in y],
        xerr=df["flow_known_std_2024"],
        fmt="o", capsize=3, color="blue", label="_nolegend_"
    )

    # Plot: 2025 inv unknown (red)
    ax.errorbar(
        df["flow_unknown_mean_2025"]-flow_unit, [i + offsets["2025_unknown"] for i in y],
        xerr=df["flow_unknown_std_2025"],
        fmt="o", capsize=3, color="red", label="2025 inv"
    )

    # Plot: 2025 measured known (green)
    ax.errorbar(
        df["flow_known_mean_2025"]-flow_unit, [i + offsets["2025_known"] for i in y],
        xerr=df["flow_known_std_2025"],
        fmt="o", capsize=3, color="green", label="2025 measured"
    )

    # Labels: add p-value only for rejected cases
    labels = []
    for pair, rej, p in zip(df["pair"], df["reject_2025_sigma_z"], df["pval_2025_sigma_z"]):
        if rej:
            labels.append(f"{pair} (p={_fmt_p_one_sig_digit(float(p))})")
        else:
            labels.append(str(pair))

    # Formatting
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    # Color rejected tick labels red
    for tick, rej in zip(ax.get_yticklabels(), df["reject_2025_sigma_z"]):
        if rej:
            tick.set_color("red")

    ax.invert_yaxis()  # top = smallest measured flow (after sorting)
    ax.set_xlabel("log10 flow (mean ± std) [ml/s]")
    ax.set_ylabel("Borehole-section (sorted by 2025 measured)")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()

    out_file = work_dir / csv_path.with_suffix(".pdf").name
    fig.savefig(out_file)



def P_cubic(p, H0, p_inf):
    """
    Cubic P(p) in the first integral:
      (H0+p)^2 (p')^2 = c * P(p)
    """
    return (2.0/3.0)*p**3 + (H0 - p_inf)*p**2 - 2.0*H0*p_inf*p + (H0*p_inf**2 + (1.0/3.0)*p_inf**3)


def x_of_p(p, c, H0, p_inf, p0=0.0):
    """
    x(p) = ∫_{p0}^{p} (H0+u)/sqrt(c*P(u)) du
    Assumes c>0, H0>0, and p is within the physically valid interval where P(u)>0.
    """
    if p == p0:
        return 0.0

    def integrand(u):
        val = P_cubic(u, H0, p_inf)
        if val <= 0:
            # Outside admissible region (or numerical roundoff near root).
            return np.inf
        return (H0 + u) / np.sqrt(c * val)

    # quad handles mild endpoint behavior; tighten tolerances if needed
    res, err = quad(integrand, p0, p, epsabs=1e-10, epsrel=1e-10, limit=200)
    return res


def p_of_x(x, c, H0, p_inf, p0=0.0):
    """
    x can be scalar or array-like.
    Returns scalar or numpy array accordingly.
    """
    x_arr = np.asarray(x)

    def solve_one(xs):
        xs = float(xs)
        if xs <= 0:
            return float(p0)

        def f(p):
            return x_of_p(p, c, H0, p_inf, p0=p0) - xs

        return brentq(f, p0 + 1e-12, p_inf - 1e-12, maxiter=200)

    if x_arr.ndim == 0:
        return solve_one(x_arr)

    return np.array([solve_one(xs) for xs in x_arr], dtype=float)




def plot_p_far_errorbars_welch(
    csv_path: Path,
    dist_df: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (9, 10),
    n_2024: int = 1000,
    n_2025: int = 1000,
    alpha: float = 0.05,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Read CSV at `csv_path` and plot p_far error bars (mean±std):
      - 2024 posterior mean±std (blue, label: '2024 inv')
      - 2025 posterior mean±std (red,  label: '2025 inv')

    One borehole-section per line, sorted by `sort_by` (default: 2025 mean).

    Significance:
      - Welch t-test on means using your existing `welch_t_from_stats`
      - Reject if p < alpha
      - For rejected cases: color y-label red and append "(p=...)" with 1 sig digit sci format.

    Saves: work_dir / csv_path.with_suffix(".pdf").name
    Returns (fig, ax).

    NOTE: This function expects `welch_t_from_stats` to be defined in scope:
        welch_t_from_stats(mean1, std1, n1, mean2, std2, n2) -> (t_stat, df, p_value)
    """
    inf_p = 1000
    cond = 1e-13
    source_thickness = 40

    # considering 4 times higher conductivity over 40m rock
    # and considering 1D problem of thickness 4m (~diameter of l5)
    cond_source = (cond / source_thickness / 4)
    lambda_exp = np.sqrt(cond_source / cond)
    model_p = 0.4*inf_p * (1.0  - np.exp(-lambda_exp * np.abs(dist_df['sensor_l5_dist'])))
    dist_df = dist_df.copy()
    dist_df['model_p_far'] = model_p
    H0 = 3 # effective saturated height at L5 wall (in  [m])
    cond_frac = 1 / source_thickness 
    dist_df['model_p2'] = p_of_x(np.abs(dist_df['sensor_l5_dist']),
                                 c=cond_frac,
                                 H0=H0,
                                 p_inf=inf_p,
                                 p0=0.0,)

    df = pd.read_csv(
        csv_path,
        comment="=",
        skip_blank_lines=True,
    )

    required = [
        "borehole", "section",
        "p_far_mean_2024", "p_far_std_2024",
        "p_far_mean_2025", "p_far_std_2025",
        "p_far manual 2024", "p_far manual 2025",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # numeric columns
    num_cols = [c for c in required if c not in ("borehole", "section")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # drop rows missing critical values
    df = df.dropna(subset=required[2:]).copy()

    # sort
    df['diff'] = df['p_far_mean_2025'] - df['p_far_mean_2024']
    df = df.sort_values(by='diff', ascending=True).reset_index(drop=True)

    df["pair"] = df["borehole"].astype(str) + "-" + df["section"].astype(str)
    y = list(range(len(df)))

    # --- ONLY change: add df['sensor_l5_dist'] and use it as y-coordinate ---
    dist_df['borehole'] = [s[len('L5-'):] for s in dist_df['borehole'].values]
    dist_df = dist_df.reset_index()
    df = df.merge(
        dist_df[["borehole", "section", "sensor_l5_dist"]],
        on=["borehole", "section"],
        how="left",
    )
    y = df["sensor_l5_dist"]

    # --- NEW: deterministic jitter for near-identical distances (treat equal if |diff| < 0.2) ---
    # Cluster distances within 0.2 into the same "row group", then spread within each group.
    _tol = 0.2
    _step = 0.3  # vertical separation within a group; adjust if you want

    # make cluster id by walking distances in sorted order (single-linkage in 1D)
    _d = df["sensor_l5_dist"].values
    _order = np.argsort(_d)
    _cluster = np.zeros(len(df), dtype=int)
    _cid = 0
    _prev = _d[_order[0]]
    _cluster[_order[0]] = _cid
    for idx in _order[1:]:
        if abs(_d[idx] - _prev) >= _tol:
            _cid += 1
        _cluster[idx] = _cid
        _prev = _d[idx]
    df["_dist_cluster"] = _cluster

    _k = df.groupby("_dist_cluster").cumcount()  # 0,1,2,...
    _n = df.groupby("_dist_cluster")["sensor_l5_dist"].transform("size")
    _centered = _k - (_n - 1) / 2.0  # e.g. n=3 -> [-1,0,1]
    df["sensor_l5_dist_jitter"] = df["sensor_l5_dist"] + _centered * _step
    y_j = df["sensor_l5_dist_jitter"]

    # Welch significance: 2025 vs 2024
    pvals = []
    reject = []
    for _, r in df.iterrows():
        _, _, p = welch_t_from_stats(
            float(r["p_far_mean_2024"]), float(r["p_far_std_2024"]), n_2024,
            float(r["p_far_mean_2025"]), float(r["p_far_std_2025"]), n_2025,
        )
        pvals.append(p)
        reject.append(p < alpha)

    df["pval_2025_vs_2024_welch"] = pvals
    df["reject_2025_vs_2024_welch"] = reject

    fig, ax = plt.subplots(figsize=figsize)

    # model line directly from dist_df (original distances, no jitter)
    # model line directly from dist_df, sorted by distance
    _dplot = dist_df.sort_values("sensor_l5_dist")
    # ax.plot(
    #     _dplot["model_p_far"],
    #     _dplot["sensor_l5_dist"],
    #     color='grey',
    #     linestyle="-",
    #     marker=None,
    #     alpha=0.5,
    #     label="homogeneous model",
    # )
    ax.plot(
        _dplot["model_p2"],
        _dplot["sensor_l5_dist"],
        color='grey',
        linestyle="-",
        marker=None,
        alpha=0.5,
        label="homogeneous model",
    )

    offsets = {"2024": -0.08, "2025": 0.08}

    # loop over idatas and get their boxplot stats

    # ax.errorbar(
    #     df["p_far_mean_2024"], y_j + offsets["2024"],
    #     xerr=df["p_far_std_2024"],
    #     fmt="o", capsize=3, color="blue", label="2024 inversion"
    # )
    # ax.errorbar(
    #     df["p_far_mean_2025"], y_j + offsets["2025"],
    #     xerr=df["p_far_std_2025"],
    #     fmt="o", capsize=3, color="red", label="2025 inversion"
    # )

    stats24, stats25 = collect_boxplot_stats_from_rundir(
        boreholes = [str(p) for p in df["pair"]],
        dataset = "posterior",
        var_name = "p_far"
    )

    print(stats24)

    ax.bxp(stats24, orientation="horizontal", showfliers=False)
    ax.bxp(stats25, orientation="horizontal", showfliers=False)

    ax.scatter(
        df["p_far manual 2024"],
        y_j + offsets["2024"],
        marker="o",
        color="cyan",
        label="2024 manual",
    )
    ax.scatter(
        df["p_far manual 2025"], y_j + offsets["2025"],
        marker="o", color="orange", label="2025 manual"
    )

    # labels: use jittered y coord for tick placement + borehole-section labels (LEFT axis)
    labels = [str(p) for p in df["pair"]]
    ax.set_yticks(y_j)
    ax.set_yticklabels(labels)

    # RIGHT axis: one tick/label per aggregated distance group, at the (original) group distance
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())

    # one distance value per cluster (choose mean; could also be min/median)
    dist_per_group = df.groupby("_dist_cluster")["sensor_l5_dist"].mean().sort_values()
    y_right = dist_per_group.values
    labels_right = [f"{v:.2f}" for v in dist_per_group.values]

    ax_r.set_yticks(y_right)
    ax_r.set_yticklabels(labels_right)
    ax_r.set_ylabel("sensor distance from L5 [m]", fontsize=14, fontweight="bold")

    # keep your x tick locators on the main axis
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    #ax.invert_yaxis()
    ax.set_xlabel("pore pressure [kPa]", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"borehole section", fontsize=14, fontweight="bold")
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.25, linewidth=0.6)

    ax.legend(loc="best")

    fig.tight_layout()
    out_file = work_dir / csv_path.with_suffix(".pdf").name
    fig.savefig(out_file)

    return fig, ax

def collect_boxplot_stats_from_rundir(
        boreholes: list[str],
        dataset: str,
        var_name: str
    ) -> list[dict[str, Any]]:
    """
    Collect boxplot statistics for multiple boreholes from .idata files in CWD.

    :param order: Order of boreholes to process
    :type order: list[str]
    :param dataset: Subset of the InferenceData to analyze (e.g., 'posterior')
    :type dataset: str
    :param var_name: Name of the variable to extract boxplot statistics for
    :type var_name: str
    :return: List of dictionaries containing boxplot statistics for each InferenceData
    :rtype: list[dict[str, Any]]
    """

    rundir = Path.cwd()
    directory = rundir / "dataset9"

    stats_2024 = []
    stats_2025 = []

    for borehole in boreholes:
        borehole_fixed = borehole.replace("-", "_")
        matching = [
            file for file in directory.iterdir()
            if file.is_file() and ".idata" in file.name and borehole_fixed in file.name
        ]

        assert len(matching) == 2, f"Expected 2 matching .idata files for {borehole}, found {len(matching)}"

        #if (len(matching) != 2):
        #    print(f"Warning: Expected 2 matching .idata files for {borehole}, found {len(matching)}")
        #    continue

        for path in matching:
            idata = read_idata_from_file(path.absolute())
            stats = get_boxplot_stats(idata, dataset, var_name)
            stats.pop("fliers", None)  
            if "2024" in path.name:
                stats_2024.append(stats)
            elif "2025" in path.name:
                stats_2025.append(stats)
            del idata

    return stats_2024, stats_2025

def collect_boxplot_stats(
    idata_paths: list[Path],
    dataset: str,
    var_name: str
) -> list[dict[str, Any]]:
    """
    Docstring for collect_boxplot_stats
    
    :param idata_paths: List of file paths to InferenceData objects
    :type idata_paths: list[str]
    :param dataset: Subset of the InferenceData to analyze (e.g., 'posterior')
    :type dataset: str
    :param var_name: Name of the variable to extract boxplot statistics for
    :type var_name: str
    :return: List of dictionaries containing boxplot statistics for each InferenceData
    :rtype: list[dict[str, Any]]
    """
    stats_list = []
    for path in idata_paths:
        idata = read_idata_from_file(path.absolute())
        stats = get_boxplot_stats(idata, dataset, var_name)
        stats_list.append(stats)
        del idata
    return stats_list

def get_boxplot_stats(
        idata: InferenceData,
        dataset: str,
        var_name: str
    ) -> dict[str, Any]:
    """
    Docstring for get_boxplot_stats
    
    :param idata: InferenceData object containing the data to be processed
    :type idata: InferenceData
    :param dataset: Subset of the InferenceData to analyze (e.g., 'posterior')
    :type dataset: str
    :param var_name: Name of the variable to extract boxplot statistics for
    :type var_name: str
    :return: Dictionary containing boxplot statistics
    :rtype: dict[str, Any]
    """

    data_array = idata[dataset][var_name].values.flatten()
    stats = boxplot_stats(data_array, whis=1.5)[0]
    return stats
