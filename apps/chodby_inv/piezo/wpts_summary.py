from pathlib import Path
from typing import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from chodby_inv import input_data as inputs
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

def plot_p_far_errorbars_welch(
    csv_path: Path,
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

    df = pd.read_csv(
        csv_path,
        comment="=",
        skip_blank_lines=True,
    )

    required = [
        "borehole", "section",
        "p_far_mean_2024", "p_far_std_2024",
        "p_far_mean_2025", "p_far_std_2025",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # numeric columns
    num_cols = [c for c in required if c not in ("borehole", "section")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # drop rows missing critical values
    df = df.dropna(subset=["p_far_mean_2024", "p_far_std_2024", "p_far_mean_2025", "p_far_std_2025"]).copy()

    # sort
    if sort_by not in df.columns:
        raise ValueError(f"sort_by='{sort_by}' not in columns; choose from {list(df.columns)}")
    df['diff'] = df['p_far_mean_2025'] - df['p_far_mean_2024']
    df = df.sort_values(sort_by, ascending=True).reset_index(drop=True)

    df["pair"] = df["borehole"].astype(str) + "-" + df["section"].astype(str)
    y = list(range(len(df)))

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

    offsets = {"2024": -0.10, "2025": 0.10}

    ax.errorbar(
        df["p_far_mean_2024"], [i + offsets["2024"] for i in y],
        xerr=df["p_far_std_2024"],
        fmt="o", capsize=3, color="blue", label="2024 inv"
    )
    ax.errorbar(
        df["p_far_mean_2025"], [i + offsets["2025"] for i in y],
        xerr=df["p_far_std_2025"],
        fmt="o", capsize=3, color="red", label="2025 inv"
    )

    # labels: add p-value only for rejected cases
    labels = [str(p) for p in df["pair"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.invert_yaxis()
    ax.set_xlabel("background pressure [kPa]")
    ax.set_ylabel(f"Borehole-section (sorted by {sort_by})")
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.25, linewidth=0.6)

    ax.legend(loc="best")

    fig.tight_layout()
    out_file = work_dir / csv_path.with_suffix(".pdf").name
    fig.savefig(out_file)

def wpt_summary(flow_file, pressure_file):
    plot_flow_errorbars(flow_file)
    plot_p_far_errorbars_welch(pressure_file)