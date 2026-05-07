from typing import *
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.dates as mdates
import cmasher as cmr
matplotlib.use('TkAgg')


from endorse import common
from chodby_inv import input_data as inputs


# interesting_sections=[
#        ("22DR", "2"),
#         ("37R", "2"),
#         ("26R", "2"),
#         ("26R", "1"),
#         ("23UR", "0"),
#         ("23UR", "2"),
#
#         ("37UR", "1"),
#         ("22DR", "1"),
#     ("50UL", "1"),
#         ("26R", "0"),
#         ("23UR", "1"),
#     ("49DL", "0"),
# ]

large_p = [
        ("22DR", "2"),
        ("26R", "2"),
        ("37R", "2"),

        ("37UR", "2"),
        ("26R", "1"),
        ("37R", "0"),

        ("23UR", "0"),
        ("23UR", "1"),
        ("23UR", "2"),

        ("22DR", "1"),
        ("26R", "0"),
        ("22DR", "0"),
]
small_p = [
    ("24DR", "2"),
    ("24DR", "1"),
    ("24DR", "0"),

    ("37UR", "1"),
    ("37R", "1"),
    ("37UR", "0"),

    ("50UL", "0"),
    ("50UL", "1"),
    ("49DL", "2"),

    ("49DL", "0"),
    ("50UL", "2"),
    ("49DL", "1"),
]


# def plot_wpts(df: pd.DataFrame, workdir: Path):
#     """
#     """
#     process_cfg = common.config.load_config(inputs.piezo_filter_yaml)
#     events_cfg = common.config.load_config(inputs.events_yaml)
#     wpts = events_cfg['water_pressure_tests']
#     wpt_df = pd.DataFrame.from_records(wpts)
#     wpt_df['start'] = pd.to_datetime(wpt_df['start'], format="%y/%m/%d %H:%M:%S")
#     wpt_df['origin'] = pd.to_datetime(wpt_df['origin'], format="%y/%m/%d %H:%M:%S")
#     wpt_df['end'] = pd.to_datetime(wpt_df['end'], format="%y/%m/%d %H:%M:%S") + pd.Timedelta(days=7)
#     for (bh, sec), grp in df.groupby(["borehole", "section"], sort=False):
#         wpt_events = wpt_df.loc[(wpt_df["borehole"] == bh) & (wpt_df["section"] == sec)]
#         times  = [wpt_events[k].to_numpy() for k in ["start", "origin", "end"]]
#         fig, _, _ = plot_three_synced_windows(grp, times,
#                                   title=f"WPT - borehole {bh} section {sec}",
#                                   file=workdir / f"wpt_{bh}_{sec}.pdf")
#


def plot_wpts(df: pd.DataFrame, workdir: Path):
    events_cfg = common.config.load_config(inputs.events_yaml)

    wpts = events_cfg["water_pressure_tests"]
    wpt_df = pd.DataFrame.from_records(wpts)
    wpt_df["start"]  = pd.to_datetime(wpt_df["start"],  format="%y/%m/%d %H:%M:%S")
    wpt_df["origin"] = pd.to_datetime(wpt_df["origin"], format="%y/%m/%d %H:%M:%S")
    wpt_df["end"]    = pd.to_datetime(wpt_df["end"],    format="%y/%m/%d %H:%M:%S")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1) Collect cases (expects 24 total)
    bhs = large_p + small_p
    cases = []
    for bh, sec in bhs:
        _bh, _sec = f"L5-{bh}", int(sec)
        grp = df[(df["borehole"] == _bh) & (df["section"] == _sec)]
        wpt_events = wpt_df[(wpt_df["borehole"] == _bh) & (wpt_df["section"] == _sec)]
        times = [wpt_events[k].to_numpy() for k in ["start", "origin", "end"]]
        cases.append((bh, sec, grp, times))

    # 2) 6 cases per PDF -> 4 PDFs for 24 cases
    if len(cases) % 6 != 0:
        raise ValueError(f"Expected number of cases to be divisible by 6, got {len(cases)}")

    FIGSIZE = (9, 14)  # 5:3


    for page_idx, start in enumerate(range(0, len(cases), 6), start=1):
        chunk = cases[start:start + 6]

        fig, axs = plt.subplots(3, 2, figsize=FIGSIZE, constrained_layout=True)
        # column-major order so "triplets in columns"
        axes = [axs[r, c] for c in range(2) for r in range(3)]

        for ax, (bh, sec, grp, times) in zip(axes, chunk):
            plot_three_synced_windows(
                grp,
                times,
                title=f"WPT - borehole {bh} section {sec}",
                ax=ax,        # <-- single axis for this single case
                file=None,    # <-- do NOT save inside; save the combined fig below
            )

        out = workdir / f"wpt_grid_{page_idx:02d}.pdf"
        fig.savefig(out, format="pdf")
        plt.close(fig)

def _window(df: pd.DataFrame, dt_start:pd.Timestamp, dt_end:pd.Timestamp) -> pd.DataFrame:
    m = (df["timestamp"] >= dt_start) & (df["timestamp"] <= dt_end)
    return df.loc[m, ["timestamp", "pressure"]]

def _local_maxima_mask(p: np.ndarray) -> np.ndarray:
    """
    Vectorized local maxima mask (no loops), with your definition:
      - interior i: p[i] > p[i-1] and p[i] > p[i+1]
      - edges: p[0] > p[1], p[-1] > p[-2]
    Strict '>' (no plateau-peak detection). This is fast and fully vectorized.
    """
    p = np.asarray(p, dtype=float)
    m = np.zeros_like(p, dtype=bool)
    assert len(p) > 2

    m[0] = p[0] > p[1]
    m[-1] = p[-1] > p[-2]
    m[1:-1] = (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])
    return m

def peak_idx_localmax(w: pd.DataFrame, peak_height_tol: float = 50.0) -> int:
    """
    Pick peak index as LOCAL maximum.
    If multiple peaks of similar height (within peak_height_tol of the highest local max),
    pick the first one.
    """
    p = w["pressure"].to_numpy(dtype=float)

    lm = _local_maxima_mask(p)
    if not lm.any():
        return int(np.argmax(p))  # fallback if no local maxima

    idx = np.flatnonzero(lm)
    pv = p[idx]
    best = pv.max()
    #print(f"  local maxima at indices {idx} with pressures {pv}, best={best}")
    good = idx[pv >= (best *0.9)]
    peak_idx = int(good[-1])
    #print(f"  picked peak at index {peak_idx} with pressure {p[peak_idx]}")
    return peak_idx

# ---- fast sync using first drop below p_target ----

def first_drop_ts_numpy(w: pd.DataFrame, peak_idx: int, p_target: float) -> pd.Timestamp:
    """
    Starting at peak_idx, return timestamp of the first sample where pressure < p_target.
    Assumes w has columns ['timestamp','pressure'] and is sorted.
    """
    p = w["pressure"].to_numpy(dtype=float)
    ts = w["timestamp"].to_numpy(dtype="datetime64[ns]")

    rel = np.flatnonzero(p[peak_idx:] < p_target)
    if rel.size == 0:
        raise ValueError("No sample after peak drops below p_target in this window.")
    return pd.Timestamp(ts[peak_idx + rel[0]])


def sync_ts_by_min_peak_numpy(
    windows: list[pd.DataFrame],
    peak_height_tol: float = 50.0,
) -> tuple[list[pd.Timestamp], float, list[int], list[float]]:
    """
    1) Find peak index per window via local maxima.
    2) p_target = min(peak pressures)
    3) sync_ts[i] = first timestamp after peak where pressure < p_target

    Returns: (sync_ts, p_target, peak_idx, peak_p)
    """
    peak_idx = [peak_idx_localmax(w, peak_height_tol=peak_height_tol) for w in windows]
    peak_p = [float(w["pressure"].iloc[i]) for w, i in zip(windows, peak_idx)]
    p_target = float(min(peak_p))

    sync_ts = [first_drop_ts_numpy(w, i, p_target) for w, i in zip(windows, peak_idx)]
    return sync_ts

# ---- fixed scheme (color + legend label) ----
_SCHEME = {
    "03/2024": dict(color="#9ecae1", sort_key=pd.Timestamp("2024-03-01")),  # light blue
    "09/2024": dict(color="#1f77b4", sort_key=pd.Timestamp("2024-09-01")),  # blue
    "04/2025": dict(color="#d62728", sort_key=pd.Timestamp("2025-04-01")),  # red
}

def _period_label_from_origin(ts: pd.Timestamp) -> str:
    """Map an origin timestamp to one of the three fixed labels."""
    ts = pd.Timestamp(ts)
    if ts.year == 2024 and ts.month <= 6:
        return "03/2024"
    if ts.year == 2024:
        return "09/2024"
    return "04/2025"


def plot_three_synced_windows(
    df: pd.DataFrame,
    times: "Tuple[Sequence[pd.Timestamp], Sequence[pd.Timestamp], Sequence[pd.Timestamp]]",
    ax,
    title: Optional[str] = None,
    file: Optional[Path] = None,
):
    print("Plot three synced windows:", title)
    start_seq, origin_seq, end_seq = times

    # ---- build windows, allowing missing entries ----
    # Treat NaT/None as missing, and keep a label per window based on origin time.
    items = []
    for s, o, e in zip(start_seq, origin_seq, end_seq):
        if s is None or o is None or e is None:
            continue
        s = pd.Timestamp(s)
        o = pd.Timestamp(o)
        e = pd.Timestamp(e)
        if pd.isna(s) or pd.isna(o) or pd.isna(e):
            continue

        label = _period_label_from_origin(o)
        w = _window(df, o - pd.Timedelta(hours=24), e)  # your original definition
        # If _window can return empty, skip it
        if w is None or len(w) == 0:
            continue
        items.append((label, w, s, o, e))

    if not items:
        raise ValueError("No windows available to plot (all missing/empty).")

    # If there are duplicates (e.g., two windows both classified as 09/2024), keep the first.
    dedup = {}
    for label, w, s, o, e in items:
        dedup.setdefault(label, (w, s, o, e))
    items = [(label, *dedup[label]) for label in dedup.keys()]

    # order by the canonical season order (03/2024 -> 09/2024 -> 04/2025)
    items.sort(key=lambda t: _SCHEME[t[0]]["sort_key"])

    labels, windows, starts, origins, ends    = zip(*items)

    sync_ts = sync_ts_by_min_peak_numpy(windows)

    # ---- plot range in shift-space (same idea as your code) ----
    # excavation = pd.Timestamp("2024-03-12 00:00:00")
    # # hack for slow decrease in 22DR-2: end time excavation otherwise
    # mask = df["timestamp"].le(ends[0])
    # idx = df.index[mask.to_numpy().nonzero()[0][-1]]
    # if df.loc[idx, "pressure"] < 100 and ends[0] > excavation:
    #     ends[0] = excavation

    xmin_shift = min(s - st for s, st in zip(starts, sync_ts))
    xmax_shift = max(e - st for e, st in zip(ends[0::2], sync_ts[0::2]))
    # skip autumn 2024 since these does not have proper endtimes

    # i_min = np.argmin(ends)
    # xmax_shift =ends[i_min]- sync_ts[i_min]   # end according to first window only
    # print(title, "\n", ends[i_min])
    reference_index = 0
    base_sync = sync_ts[reference_index]
    x_left = base_sync + xmin_shift
    x_right = base_sync + xmax_shift

    lines = []
    for i, (label, w) in enumerate(zip(labels, windows)):
        w = w[w["timestamp"].between(x_left - base_sync + sync_ts[i],
                                     x_right - base_sync + sync_ts[i])]
        x = base_sync + (w["timestamp"] - sync_ts[i])
        (ln,) = ax.plot(
            x,
            w["pressure"].to_numpy(dtype=float),
            label=label,                      # FIX: "03/2024" etc.
            color=_SCHEME[label]["color"],     # FIX: forced colors
        )
        lines.append(ln)

    ax.set_xlim(x_left, x_right)
    ax.set_ylabel("pressure [kPa]")
    if title:
        ax.set_title(title)

    # vertical sync line
    ax.axvline(base_sync, linestyle="--")

    # legend with fixed labels/colors
    ax.legend(loc="best")

    # ---- remove default x axis visuals (we'll draw ALL time axes ourselves) ----
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.spines["bottom"].set_visible(False)

    # ---- formatter factory: map base x coordinate -> original time for window i ----
    def _fmt_factory(sync_i: pd.Timestamp, base_sync_: pd.Timestamp):
        def _fmt(x, pos=None):
            base_dt = pd.Timestamp(mdates.num2date(x).replace(tzinfo=None))
            orig = sync_i + (base_dt - base_sync_)
            return orig.strftime("%m/%d\n%H:%M")  # <- no year
        return _fmt

    # ---- create one colored time axis per window, stacked downward in SEASON order ----
    gap = 0.12
    base_labelsize = ax.xaxis.get_ticklabels()[0].get_size() if ax.xaxis.get_ticklabels() else 10
    labelsize = max(1, base_labelsize - 2)  # two font sizes smaller

    for k, (label, ln) in enumerate(zip(labels, lines)):
        ax_t = ax.twiny()
        ax_t.set_xlim(ax.get_xlim())

        ax_t.xaxis.set_ticks_position("bottom")
        ax_t.xaxis.set_label_position("bottom")
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["bottom"].set_position(("axes", -gap * k))

        color = _SCHEME[label]["color"]
        ax_t.spines["bottom"].set_color(color)

        # limit to ~8 major ticks (labels)
        ax_t.xaxis.set_major_locator(MaxNLocator(nbins=8))

        ax_t.tick_params(axis="x", colors=color, labelsize=labelsize)
        i = k  # aligned with sync_ts index because we sorted items already
        ax_t.xaxis.set_major_formatter(FuncFormatter(_fmt_factory(sync_ts[i], base_sync)))
        ax_t.patch.set_alpha(0)


    # optional save
    # ax.figure.savefig(file) if file else None

    return sync_ts

    # fig.tight_layout()
    # fig.savefig(file) if file else None
    # fig.show()
    # return fig, ax, sync_ts
