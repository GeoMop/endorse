from typing import *
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import cmasher as cmr
matplotlib.use('TkAgg')

interesting_sections=[
    ("22DR", "2"),
    ("37R", "2"),
    ("26R", "2"),
    ("26R", "1"),
    ("23UR", "0"),
    ("23UR", "2"),

    ("37UR", "1"),
    ("22DR", "1"),
    ("50UL", "1"),
    ("26R", "0"),
    ("23UR", "1"),
    ("49DL", "0"),
]
from endorse import common
from chodby_inv import input_data as inputs

def plot_wpts(df: pd.DataFrame, workdir: Path):
    """
    """
    process_cfg = common.config.load_config(inputs.piezo_filter_yaml)
    events_cfg = common.config.load_config(inputs.events_yaml)
    wpts = events_cfg['water_pressure_tests']
    wpt_df = pd.DataFrame.from_records(wpts)
    wpt_df['start'] = pd.to_datetime(wpt_df['start'], format="%y/%m/%d %H:%M:%S")
    wpt_df['origin'] = pd.to_datetime(wpt_df['origin'], format="%y/%m/%d %H:%M:%S")
    wpt_df['end'] = pd.to_datetime(wpt_df['end'], format="%y/%m/%d %H:%M:%S") + pd.Timedelta(days=7)
    for (bh, sec), grp in df.groupby(["borehole", "section"], sort=False):
        wpt_events = wpt_df.loc[(wpt_df["borehole"] == bh) & (wpt_df["section"] == sec)]
        times  = [wpt_events[k].to_numpy() for k in ["start", "origin", "end"]]
        fig, _, _ = plot_three_synced_windows(grp, times,
                                  title=f"WPT - borehole {bh} section {sec}",
                                  file=workdir / f"wpt_{bh}_{sec}.pdf")





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
    print(f"  local maxima at indices {idx} with pressures {pv}, best={best}")
    good = idx[pv >= (best *0.9)]
    peak_idx = int(good[-1])
    print(f"  picked peak at index {peak_idx} with pressure {p[peak_idx]}")
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

def plot_three_synced_windows(
    df: pd.DataFrame,
    times: '3*Tuple[Sequence[pd.Timestamp,]]',
    title: Optional[str] = None,
    file: Optional[Path] = None,
):
    print("Plot three synced windows:", title)
    start, origin, end = times
    windows = [_window(df, a, b) for a, b in zip(origin - pd.Timedelta(hours=24), end)]
    sync_ts = sync_ts_by_min_peak_numpy(windows)

    # Plot range in shift-space
    exxcavation = '2024-03-12 00:00:00'
    # hack for slow decrease in 22DR-2
    # end time excavation otherwise
    mask = df["timestamp"].le(end[0])
    idx = df.index[mask.to_numpy().nonzero()[0][-1]]
    if df.loc[idx, "pressure"] < 100 and end[0] > pd.Timestamp(exxcavation):
        end[0] = pd.Timestamp(exxcavation)
    xmin_shift = min(s - st for s, st in zip(start, sync_ts))
    xmax_shift = min(e - st for e, st in zip(end, sync_ts))
    reference_index = 0
    base_sync = sync_ts[reference_index]
    x_left = base_sync + xmin_shift
    x_right = base_sync + xmax_shift

    fig, ax = plt.subplots(figsize=(10, 8))

    lines = []
    for i, w in enumerate(windows):
        x = base_sync + (df["timestamp"] - sync_ts[i])
        (ln,) = ax.plot(x, df["pressure"].to_numpy(dtype=float), label=f"window {i + 1}")
        lines.append(ln)

    ax.set_xlim(x_left, x_right)
    ax.set_ylabel("pressure [kPa]")
    if title:
        ax.set_title(title)

    # vertical sync line
    ax.axvline(base_sync, linestyle="--")

    # keep legend for the lines
    ax.legend(loc="best")

    # ---- remove the default x axis visuals (we'll draw ALL time axes ourselves) ----
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.spines["bottom"].set_visible(False)

    # ---- formatter factory: map base x coordinate -> original time for window i ----
    def _fmt_factory(sync_i: pd.Timestamp, base_sync_: pd.Timestamp):
        def _fmt(x, pos=None):
            base_dt = pd.Timestamp(mdates.num2date(x).replace(tzinfo=None))
            orig = sync_i + (base_dt - base_sync_)
            return orig.strftime("%y/%m/%d\n%H:%M")

        return _fmt

    # ---- create one colored time axis per window, ordered by time (sync_ts) ----
    order = sorted(range(len(windows)), key=lambda i: sync_ts[i])  # earliest -> latest

    # spacing between stacked axes (in axes coordinates, negative = below plot)
    gap = 0.12

    # make room for stacked axes
    fig.subplots_adjust(bottom=0.20 + gap * max(0, len(order) - 1))

    for k, i in enumerate(order):
        ax_t = ax.twiny()
        ax_t.set_xlim(ax.get_xlim())

        # put this axis at the bottom, stacked downward
        ax_t.xaxis.set_ticks_position("bottom")
        ax_t.xaxis.set_label_position("bottom")
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["bottom"].set_position(("axes", -gap * k))

        # consistent color = the line color for that window
        color = lines[i].get_color()
        ax_t.spines["bottom"].set_color(color)
        ax_t.tick_params(axis="x", colors=color)
        ax_t.xaxis.set_major_formatter(FuncFormatter(_fmt_factory(sync_ts[i], base_sync)))

        # optional: keep the background clean
        ax_t.patch.set_alpha(0)

    fig.tight_layout()
    fig.savefig(file) if file else None
    return fig, ax, sync_ts
