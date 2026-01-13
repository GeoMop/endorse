from typing import *
from pathlib import Path
import pandas as pd

import cmasher as cmr
import matplotlib
from matplotlib.colors import TwoSlopeNorm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker


def plot_pressure_overview_phases(df: pd.DataFrame, pdf_path: Path, orig_df: pd.DataFrame = None):
    """
    Plot pressure over time for each (borehole, section) series, colored by sensor_l5_dist bins,
    and with background shading for user-defined phases.
    """

    # -----------------------------
    # CONFIG (kept inside function)
    # -----------------------------

    # Fixed mapping: (min_inclusive, max_exclusive) -> color
    dist_color_bins: Dict[Tuple[float, float], str] = {
        (-100.0, 0.0):  "#1f77b4",   # blue, North chamber
        (0.2, 2.0): "#2ca02c",   # green
        (2.0, 5.0): "#ff7f0e",  # orange
        (5.0, 30.0): "#d62728",   # red
    }

    # Custom phases: (start_dt_str, end_dt_str, phase_name, color)
    # NOTE: colors are used as *very light* backgrounds via alpha below.
    phases: Sequence[Tuple[str, str, str, str]] = [
        ("2024-01-01 00:00:00", "2024-03-12 00:00:00", "WPT 24/02", "#1f77b4"),  # blue (WPT)
        ("2024-03-12 00:00:00", "2024-04-10 00:00:00", "Excavation", "#ff7f0e"),  # orange
        ("2024-04-10 00:00:00", "2024-06-01 00:00:00", "Relaxation", "#2ca02c"),  # green
        ("2024-06-01 00:00:00", "2024-09-11 00:00:00", "Stationary", "#7f7f7f"),  # gray
        ("2024-09-11 00:00:00", "2024-11-01 00:00:00", "WPT 24/09", "#1f77b4"),  # blue (WPT)
        ("2024-11-01 00:00:00", "2025-04-10 00:00:00", "Stationary", "#7f7f7f"),  # gray
        ("2025-03-25 00:00:00", "2025-06-01 00:00:00", "WPT 25/03", "#1f77b4"),  # blue (WPT)
        ("2025-06-01 00:00:00", "2025-08-06 00:00:00", "Stationary", "#7f7f7f"),  # blue (WPT)

    ]

    sensor_dist_col = "sensor_l5_dist"
    phase_alpha = 0.08

    # -----------------------------
    # DATA PREP
    # -----------------------------
    unit = df.attrs.get("units", {}).get("pressure", "")  # e.g. 'kPa'
    dfp = df.reset_index().copy()

    if "timestamp" not in dfp.columns:
        raise ValueError("Expected a 'timestamp' column or index named 'timestamp'.")

    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"])

    if orig_df is not None:
        orig_df = orig_df.copy()
        orig_df["timestamp"] = pd.to_datetime(orig_df["timestamp"])

    if sensor_dist_col not in dfp.columns:
        raise ValueError(f"Missing required column '{sensor_dist_col}' in df.")

    def color_for_dist(x: float) -> str:
        for (lo, hi), c in dist_color_bins.items():
            if lo <= x < hi:
                return c
        return "black"

    # -----------------------------
    # PLOT
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # Phase background shading + phase name at top
    for start_s, end_s, name, color in phases:
        start = pd.to_datetime(start_s)
        end = pd.to_datetime(end_s)

        ax.axvspan(start, end, color=color, alpha=phase_alpha, zorder=0)

        mid = start + (end - start) / 2
        ax.text(
            mid,
            0.985,
            name,
            transform=ax.get_xaxis_transform(),  # x=data coords, y=axes fraction
            ha="center",
            va="top",
            fontsize=9,
            color=color,
            alpha=0.9,
            zorder=3,
        )
    dist_vals = dfp[sensor_dist_col].dropna().unique()
    cmap = plt.get_cmap('cmr.wildfire')  # diverging (green- black- orange);
    norm = TwoSlopeNorm(vmin=dist_vals.min(), vcenter=0, vmax=dist_vals.max())

    # Series: one line per (borehole, section), colored by sensor_l5_dist bin
    for (bh, sec), grp in dfp.groupby(["borehole", "section"], sort=False):
        dist_val = float(grp[sensor_dist_col].mean())  # choose mean distance for the group
        #line_color = color_for_dist(dist_val)
        line_color = cmap(norm(dist_val))
        ax.plot(
            grp["timestamp"],
            grp["pressure"],
            color=line_color,
            linewidth=0.6,
            alpha=0.95,
            zorder=2,
        )

        # Optional original overlay (same color, lower alpha)
        if orig_df is not None:
            orig_grp = orig_df[(orig_df["borehole"] == bh) & (orig_df["section"] == sec)]
            if not orig_grp.empty:
                ax.plot(
                    orig_grp["timestamp"],
                    orig_grp["pressure"],
                    color=line_color,
                    linewidth=1.0,
                    alpha=0.35,
                    zorder=1.5,
                )

    # -----------------------------
    # AXES FORMATTING
    # -----------------------------
    ax.set_title("All pore pressure measurements, colored by L5 sensor distance [m]")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Pressure {unit}" if unit else "Pressure")
    ax.set_ylim(-200, 1000)
    ax.set_xlim(dfp.timestamp.min(), pd.to_datetime(phases[-1][1]))
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.format_xdata = mdates.DateFormatter("%Y-%m-%d %H:%M:%S")
    fig.autofmt_xdate()

    # Make room for legend on the right
    fig.subplots_adjust(right=0.80)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    # Option A: label along the side (default colorbar label)
    cbar.set_label("L5 sensor distance [m]", rotation=90, labelpad=12)

    # Option B: a title above the bar (often nicer if you also have in-bar text)
    # cbar.ax.set_title(quantity_name, pad=8)

    # --- Two vertical labels INSIDE the colorbar ---
    neg_label = "ZK5-1S"
    pos_label = "ZK5-1J"

    # Where is the diverging center inside the bar?
    # If you used TwoSlopeNorm(vcenter=...), this maps the center value to 0.5.
    center_y = 0.5

    # Place text in colorbar-axes coordinates (0..1)
    text_kw = dict(
        transform=cbar.ax.transAxes,
        rotation=90,  # vertical text
        ha="center",
        va="center",
        fontsize=10,
        color="black",
        # Optional: add a light box so it’s readable on any color
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5)
    )

    # Put each label centered within its half
    cbar.ax.text(0.5, center_y / 2, neg_label, **text_kw)  # middle of lower half
    cbar.ax.text(0.5, center_y + (1 - center_y) / 2, pos_label, **text_kw)  # middle of upper half
    # TODO:
    # L5 distance label
    # chamber labels
    # ax.text(
    #     x_pad,
    #     dist_val,
    #     "L5 sensor distance",
    #     va="center",
    #     ha="left",
    #     fontsize=8,
    # )


    # series_df = (
    #     dfp[["borehole", "section", sensor_dist_col]]
    #     .drop_duplicates()
    #     .reset_index(drop=True)
    # )


    # 2) create a small axis on the right for the scatter "legend"
    #    (tweak numbers if you want it wider/taller)
    # ax_scatter = fig.add_axes([0.82, 0.15, 0.17, 0.70])  # [left, bottom, width, height] in fig coords
    #
    # # 3) scatter points (y is just an index so they are stacked)
    # ys = list(range(len(series_rows)))
    # xs = [0.0 for r in series_rows]
    # cs = [cmap(norm(r[2])) for r in series_rows]  # SAME color as the time lines


    # # 4) labels to the right of each point
    # #    little x-offset as a fraction of data range
    # x_pad = 0.02
    # for bh, sec, dist_val in series_df.itertuples(index=False):
    #     cs = cmap(norm(dist_val))
    #     ax_scatter.scatter(0.0, dist_val, c=cs, s=30, linewidths=0.0)
    #
    #     ax_scatter.text(
    #         x_pad,
    #         dist_val,
    #         f"{bh}_{sec}",
    #         va="center",
    #         ha="left",
    #         fontsize=8,
    #     )
    #
    # # 5) format the scatter axis to look like a legend panel
    # ax_scatter.set_title(sensor_dist_col, fontsize=9)
    # ax_scatter.set_yticks([])  # hide y ticks
    # ax_scatter.tick_params(axis="y", length=0)
    # ax_scatter.grid(False)
    #
    # # keep a small x-axis so the depth/dist values are interpretable
    # ax_scatter.set_xlabel("")  # optional: leave blank to keep it clean
    # #ax_scatter.set_ylim(-1, len(series_rows))
    # #ax_scatter.set_xlim(x_min - x_pad, x_max + 6 * x_pad)
    #
    # # optional: make it visually lighter
    # for spine in ["top", "right", "left"]:
    #     ax_scatter.spines[spine].set_visible(False)
    #
    # # IMPORTANT: remove/avoid the old right-margin adjustments used for legends/colorbar.
    # # If you previously had fig.subplots_adjust(right=0.80), update it to give room for this panel:
    # fig.subplots_adjust(right=0.80)  # tune as needed for your layout

    # # Legend: distance bins
    # sorted_bins = sorted(dist_color_bins.items(), key=lambda kv: kv[0][0])
    # proxies = [Line2D([0], [0], color=c, linewidth=3) for (_rng, c) in sorted_bins]
    # labels = [
    #     f"{lo:g}–{hi:g}" if hi < 1e8 else f"{lo:g}+"
    #     for (lo, hi), _c in sorted_bins
    # ]
    # ax.legend(
    #     proxies,
    #     labels,
    #     title=sensor_dist_col,
    #     loc="upper left",
    #     bbox_to_anchor=(0.82, 1.0),
    #     frameon=False,
    # )

    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.93)

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()

def to_datetime(x):
    return pd.to_datetime(x, format="%y/%m/%d %H:%M:%S")

def plot_pressure_overview(df: pd.DataFrame, pdf_path: Path,
                           orig_df: pd.DataFrame = None,
                           events:pd.DataFrame = None,
                           sections=None,
                           xlims:Tuple[pd.Timestamp, pd.Timestamp]=None,
                           ylims:Tuple[float, float]=None):
    """
    Plot pressure over time for each borehole and section combination,
    with both legends to the right, weekly date ticks, and tight layout.
    """
    unit = df.attrs['units']['pressure']  # e.g. 'kPa'
    df = df.reset_index()
    #df = df.set_index('timestamp', drop=True)

    if sections is None:
        # full range present in df
        sections = list(df[['borehole', 'section']]
                        .drop_duplicates().itertuples(index=False, name=None))

    else:
        sections = [(f"L5-{bh}", int(sec)) for bh, sec in sections]

    # keep only requested (borehole, section) pairs
    pairs = pd.MultiIndex.from_tuples(sections, names=['borehole', 'section'])
    df_sub = df.set_index(['borehole', 'section']).loc[pairs].reset_index()


    # color per borehole
    boreholes = df['borehole'].unique()


    palette = sns.color_palette(n_colors=len(boreholes))
    color_map = dict([
        ('L5-49DL', '#c06800'),
        ('L5-50UL', '#6f22aa'),
        ('L5-22DR', '#ff0000'),
        ('L5-23UR', '#00c900'),
        ('L5-24DR', '#00d6cb'),
        ('L5-26R',  '#ea60a3'),
        ('L5-37R',  '#ed9600'),
        ('L5-37UR', '#1871bf'),

        ])

    # linestyle per chamber (section)
    iints = sorted(df['section'].unique())
    assert len(iints) <= 3, "Too many unique sections for line styles."
    fig, ax = plt.subplots(figsize=(12, 6))

    # 4) vertical lines for blast events
    side_map = dict(
        N=('dashed', 'ZK5-1S'),
        S=('dotted', 'ZK5-1J'),
    )
    for blast in events:
        ls, label = side_map[blast['side']]
        blast_time = to_datetime(blast['datetime'])
        ax.axvline(x=blast_time, color='grey',
                   linestyle=ls, label=label, alpha=0.5)
        # label text
        txt = f"{blast['side']}:{blast['face_stationing']}"

        # put text near the top of the axes at the same x
        # ax.text(
        #     blast_time, 0.98, txt,
        #     transform=ax.get_xaxis_transform(),   # x in data coords, y in axes fraction (0..1)
        #     xytext=(0.3, 0),  # shift right
        #     textcoords="offset points",
        #     rotation=90,
        #     va="top", ha="left",
        #     fontsize=8,
        #     color="grey",
        #     alpha=0.9
        # )

        x = mdates.date2num(blast_time)  # <-- key
        ax.annotate(
            txt,
            xy=(x, 0.98),
            xycoords=ax.get_xaxis_transform(),
            xytext=(3, 0),          # shift right
            textcoords="offset points",
            rotation=90,
            va="top", ha="left",
            fontsize=8,
            color="grey",
            alpha=0.9,
        )
    # 2) Plot series
    style_map = {
        0: (0, ()),  # solid
        1: (0, (10, 4)),  # long dashes
        2: (0, (2, 3)),  # sparse dots (widely spaced)
    }
    marker_map = [
        '^',
        's',
        'o'
    ]
    marker_step = len(df_sub['timestamp']) // (20 * len(boreholes) * len(iints))
    for (bh, iint), grp in df_sub.groupby(['borehole', 'section']):
        line_width = 0.8
        line_style = 'solid'  # solid by default
        if orig_df is not None:
            orig_grp = orig_df[(orig_df['borehole'] == bh) & (orig_df['section'] == iint)]
            ax.plot(
                orig_grp['timestamp'],
                orig_grp['pressure'],
                color=color_map[bh],
                linestyle='-',
                linewidth=0.7,
                alpha=0.5
            )
            line_width = 0.5
            line_style = (0, (2, 3))    # sparse dots (widely spaced)
        ax.plot(
            grp['timestamp'],
            grp['pressure'],
            color=color_map[bh],
            linestyle=line_style,
            linewidth=line_width,
            marker=marker_map[iint],  # different marker per “position”
            markersize=5,
            markerfacecolor=color_map[bh],
            markeredgewidth=0.0,
            markevery=marker_step
        )

    # 3) Axes formatting
    ax.set_title('Pressure Overview by Borehole & Chamber')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Pressure ({unit})')
    ylims = ylims if ylims is not None else (-200, 1000)
    ax.set_ylim(*ylims)
    if xlims:
        xlims = [to_datetime(x) for x in xlims]
        ax.set_xlim(*xlims)


    # weekly ticks on Mondays
    # mondays = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
    # ax.xaxis.set_major_locator(mondays)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # 1) use an AutoDateLocator so that when you zoom/pan it re‐computes tick spacing:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)

    # 2) use an AutoDateFormatter so the labels adapt (show hours:minutes when you zoom in):
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_formatter(formatter)

    # 3) ensure the status‐bar shows full date+time (not just days):
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    fig.autofmt_xdate()


    # shrink plot area to make room for legends
    fig.subplots_adjust(right=0.75)

    # 4) Borehole legend (colors)
    proxies_bh = [
        Line2D([0], [0], color=color_map[bh], linestyle='-', linewidth=2)
        for bh in boreholes
    ]
    legend1 = ax.legend(
        proxies_bh, boreholes,
        title='Borehole',
        loc='upper left',
        bbox_to_anchor=(0.85, 1),
        frameon=False
    )
    ax.add_artist(legend1)

    # 5) Chamber legend (line-styles)
    proxies_i = [
        Line2D(
            [0], [0],
            color='black',
            linestyle=style_map[i],
            linewidth=2,
            marker=marker_map[i],
            markersize=7,  # larger in legend for readability
            markerfacecolor='black',
            markeredgewidth=0.0
        )
        for i in iints
    ]
    ax.legend(
        proxies_i, [f'section={i}' for i in iints],
        title='Chamber',
        loc='upper left',
        bbox_to_anchor=(0.7, 1),
        frameon=False
    )
    fig.tight_layout(pad=0)  # pad=0 removes all padding

    # Option B: manual margins (fractions of figure size)
    fig.subplots_adjust(
        left=0.10,  # 2% in from left
        right=0.98,  # 2% in from right
        bottom=0.10,  # 2% in from bottom
        top=0.93  # 2% in from top
    )

    # 6) Save & show
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.show()


def plot_pressure_graphs(flat_df, epoch, events, work_dir):
    """
    Vytvoří a uloží jediný graf pro každý list v intervalu tmin až tmax do složky `script_dir`.
    Pokud jsou hodnoty `data_s` nebo `data_j` mimo interval [tmin, tmax], ignorují se.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        labels (list): Seznam listů, které mají být zpracovány.
        columns (list of list): Seznam sloupců, které mají být vykresleny.
        data_s (list): Seznam časů pro jižní orientaci.
        data_j (list): Seznam časů pro severní orientaci.
        orientace (list): Seznam orinput_dir / 'blast_events.xlsx'ientací ('S' nebo 'J') pro jednotlivé listy.
        tmin (float): Dolní hranice vykresleného intervalu.
        tmax (float): Horní hranice vykresleného intervalu.
        script_dir (str): Cesta k výstupní složce pro uložení grafů.
    """
    boreholes = flat_df.borehole.unique()
    assert len(boreholes) == 8, "Expected exactly two boreholes in the DataFrame."
    fig, axes = plt.subplots(4, 2, sharex='all', figsize=(20, 32))
    axes = axes.flatten()
    unit = flat_df.attrs['units']['pressure']  # e.g. 'kPa'

    palette = sns.color_palette("bright", n_colors=3)

    side_name = {'L': 'N', 'R': 'S'}
    for bh, ax  in zip(boreholes, axes):
        bh_df = flat_df[flat_df.borehole == bh]
        # plot, one line per location
        sns.lineplot(
            data=bh_df,
            x='time_days',
            y='pressure',
            hue='section',
            palette=palette,  # <— your custom palette
            ax=ax
        )

        # labels
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(f'Pressure ({unit})')

        # set y‑limits
        ax.set_ylim(-100, bh_df.pressure.max())
        ax.set_title(f'Pressure at {bh} borhole chambers.')

        # set major ticks every 1 day on x, and every 100 kPa on y
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1/24))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))

        # turn on grid for both axes
        ax.grid(which='major', axis='both', linestyle='--', linewidth=0.5)

        for blast in events:
            # plot vertical line for each event
            if blast['side'] == side_name[bh[-1]]:
                ls = 'dashed'
                label = "blast"
            else:
                ls = 'dotted'
                label = "blast opposite"
            ax.axvline(x=blast['linear_time'], color='red', linestyle=ls, label=label)
        ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    dict_leg = dict(zip(labels, handles))
    fig.legend(
        dict_leg.values(), dict_leg.keys(),
        title="All series (common legend)",
        loc='upper right',
        bbox_to_anchor=(1, 1),  # just outside the axes
        borderaxespad=0.  # no padding
    )
    fig.tight_layout()
    fig.savefig(work_dir / f"{epoch}.pdf")
    plt.show()
    #     plt.grid(True, which='both', linestyle='--', linewidth=0.1)
        #     plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        #     plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        #     plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        #     plt.xlim(tmin, tmax)  # Nastavení rozsahu vodorovné osy
        #
        #     # Přidání vertikálních čar pro odstřely pouze v rozsahu [tmin, tmax]
        #     if orient == 'S':
        #         for i, cas in enumerate(data_s):
        #             if tmin <= cas <= tmax:
        #                 plt.axvline(x=cas, color='red', linestyle='--', label='Odstřel' if i == 0 else "",
        #                             linewidth=0.5)
        #     elif orient == 'J':
        #         for i, cas in enumerate(data_j):
        #             if tmin <= cas <= tmax:
        #                 plt.axvline(x=cas, color='blue', linestyle='--', label='Odstřel' if i == 0 else "",
        #                             linewidth=0.5)
        #
        #     # Sestavení názvu souboru a uložení do work_dir
        #     graph_filename = os.path.join(work_dir, f'PRESSURE_{label}.pdf')
        #     plt.savefig(graph_filename, format='pdf')
        #     plt.close()
        #
        #     print(f'Hotový graf pro {label} v intervalu {tmin} až {tmax}, uložen do {graph_filename}')
        # else:
        #     print(f"Sloupec 'cas' chybí v datech pro {label}.")



