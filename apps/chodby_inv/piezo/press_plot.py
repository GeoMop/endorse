from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker


def plot_pressure_overview(df: pd.DataFrame, pdf_path: Path, orig_df: pd.DataFrame = None):
    """
       Plot pressure over time for each borehole and i_int combination,
       with both legends to the right, weekly date ticks, and tight layout.
       """
    unit = df.attrs['units']['pressure']  # e.g. 'kPa'
    df = df.reset_index()
    #df = df.set_index('timestamp', drop=True)

    # color per borehole
    boreholes = df['borehole'].unique()
    palette = sns.color_palette(n_colors=len(boreholes))
    color_map = dict(zip(boreholes, palette))

    # linestyle per chamber (i_int)
    iints = sorted(df['i_int'].unique())
    styles = ['-', '--', ':', '-.']
    style_map = {i: styles[idx % len(styles)] for idx, i in enumerate(iints)}

    # 2) Plot series
    fig, ax = plt.subplots(figsize=(12, 6))
    for (bh, iint), grp in df.groupby(['borehole', 'i_int']):
        ax.plot(
            grp['timestamp'],
            grp['pressure'],
            color=color_map[bh],
            linestyle=style_map[iint],
            linewidth=0.8
        )
        if orig_df is not None:
            orig_grp = orig_df[(orig_df['borehole'] == bh) & (orig_df['i_int'] == iint)]
            ax.plot(
                orig_grp['timestamp'],
                orig_grp['pressure'],
                color=color_map[bh],
                linestyle='-',
                linewidth=1,
                alpha=0.5
            )

    # 3) Axes formatting
    ax.set_title('Pressure Overview by Borehole & Chamber')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Pressure ({unit})')
    ax.set_ylim(-200, 1000)

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

    #sns.despine()

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
        Line2D([0], [0], color='black', linestyle=style_map[i], linewidth=2)
        for i in iints
    ]
    ax.legend(
        proxies_i, [f'i_int={i}' for i in iints],
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
            hue='i_int',
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

