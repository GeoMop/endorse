"""
TODO:
- put file names and column specifications to the input YAML
- use a config attrs class
- use python logging instead of print
- optimize initial reading of XLSX, rather work only with CSV files
-
"""

from typing import Iterable, Union
import re
import numpy as np
import pandas as pd


from endorse import common

from .decorators import common_report
from chodby_inv import get_logger
logger = get_logger(__name__)

from .press_plot import plot_pressure_overview, plot_pressure_graphs
from chodby_inv import input_data as inputs
input_dir = inputs.input_dir
work_dir = inputs.work_dir


# funkce v programu
# zjištění názvů vrtů a jejich orientace S/J
def bh_config():
    """
    Načtení souboru Konfigurace vrtu, kde je seznam listu a jejich orientace
    """
    return common.config.load_config(inputs.bh_cfg_yaml).boreholes


@common_report
@common.memoize
def read_piezo_file(input_file: common.File):
    """
    Načte všechny listy ze souboru piezo.xlsx, nahradí prázdné hodnoty NaN
    a volitelně uloží výsledek do Excelu.

    Args:
        input_file (str or Path): Cesta k vstupnímu souboru.
        output_file (str or Path, optional): Cesta k výstupnímu souboru. Výchozí None.
        save_to_excel (bool, optional): Pokud True, uloží výstup do Excelu. Výchozí True.

    Returns:
        dict: Slovník obsahující DataFrame pro každý list.
    """
    excel_data = pd.read_excel(input_file.path, sheet_name=None)

    # Iterace přes všechny listy a náhrada prázdných buněk NaN
    for sheet_name, df in excel_data.items():
        df = df.replace(r'^\s*$', np.nan, regex=True)
        excel_data[sheet_name] = df

    return excel_data


def dict_merge_safe(dict1, dict2):
    """
    Merges two dictionaries, dict1 and dict2, recursively.
    If a key exists in both dictionaries, the value from dict2 overwrites the value from dict1.
    """
    merged = dict1.copy()  # Start with keys and values of dict1
    for key, value in dict2.items():
        if key in merged:
            # If the key exists in both dictionaries and both values are dictionaries, merge them recursively
            if merged[key] != value:
                logger.warning(f"Key '{key}' exists in both dictionaries, but values differs:\n"
                               f"dict1: {merged[key]}\n"
                               f"dict2: {value}\n"
                               f"Using value from dict1.")
                value = merged[key]
        merged[key] = value
    return merged


def df_concat(frames: Iterable[pd.DataFrame], **concat_kwargs) -> pd.DataFrame:
    """
    Concatenate DataFrames while merging attrs.

    Parameters
    ----------
    frames : Iterable[pd.DataFrame] or pd.DataFrame
        List (or other iterable) of DataFrames to concat, or a single DataFrame.
    **concat_kwargs
        All additional keyword args passed to pd.concat (e.g. ignore_index=True).

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame, with each column's .attrs
        being the dict-merge of the source cols' attrs (later frames overwrite).
    """
    frames = list(frames)   # allow repeted iteration
    result = pd.concat(frames, **concat_kwargs)

    # collect and merge attrs for each column
    for df in frames:
        result.attrs = dict_merge_safe(result.attrs, df.attrs)
        for col in df.columns:
            result[col].attrs = dict_merge_safe(result[col].attrs, df[col].attrs)
    return result




def chamber_col_name(borehole, depth):
    return f"{borehole} {depth} m [kPa]"


borehole_chambers = {
    'L5-49DL': ['5,68', '7,67','8,64'],
    'L5-50UL': ['7,72', '9,72', '12,58'],
    'L5-37R': ['3,69', '6,67', '10,62'],
    'L5-26R': ['3,74', '7,19', '10,10'],
    'L5-23UR': ['3,68', '7,18', '10,12'],
    'L5-24DR': ['2,79', '6,20', '9,15'],
    'L5-22DR': ['8,65', '11,19', '14,66'],
    'L5-37UR': ['17,67', '20,75', '22,74']
}

@common_report
def log_large_differences(final_data_frames, v_diff, a_diff, rozrazka_dframes, output_excel_file, output_csv_file):
    """
    Analyzuje rozdíly v tlaku mezi řádky v `final_data_frames` a zapisuje pouze významné rozdíly do Excelu a CSV.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        v_diff (float): Minimální hodnota rozdílu pro zápis.
        a_diff (float): Minimální hodnota rozdílu průměrů pro zápis.
        rozrazka_file (str or Path): Cesta k souboru s daty rozrážek.
        output_excel_file (str or Path): Cesta k výstupnímu Excel souboru.
        output_csv_file (str or Path): Cesta k výstupnímu CSV souboru.

    Returns:
        dict: Aktualizovaný `final_data_frames` s přidanými hodnotami.

    TODO: simplify, split to smaller pieces
    """

    # Ověření existence listu "vystup" v datech
    if "vystup" not in final_data_frames:
        raise ValueError("❌ Chyba: List 'vystup' neexistuje v `final_data_frames`.")

    df = final_data_frames["vystup"]

    # Načtení časů rozrážek

    times_s, times_j = rozrazka_dframes
    all_rozrazka_times = times_s + times_j

    # Data pro zápis
    output_minus = []
    output_plus = []
    last_shot_time = {}

    chambers = [
        (bh, i_chmbr, depth)
        for bh, chamber_depths in borehole_chambers.items()
        for i_chmbr, depth in enumerate(chamber_depths)]

    for i_row in range(4, len(df) - 5):
        for bh_name, i_chmbr, depth in chambers:
            column = chamber_col_name(bh_name, depth)
            if column not in df.columns:
                continue

            # TODO: following split into functions
            # avoid nested conditions
            j = i_row
            while j >= 0 and pd.isna(df.loc[j, column]):
                j -= 1

            if j >= 0 and pd.notna(df.loc[j, column]) and pd.notna(df.loc[i_row + 1, column]):
                value_diff = abs(df.loc[i_row + 1, column] - df.loc[j, column])

                avg_recent = df[column].iloc[i_row + 2:i_row + 6].dropna().mean()
                avg_previous = df[column].iloc[max(0, j - 4):j].dropna().mean()

                if not pd.isna(avg_recent) and not pd.isna(avg_previous):
                    avg_diff = abs(avg_recent - avg_previous)

                    if value_diff > v_diff and avg_diff > a_diff:
                        current_time = df.loc[i_row + 1, 'cas']
                        col_last_shot_time = last_shot_time.get(column, None)
                        at_blast = any(abs(current_time - t) <= 0.02 for t in all_rozrazka_times)
                        close_after_blast = False
                        if col_last_shot_time:
                            close_after_blast = col_last_shot_time < current_time <= col_last_shot_time + 0.05

                        if col_last_shot_time is None and at_blast:
                            change_type = "střílení"
                            last_shot_time[column] = current_time
                        elif close_after_blast:
                            change_type = "reakce_na_střílení"
                        else:
                            change_type = "nevysvětleno"
                            last_shot_time[column] = None

                        output_data = {
                            'sim_time': current_time,
                            'Year': df.loc[i_row + 1, 'Year'],
                            'Month': df.loc[i_row + 1, 'Month'],
                            'Day': df.loc[i_row + 1, 'Day'],
                            'Hour': df.loc[i_row + 1, 'Hour'],
                            'Minute': df.loc[i_row + 1, 'Minute'],
                            'Seconds': df.loc[i_row + 1, 'Seconds'],
                            'Borehole': bh_name,
                            'Chamber': i_chmbr,
                            'depth in borehole': depth,
                            'pressure': df.loc[i_row + 1, column],
                            'pressure_window_start': df.loc[j, column],
                            'pressure_window_end': df.loc[i_row + 1, column],
                            'pressure_diff': value_diff,
                            'pressure_avgPrůměr před': avg_previous,
                            'Průměr po': avg_recent,
                            'Rozdíl průměrů': avg_diff,
                            'Druh změny': change_type
                        }
                        if current_time < 0:
                            output_minus.append(output_data)
                        else:
                            output_plus.append(output_data)

    # Vytvoření DataFrame pouze s relevantními daty
    df_output = pd.DataFrame(output_minus + output_plus)

    # Pokud jsou nějaká data k uložení, zapíšeme je
    if not df_output.empty:
        with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
            df_output.to_excel(writer, sheet_name='Filtered_Output', index=False)

        df_output.to_csv(output_csv_file, index=False, sep=';')

        print(f"✅ Filtrované výsledky byly uloženy:\n   📂 Excel: {output_excel_file}\n   📂 CSV: {output_csv_file}")
    else:
        print("⚠️ Nebyly nalezeny žádné významné rozdíly k uložení.")

    return final_data_frames




# def export_all_pressure_readings(final_data_frames, output_excel_file, output_csv_file):
#     """
#     Exportuje všechny tlakové hodnoty z listu 'vystup' ve formátu:
#     čas, Borehole, Chamber, depth in borehole, tlak
#     """
#
#     if "vystup" not in final_data_frames:
#         raise ValueError("List 'vystup' neexistuje v datech.")
#
#     df = final_data_frames["vystup"]
#
#     exclude_columns = {'Year', 'Month', 'Day', 'Hour', 'Minute', 'Seconds'}
#     pressure_columns = [col for col in df.columns if col not in exclude_columns and col != 'cas']
#
#     rows = []
#     for _, row in df.iterrows():
#         cas = row['cas']
#         for col in pressure_columns:
#             tlak = row[col]
#             if pd.notna(tlak):
#                 found = False
#                 for bh_name, depths in borehole_chambers.items():
#                     for i_chmbr, depth in enumerate(depths):
#                         depth_pattern = f"{bh_name} {depth} m [kPa]"
#                         if col.strip() == depth_pattern:
#                             rows.append({
#                                 'sim_time': cas,
#                                 "Borehole": bh_name,
#                                 "Chamber": i_chmbr,
#                                 "depth in borehole": depth,
#                                 "tlak": tlak
#                             })
#                             found = True
#                             break
#                     if found:
#                         break
#                 if not found:
#                     print(f"⚠️ Čidlo nerozpoznáno ve struktuře: {col}")
#
#     df_long = pd.DataFrame(rows)
#
#     # Uložení do souborů
#     with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
#         df_long.to_excel(writer, sheet_name="tlaky", index=False)
#
#     df_long.to_csv(output_csv_file, sep=";", index=False)
#
#     print(f"📤 Tlaková data exportována do:\n   📁 {output_excel_file}\n   📁 {output_csv_file}")






def channels_rename(cols, boreholes, re_col, quantity, unit):
    # Auxiliary fn for `df_column_map`.
    # Renames column matchin `re_col` with `(\d+)` group giving channel number.
    # the borehole label and *packer section* is determined from channel.
    mapping = {}
    for col in cols:
        m = re.match(re_col, col)
        if not m:
            continue
        ch = int(m.group(1))
        grp = (ch - 1) // 4  # 0..3 → which borehole
        i_interval = (ch - 1) % 4  # 0,1,2,3 within group
        if i_interval < 3 and grp < len(boreholes):
            bh = boreholes[grp]
            mapping[col] = (bh, 3 - i_interval, quantity, "[dgts]")
    return mapping

def df_column_map(df, boreholes):
    """
    Rename columns according to:
      1. Triples of 'tlak * [kPa]' following a borehole column 'L5-*' → '{borehole}/{k}', k = 2,1,0
      2. 'Sensor Reading...Channel N' → '{borehole}/{k} digits' for N%4 in (1,2,3)
      3. 'Sensor Temp...Channel N'    → '{borehole}/{k} temp'   for N%4 in (1,2,3)
      All other columns are left as‐is.
    """
    cols = list(df.columns)
    mapping = {}

    # 1) Pressure triples under each borehole marker
    current_bh = None
    i_press_col = 0
    for col in cols:
        # pressure columns
        if re.match(r"^tlak.*\[kPa\]$", col):
            i_bh = i_press_col // 3
            bh = boreholes[i_bh]
            i_interval = i_press_col % 3
            mapping[col] = (bh, 3 - i_interval, "pressure", "[kPa]")
            i_press_col += 1

    map = channels_rename(df, boreholes,
                    r"^Sensor Reading.*Channel\s*(\d+)", "digits", "[dgts]")
    mapping.update(map)
    map = channels_rename(df, boreholes,
                    r"^Sensor Temp.*Channel\s*(\d+)", "temp", "[C]")
    mapping.update(map)
    # apply and return
    return mapping



def make_timestamp(df: pd.DataFrame) -> pd.Series:
    """
    Assert that the columns Year, Month, Day, Hour, Minute correspond to
    the `datum_col` strings (format "%m/%d/%Y %H:%M"), ignoring Seconds.
    Returns a pd.Series of full timestamps (including Seconds).
    Raises AssertionError listing all mismatched rows.
    """

    # 1) build timestamps from the six columns
    ts = pd.to_datetime({
        "year":   df["Year"],
        "month":  df["Month"],
        "day":    df["Day"],
        "hour":   df["Hour"],
        "minute": df["Minute"],
        "second": df["Seconds"],
    })
    return ts


def extract_bholes(df, bholes):
    """
    Extracts the standard borehole DataFrame from the given DataFrame.
    """
    mapping = df_column_map(df, bholes)

    # assert form missing measurements
    ids = df['Array #'].to_numpy().astype(int)
    diff = ids[1:] - ids[:-1]
    diff_wrong = (diff != 1) & (diff != -3554)
    if np.any(diff_wrong):
        missing_reads = np.arange(len(df)-1)[diff_wrong]
        logger.warning(f'Missing measurement IDs: {missing_reads}\n')
    # get time stamps
    ts = make_timestamp(df)

    # 1) build a long-form list of rows
    rows = []
    units = {
        'battery_voltage': '[V]',
        'air_pressure': '[kPa]',
        'station_temperature':'[C]'
    }   # remember unit per q
    for col, (bh, i_int, q, unit) in mapping.items():
        units[q] = unit
        # pick out datum + this single value column
        tmp = pd.DataFrame()
       # can not use nan as that is invalid for pandas index

        tmp['timestamp'] = ts
        tmp['borehole'] = bh
        tmp['section']    = i_int - 1
        tmp['q']        = q
        tmp['value']    = df[col]

        tmp['battery_voltage'] = df['Battery Voltage(v)']
        tmp['station_temperature'] = df['Internal Temp(ｰC)']
        orig_bar_col = 'barometr [kPa]'
        if orig_bar_col in df.columns:
            tmp['air_pressure'] = df[orig_bar_col]
        else:
            tmp['air_pressure'] = np.full(len(df), -1)

        rows.append(tmp)

    long = pd.concat(rows, ignore_index=True)
    idx_cols = long.columns.difference(['q', 'value'], sort=False).tolist()
    #long_unique = long.drop_duplicates(subset=idx_cols + ['q'])

    # 2) pivot to wide: one column for each q
    wide = (
        long
        .pivot(index=idx_cols, columns='q', values='value')
        .reset_index()
        # if you want, rename the columns‐axis away:
        .rename_axis(columns=None)
    )

    # 4) attach units as attrs
    wide.attrs['units'] = units
    return wide


def full_flat_df(bh_cfg, piezo_file):
    # Hlavni program - načtení vstupních dat
    # Otevře soubor konfigurace_vrtu a zjistí názvy vrtů a jejich orientaci S/J
    #labels, orientace = read_konfigurace(input_dir / 'konfigurace_vrtu.xlsx')
    bh_labels = [bh.name for bh in bh_cfg]
    # Hlavni program - úprava dat
    # Načtení dat do paměti a zároveň uložení do "piezo3.xlsx"
    data_frames = read_piezo_file(
        common.File(piezo_file),
        save_to_excel=False
    )
    jz_bholes = [ 'L5-22DR', 'L5-23UR', 'L5-24DR', 'L5-26R']
    df_bholes = [extract_bholes(data_frames['JZ'], jz_bholes)]
    for bh in bh_labels:
        if bh not in data_frames:
            continue
        df_bholes.append(extract_bholes(data_frames[bh], [bh]))

    flat_df = df_concat(df_bholes)

    # Check for fuplicities in index
    # 1) Identify any exact duplicate triples
    dup_mask = flat_df.duplicated(
        subset=['timestamp', 'borehole', 'section'],
        keep=False
    )
    dupes = flat_df.loc[dup_mask].sort_values(
        ['timestamp', 'borehole', 'section']
    )
    if len(dupes) > 0:
        logger.warning(f"❌ Found {len(dupes)} rows with duplicate (timestamp, borehole, section):")
        #logger.info(dupes.to_string())
    #flat_df.set_index(['timestamp', 'borehole', 'section'], inplace=True, drop=False)
    return flat_df

def to_datetime(series):
    """
    Convert a Series of strings to datetime objects.
    """
    return pd.to_datetime(series, format='%y/%m/%d %H:%M:%S')

def linear_time(times, epoch_cfg):
    """
    Convert a Series of datetime objects to linear time in days from the origin.
    """
    dt_origin = to_datetime(epoch_cfg.origin)
    times_s = to_datetime(pd.Series(times))
    return (times_s - dt_origin) / pd.Timedelta(days=1)

def get_epoch(df, epoch_cfg):
    """
    Get the datetime range given by the epoch_cfg.
    Return slice of original DataFrame with added column 'time_days'
    for linear time in days from the epoch's origin.
    """
    ts_origin = epoch_cfg.get('origin', epoch_cfg.start)
    dt_min    = to_datetime(epoch_cfg.start)
    dt_max   = to_datetime(epoch_cfg.end)
    dt_origin = to_datetime(ts_origin)

    # filter between min and max (inclusive)
    mask = (df['timestamp'] >= dt_min) & (df['timestamp'] <= dt_max)
    df_slice = df.loc[mask].copy()

    # compute elapsed time in days from origin
    df_slice['time_days'] = (df_slice['timestamp'] - dt_origin) / pd.Timedelta(days=1)
    return df_slice


def denoise_pressure(pressure_series):
    """
    Apply two step nenoising procedure to a single time series.
    1. remove single point outliers, based on relative curvature
    2. Mask for wrong points according to high absolute curvature.
       Replace wrong points by convolve *good* points in neighbourhood with
       the Gaussian kernel.
    Current setting only removes the largest outliers.
    """
    pressure = pressure_series.values
    pressure[pressure > 2000] = -50.0

    # detect single value outliers
    a_diff = np.maximum(np.abs(pressure[2:] - pressure[:-2]), 1.0)
    mean = (pressure[2:] + pressure[:-2]) / 2
    curvature = 2 * (mean - pressure[1:-1])
    mask = np.abs(curvature) / a_diff  > 1.5
    pressure[1:-1][mask] = mean[mask]

    mask1 = np.abs(curvature)  > 40

    pressure_ok = np.where(mask1, 0.0, pressure[1:-1])
    kernel = np.arange(-20,20,1)
    kernel = np.exp(-0.5 * (kernel / 5) ** 2)
    smooth = np.convolve(pressure_ok, kernel, mode='same')
    smooth /= np.convolve(~mask1, kernel, mode='same')
    pressure[1:-1][mask1] = smooth[mask1]

    #mean = (pressure[2:] + pressure[:-2]) / 2
    #curvature = 2 * (mean - pressure[1:-1])
    #mask2 = np.abs(curvature)  > 40
    #print('n_curve points: ', np.sum(mask1), np.sum(mask2))

    return pd.Series(data=pressure, index=pressure_series.index)


def detect_jumps_irregular(
    s: pd.Series,
    T: pd.Timedelta,
    factor: float = 3.0
) -> pd.Series:
    """
    Build the step‑function J(t) for an irregular time series s by
    detecting jumps where
      |Δs[i]| > factor  &  small derivative mean over +/- T/2 (excluding i).
    but that is just total variance of s[i] over +/- T/2 (excluding i).
    so the condition is:
    diff s > factor * threshold and mean TV(s) < threshold

    Returns
    -------
    J : pd.Series
        The cumulative step function at each timestamp of s.
    """
    # ensure sorted datetime index
    s = s.sort_index()
    # 1) first differences
    diffs     = s.diff().fillna(0.0)
    abs_diffs = diffs.abs()
    dt = s.index.to_series().diff().dt.total_seconds().bfill()
    abs_time_diffs =  abs_diffs / dt

    # 2) rolling sums and counts over a centered window of width T
    #    .sum() and .count() include the i-th point
    roll = abs_diffs.rolling(window=T, center=True)
    T_seconds = T.total_seconds()
    mean_TV  = (roll.sum() - abs_diffs) / T_seconds

    # 3) local average excluding self:
    #    (total_sum - |diff_i|) / (count - 1)
    # #    beware division by zero when count <= 1
    # numer = window_sum - abs_diffs
    # denom = (window_count - 1).clip(lower=1)
    # local_avg = numer.div(denom)

    # 4) where we don’t have a valid local neighborhood, fall back to global median
    # global_med = abs_diffs.median()
    # local_avg = local_avg.fillna(global_med)

    threshold = mean_TV.quantile(q=0.99)
    # 5) detect jumps by comparing abs_diffs to factor * local_avg
    is_jump = (abs_time_diffs > factor * threshold) & (mean_TV < threshold)

    # 6) build the step function (cumulative sum of actual signed diffs at jumps)
    jumps = diffs.where(is_jump, 0.0)
    J     = jumps.cumsum()

    return J


def smooth_preserve_jumps(s: pd.Series,
                          T: pd.Timedelta = pd.Timedelta('120min'),
                          jump_factor: float = 5.0
                          ) -> pd.Series:
    """
    Subtracts step function, applies Gaussian smoothing over window T,
    then adds steps back.

    Parameters
    ----------
    s : pd.Series
        Irregular‐time series with DatetimeIndex.
    T : pd.Timedelta
        Window width for Gaussian smoothing.
    jump_factor : float
        Detect jumps as |Δ| > jump_factor * median(|Δ|).

    Returns
    -------
    pd.Series
        Smoothed series with jumps preserved.
    """
    # ensure sorted
    s = s.sort_index()

    # 1) build step function
    J = detect_jumps_irregular(s, T, jump_factor)

    # 2) detrend
    det = s - J

    smooth_det = det.ewm(
        halflife=T/2,
        times=det.index,
        adjust=True
    ).mean()

    # 4) add steps back
    return smooth_det + J


def filter_noise(full_df):
    # 1) make sure data are sorted by time within each series
    df = full_df.sort_values(['borehole', 'section', 'timestamp'])

    # 2) make timestamp the index (so groupby‐transform sees a DatetimeIndex)
    df = df.set_index('timestamp')

    smooth_fn = lambda series: (
        #smooth_preserve_jumps(
            denoise_pressure(series)#,
         #   T=pd.Timedelta('120min'), jump_factor=2.0)
    )

    # 2) apply the filter per (borehole, section) group
    df['pressure'] = (
        df
        .groupby(['borehole', 'section'])['pressure']
        .transform(smooth_fn)
    )
    df.reset_index(inplace=True)
    df.sort_values(['timestamp', 'borehole'], inplace=True)
    return df


def denoised_df():
    """
    Returns processed piezo measurement.
    main columns:
    - timestamp
    - borehole
    - section
    - pressure

    suplementary columns:
    - battery_voltage
    - air_pressure
    - station_temperature

    original data columns, pressure is in fact computed from these in original xlsx
    - digits
    - temp
    """
    # Basic test resulting to separated data tables and plots
    bh_cfg = bh_config()

    full_df = full_flat_df(bh_cfg, inputs.piezo_measurement_file)
    denoised_df = filter_noise(full_df)

    logger.info("Full pressure table head:\n{full_df.head(n=10).to_string()}")
    return denoised_df


def excavation_epoch_df():
    """
    Returns the excavation epoch DataFrame.
    """
    epoch = "excavation"
    process_cfg = common.config.load_config(inputs.piezo_filter_yaml)

    epoch_cfg = process_cfg[epoch]
    epoch_df = get_epoch(denoised_df(), epoch_cfg)

    return epoch_df

if __name__ == '__main__':
    bh_cfg = bh_config()
    full_df = full_flat_df(bh_cfg, inputs.piezo_measurement_file)

    denoised = denoised_df()
    plot_pressure_overview(denoised, work_dir / "overview_plot.pdf", orig_df = full_df)

    epoch = "excavation"
    process_cfg = common.config.load_config(inputs.piezo_filter_yaml)
    events_cfg = common.config.load_config(inputs.events_yaml)

    epoch_cfg = process_cfg[epoch]
    epoch_df = get_epoch(denoised, epoch_cfg)

    epoch_blasts = [ blast.copy() for blast in events_cfg.blasts]
    for blast in epoch_blasts:
        blast['linear_time'] = linear_time([blast['datetime']], epoch_cfg)[0]
    plot_pressure_graphs(epoch_df, epoch, epoch_blasts, work_dir)
