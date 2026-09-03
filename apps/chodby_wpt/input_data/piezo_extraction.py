"""
Extraction of data from piezo_*_.xlsx and wpt_*_on_multipacker.xlsx. 
"""
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np

from input_data import Borehole, Section
import input_data

## File constants
# File folder path is defined in input_data's __init__
PIEZO_PATH = input_data.input_dir / "piezo_2025_09_24.xlsx"
WPT_PATH = input_data.input_dir / "wpt_2025_04_with_flux_on_multipacker.xlsx"
DATA_FILE_2024 = input_data.input_dir / "wpt_2024.csv"
DATA_FILE_2025 = input_data.input_dir / "wpt_2025.csv"

## Piezo constants
# Relevant sheets
DATA_SHEETS = {
    Borehole.L5_50UL: "L5-50UL",
    Borehole.L5_49DL: "L5-49DL",
    Borehole.L5_37UR: "L5-37UR",
    Borehole.L5_37R : "L5-37R",
    Borehole.L5_26R : "JZ",
    Borehole.L5_24DR: "JZ",
    Borehole.L5_23UR: "JZ",
    Borehole.L5_22DR: "JZ",
}
# Relevant columns
# Columns are ordered from deepest to shallowest section
DATA_COLUMNS = {
    Borehole.L5_50UL: "T,U,V",
    Borehole.L5_49DL: "T,U,V",
    Borehole.L5_37UR: "T,U,V",
    Borehole.L5_37R : "T,U,V",
    Borehole.L5_26R : "BF,BG,BH",
    Borehole.L5_24DR: "BA,BB,BC",
    Borehole.L5_23UR: "AW,AX,AY",
    Borehole.L5_22DR: "AS,AT,AU",
}

# Datetime columns where full datetime is located
DATETIME_COLUMNS = {
    Borehole.L5_50UL: "S",
    Borehole.L5_49DL: "S",
    Borehole.L5_37UR: "S",
    Borehole.L5_37R : "S",
    Borehole.L5_26R : "AQ",
    Borehole.L5_24DR: "AQ",
    Borehole.L5_23UR: "AQ",
    Borehole.L5_22DR: "AQ",
}

# Start datetimes, separated for 2024 and 2025
# should start ~2 days before first WPT, but data only starts 5 hours before in 2024
DATETIME_START_2024 = datetime(2024, 3, 6, 10, 0, 0)
DATETIME_START_2025 = datetime(2025, 3, 25, 10, 0, 0)

# Time interval to capture, assuming the same for both
CAPTURE_INTERVAL = timedelta(days = 2 * 30)

## WPT Multipacker constants
FLOW_RATE_SHEET = "data (2)"
BOREHOLE_NAME_COLUMNS = "A,B"
FLOW_RATE_COLUMN = "H"
TIME_START_COLUMN = "J"
TIME_OFFSET_COLUMN = "F"
# Flow rate is in units mm^3/min, needs to be mm^3/s
# same goes for time offset
FLOW_RATE_NORMALIZATION_FACTOR = 60

OUTPUT_TIME_COLUMN = "Time"


def find_row_interval(
        path: Path,
        borehole: Borehole,
        dt_start: datetime,
        dt_end: datetime) -> tuple[int, int]:
    """Returns the row info for an interval closest to specified start and end datetime. 

    Arguments:
        sheet -- Sheet containing the data to analyze.
        dt_start -- Start of interval to match.
        dt_end -- End of interval to match.

    Returns:
        Start row number and number of rows in interval.
    """

    # read dates from file
    sheet = DATA_SHEETS[borehole]
    df = pd.read_excel(path, sheet, usecols=DATETIME_COLUMNS[borehole], names=["datetime"])

    # sort by distance to start datetime and get index for closest
    start_sorted_df = np.abs(df['datetime'] - dt_start)
    start_index = start_sorted_df.argmin()

    # sort by distance to end datetime and get index for closest
    end_sorted_df = np.abs(df['datetime'] - dt_end)
    end_index = end_sorted_df.argmin()

    # number of lines includes both edge values
    linecount = end_index - start_index + 1

    return start_index, linecount

def get_multipacker_flowrate_series(
        path: Path,
        sheet: str,
        borehole: Borehole,
        section: Section
    ) -> pd.DataFrame:
    """Get timeseries of flow rates for corresponding borehole from specified sheet and file.

    Arguments:
        path -- Path to file with flowrates.
        sheet -- Sheet name with flowrates.
        borehole -- Borehole to find flowrates for.
        section -- Borehole section to find florates for.

    Returns:
        Dataframe with time series.
    """

    # read name columns and drop empty rows
    name_rows = pd.read_excel(
        path,
        sheet,
        usecols=','.join([BOREHOLE_NAME_COLUMNS, TIME_START_COLUMN]),
        names=["Borehole", "Section", OUTPUT_TIME_COLUMN]).dropna(how="all")
    name_indexes = name_rows.index.to_list()

    # filter down to only relevant ones
    # there can be multiple
    relevant_rows = name_rows[
        (name_rows["Borehole"] == borehole.value)
        & (name_rows["Section"] == section.value)]

    #print(relevant_rows)

    # some of these will be "empty", aka no time series
    # only one is always correct
    correct_row = -1
    for row in relevant_rows.index.to_list():
        if row + 1 not in name_indexes:
            correct_row = row
            break
    assert correct_row != -1, "Could not find starting row"

    # figure out interval of time series
    # start is inclusive, end is exclusive
    interval_start = correct_row
    interval_end = name_indexes[name_indexes.index(interval_start) + 1]
    linecount = interval_end - interval_start - 1
    time_start = relevant_rows[OUTPUT_TIME_COLUMN].loc[correct_row]

    # index is 0-indexed, needs to be 1-indexed
    # -> skip one more row
    flow_rate_df = pd.read_excel(
        path,
        sheet,
        usecols=",".join([TIME_OFFSET_COLUMN, FLOW_RATE_COLUMN]),
        names=["Offset", "Flowrate"],
        skiprows=interval_start + 1,
        nrows=linecount)

    #print(interval_start + 1, linecount)

    # flowrate and time offset both in wrong units (mm^3/min and min respectively)
    # -> conversion to SI units
    flow_rate_df["Flowrate"] = flow_rate_df["Flowrate"] * FLOW_RATE_NORMALIZATION_FACTOR
    flow_rate_df["Offset"] = flow_rate_df["Offset"] * FLOW_RATE_NORMALIZATION_FACTOR

    # construct final dataframe with absolute time values instead of offsets
    # timedelta value is now in SI units (seconds)
    flow_rate_df[OUTPUT_TIME_COLUMN] = flow_rate_df.apply(lambda row: time_start + timedelta(seconds=int(row["Offset"])), axis=1)
    flow_rate_df.drop("Offset", axis=1, inplace=True)

    return flow_rate_df




if __name__ == "__main__":
    # boreholes to cover all sheets
    # all SW boreholes (26R, 24DR, 23UR, 22DR) share the same sheet, so only one is needed
    BOREHOLES = [
        Borehole.L5_50UL,
        Borehole.L5_49DL,
        Borehole.L5_37UR,
        Borehole.L5_37R,
        Borehole.L5_26R
    ]

    # collect all borehole time points
    dataframes = []
    intervals = {}

    for bh in BOREHOLES:
        START_INDEX, LINECOUNT = find_row_interval(
            PIEZO_PATH,
            bh,
            DATETIME_START_2025,
            DATETIME_START_2025 + CAPTURE_INTERVAL
        )

        intervals[bh] = [START_INDEX, LINECOUNT]

        time_rows = pd.read_excel(
            PIEZO_PATH,
            DATA_SHEETS[bh],
            usecols=DATETIME_COLUMNS[bh],
            names=[OUTPUT_TIME_COLUMN],
            skiprows=START_INDEX,
            nrows=LINECOUNT
        )

        print(START_INDEX, LINECOUNT)
        print(time_rows)

        dataframes.append(time_rows)

    # fill remaining boreholes for SW
    intervals[Borehole.L5_24DR] = intervals[Borehole.L5_26R]
    intervals[Borehole.L5_23UR] = intervals[Borehole.L5_26R]
    intervals[Borehole.L5_22DR] = intervals[Borehole.L5_26R]

    # unified time series has all unique time points across all time series
    unified_time_series = pd.concat(dataframes, ignore_index=True, sort=True) \
        .drop_duplicates() \
        .set_index(OUTPUT_TIME_COLUMN)

    # iterate across all boreholes and all sections
    # appending data rows to the final dataframe at the correct times
    for bh in Borehole:
        for sec in Section:
            # every borehole + section combination has two unique columns
            column_name_base = bh.value + "_" + str(sec.value)
            column_name_pressure = column_name_base + "_pressure"
            column_name_flow = column_name_base + "_flow"

            # part 1: add piezo pressure data to dataframe
            # hacky way to get correct column instead of reading all 3 and filtering after
            section_column = DATA_COLUMNS[bh].split(",")[sec.value]
            pressure_data = pd.read_excel(
                PIEZO_PATH,
                DATA_SHEETS[bh],
                usecols=",".join([DATETIME_COLUMNS[bh], section_column]),
                names=[OUTPUT_TIME_COLUMN, column_name_pressure],
                skiprows=intervals[bh][0],
                nrows=intervals[bh][1]
            ).set_index(OUTPUT_TIME_COLUMN)

            # join on time column
            unified_time_series = unified_time_series.join(pressure_data, how="outer")

            # part 2: add multipacker flow rate data to dataframe
            flowrate_data = get_multipacker_flowrate_series(WPT_PATH, FLOW_RATE_SHEET, bh, sec) \
                    .rename(columns={"Flowrate": column_name_flow}) \
                    .set_index(OUTPUT_TIME_COLUMN)

            unified_time_series.info(verbose=True)
            print(unified_time_series)

            unified_time_series = unified_time_series.join(flowrate_data, how="outer", )

    cols = unified_time_series.columns.to_list()

    # fill empty rows of flow columns with 0s
    flow_cols = filter(lambda col: col.endswith("_flow"), cols)
    flow_fill_dict = {col: 0 for col in flow_cols}
    unified_time_series.fillna(value=flow_fill_dict, inplace=True)

    # interpolate empty rows of pressure columns
    #pressure_cols = cols.endswith("_pressure")
    unified_time_series.interpolate(inplace=True, limit_direction="forward")

    # save result to csv
    unified_time_series.to_csv(DATA_FILE_2025, index=True, index_label="Date")
