from pathlib import Path
import sys
import csv

import arviz
import numpy as np

from . import read_idata_from_file


# script should be launched from the parent directory to the two directories containing the .idata files
# TODO: let user specify the paths to the two directories
DIRECTORY_WITH_FLOW = "dataset8_idata"
DIRECTORY_NO_FLOW = "dataset7_idata"

BOREHOLES = [
    "22DR",
    "23UR",
    "24DR",
    "26R",
    "37R",
    "37UR",
    "49DL",
    "50UL"
]

SECTIONS = [
    0,
    1,
    2
]

if __name__ == "__main__":
    parent = Path.cwd()
    borehole = sys.argv[1]

    assert borehole in BOREHOLES, f"Borehole {borehole} not recognized."

    section = sys.argv[2]

    assert int(section) in SECTIONS, f"Section {section} not recognized."
    section = int(section)

    borehole_full = f"{borehole}_{section}"
    borehole_ext = ".idata"

    with_flow_files = [
        file for file in (parent / DIRECTORY_WITH_FLOW).resolve().iterdir()
        if file.is_file() and borehole_full in file.name and borehole_ext in file.name
    ]

    assert np.all([
        "2024" in with_flow_files[0].name or "2024" in with_flow_files[1].name,
        "2025" in with_flow_files[0].name or "2025" in with_flow_files[1].name,
    ]), "Could not find data for both years"

    # idata a - 2024-xx
    # if reverse order, swap
    if "2024" in with_flow_files[1].name:
        with_flow_files = [with_flow_files[1], with_flow_files[0]]

    no_flow_files = [
        file for file in (parent / DIRECTORY_NO_FLOW).resolve().iterdir()
        if file.is_file() and borehole_full in file.name and borehole_ext in file.name
    ]

    assert np.all([
        "2024" in no_flow_files[0].name or "2024" in no_flow_files[1].name,
        "2025" in no_flow_files[0].name or "2025" in no_flow_files[1].name,
    ]), "Could not find data for both years"

    # idata a - 2024-xx
    # if reverse order, swap
    if "2024" in no_flow_files[1].name:
        no_flow_files = [no_flow_files[1], no_flow_files[0]]

    print("Found all necessary files.")

    # collect all needed data
    # format for pressures: borehole name + section, p_far mean 2024, p_far std 2024, p_far mean 2025, p_far std 2025
    # format for flows: borehole name + section, flow mean 2024 unknown flow, flow std 2024 unknown flow, flow mean 2025 unknown flow, flow std 2025 unknown flow
    # flow_mean 2024 known flow, flow std 2024 known flow, flow mean 2025 known flow, flow std 2025 known flow

    pressure_header = [
        "borehole",
        "section",
        "p_far_mean_2024",
        "p_far_std_2024",
        "p_far_mean_2025",
        "p_far_std_2025"
    ]

    flow_header = [
        "borehole",
        "section",
        "flow_unknown_mean_2024",
        "flow_unknown_std_2024",
        "flow_unknown_mean_2025",
        "flow_unknown_std_2025",
        "flow_known_mean_2024",
        "flow_known_std_2024",
        "flow_known_mean_2025",
        "flow_known_std_2025"
    ]

    # start of lines - borehole name and section
    pressure_data = [borehole, section]
    flow_data = [borehole, section]

    # pressure should not have duplicate values from both sets
    is_known_flow = [False, False, True, True]
    file_order = no_flow_files + with_flow_files

    print("Processing files...")

    for known, file in zip(is_known_flow, file_order):
        times = []
        pressures = []
        flows = []

        # read idata, extract summary, unreference idata
        idata = read_idata_from_file(str(file))
        posterior_summary = arviz.summary(idata)
        output_summary = arviz.summary(idata, group="posterior_predictive")
        del idata

        if not known:
            pressure_data.append(posterior_summary.loc["p_far", ["mean", "sd"]].values)

        flow_data.append(output_summary.loc["obs_0", ["mean", "sd"]].values)

    print("Writing to summary files...")

    # write pressure data
    pressure_summary_file = parent / "pressure_summary.csv"
    with open(pressure_summary_file, "w", newline="") as f:
        sniffer = csv.Sniffer()
        has_header = sniffer.has_header(f.read(2048))
        f.seek(0)        
        writer = csv.writer(f)
        if not has_header:
            writer.writerow(pressure_header)
        writer.writerow(pressure_data)

    # write flow data
    flow_summary_file = parent / "flow_summary.csv"
    with open(flow_summary_file, "w", newline="") as f:
        sniffer = csv.Sniffer()
        has_header = sniffer.has_header(f.read(2048))
        f.seek(0)
        writer = csv.writer(f)
        if not has_header:
            writer.writerow(flow_header)
        writer.writerow(flow_data)

    print("Done.")