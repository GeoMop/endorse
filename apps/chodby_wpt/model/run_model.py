"""Prepare and run the local hydro-mechanical Flow123d model."""

import argparse
import shutil
import sys
from pathlib import Path
import yaml
import traceback
import math
import traceback
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import input_data
from endorse import common
from mesh.create_mesh import geometry_points, borehole_fractures

module_dir = Path(__file__).resolve().parent
work_dir = input_data.work_dir


DEFAULT_REPLACEMENTS = {
    "rock_conductivity": "1e-13",
    "rock_storativity": "1",
    "packer_conductivity": "1e-13",
    "packer_storativity": "1",
    "water_conductivity": "1e-5",
    "watyer_storativity": "1",
    "fracture_conductivity": "1e-6",
    "fracture_storativity": "1",
    "fracture_cross_section": "1e-3",
    "rock_young": "60e9",
    "rock_poisson": "0.25",
    "fracture_young": "1e7",
    "fracture_poisson": "0.25",
}


def machine_config(config_path: Path | None, flow_executable: str) -> common.dotdict:
    """Return Flow123d machine configuration."""
    if config_path is not None and config_path.exists():
        return common.load_config(config_path).machine_config
    return common.dotdict({"flow_executable": [flow_executable]})


def prepare_mesh_file() -> None:
    """Make the mesh filename expected by the YAML template available."""
    expected_mesh = work_dir / "wpt_section.msh"
    generated_mesh = work_dir / "wpt_section.msh2"
    if not expected_mesh.exists() and generated_mesh.exists():
        shutil.copy2(generated_mesh, expected_mesh)


def run_model(cfg: common.dotdict, replacements: dict[str, str] | None = None) -> common.FlowOutput:
    """Substitute YAML template placeholders and run Flow123d."""
    yaml_replacements = DEFAULT_REPLACEMENTS.copy()
    if replacements is not None:
        yaml_replacements.update(replacements)

    prepare_mesh_file()
    with common.workdir(work_dir):
        return common.call_flow(cfg, input_data.hm_sim_tmpl_yaml, yaml_replacements)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=module_dir / "config.yaml",
        help="Optional config file with machine_config.",
    )
    parser.add_argument(
        "--flow-executable",
        default="flow123d",
        help="Flow123d executable used when --config is not present.",
    )
    return parser.parse_args()


def compute_water_volume(borehole: input_data.Borehole, section: input_data.Section) -> float:
    """Water volume calculation for a specified borehole and section.

    Arguments:
        borehole -- Borehole to calculate volume for.
        section -- Section to calculate volume for.

    Returns:
        Volume in SI units (m^3).
    """
    # try to get borehole radius
    config = common.config.load_config(input_data.mesh_cfg_yaml)
    borehole_radius = float(config["geometry"]["borehole_radius"])

    # try to get borehole length
    config = common.config.load_config(input_data.bh_cfg_yaml)["boreholes"]
    bh_data = {}
    for bh in config:
        if bh["name"] == borehole:
            bh_data = bh
            break

    assert bh_data != {}, "Unable to find borehole data"

    packer_width = bh_data["packer_width"]
    # figure out starting depths for all sections by adding half of packer's width
    section_starts = [sec + packer_width / 2 for sec in bh_data["packer_centers"]]
    # figure out ending depths for all sections
    # section ends 1x packer length before the next starts
    # last section ends at the depth of the borehole
    section_ends = [sec - packer_width for sec in section_starts[1:]] + [bh_data["length"]]
    assert len(section_starts) == len(section_ends), "Number of section starts and ends doesn't match"
    # now length is just the difference of correct section end and start
    borehole_length = section_ends[section.value] - section_starts[section.value]

    # compute water volume
    return math.pi * borehole_length * borehole_radius**2


def get_flow_time_series(borehole: input_data.Borehole, section: input_data.Section) -> list:
    """Reads time series for a specific borehole and section.
    Return format is dict with keys being time and values being flow values.  

    Arguments:
        borehole -- _description_
        section -- _description_

    Returns:
        List of lists, where inner list is two element with first element being time and second water density.
    """

    # relevant columns in CSV file
    flow_column = borehole.value + "_" + str(section.value) + "_flow"
    #print(flow_column)

    # 2025 data only has 1 WPT -> only one series of nonzero values per column
    data = pd.read_csv(input_data.data_2025, usecols=["Date", flow_column])
    # filter only nonzero values in flow column
    filtered = data[data[flow_column] > 0]

    # offset the data so that it starts on time 0
    # precompute time=0
    time_zero = pd.to_datetime(filtered.iloc[0]["Date"], format="%Y-%m-%d %H:%M:%S")
    # compute time offsets
    filtered["Time"] = filtered.apply(lambda row: (pd.to_datetime(row["Date"], format="%Y-%m-%d %H:%M:%S") - time_zero).total_seconds(), axis=1)
    # remove redundant column
    filtered.drop(columns="Date", inplace=True)
    # swap column order
    filtered = filtered[filtered.columns[::-1]]

    # adjust values to water density
    # by dividing flow by volume
    volume = compute_water_volume(borehole, section)
    assert volume != -1, "Unable to calculate volume"
    # volume is in m^3, convert to mm^3 to match units of flow (?)
    volume = volume * 1e6
    filtered[flow_column] = filtered[flow_column] / volume

    return filtered.values.tolist()

# TODO: adjust this to work with any WPT, not just 2025 ones
def get_initial_pressure(borehole: input_data.Borehole, section: input_data.Section) -> float:
    # get start datetime
    events = common.config.load_config(input_data.events)["water_pressure_tests"]
    target_event = {}
    for event in events:
        # probably should use better way to identify year
        if event["borehole"] == borehole.value and event["section"] == section.value and event["start"][:2] == "25":
            target_event = event
            break

    assert target_event != {}, f"Could not find target event for borehole {borehole}, section {section}"

    # load data from appropriate .csv column
    pressure_column = flow_column = borehole.value + "_" + str(section.value) + "_pressure"
    pressure_data = pd.read_csv(input_data.data_2025, usecols=["Date", pressure_column])

    # transform datetime to distance from target event's datetime
    target_datetime = pd.to_datetime(event["start"], format="%y/%m/%d %H:%M:%S")
    pressure_data["Time"] = pressure_data.apply(lambda row: abs(pd.to_datetime(row["Date"], format="%Y-%m-%d %H:%M:%S") - target_datetime).total_seconds(), axis=1)
    target_idx = pressure_data["Time"].argmin()
    target_pressure = pressure_data.iloc[target_idx][pressure_column]

    return target_pressure

if __name__ == "__main__":
    args = parse_args()
    cfg = machine_config(args.config, args.flow_executable)

    # read borehole and section from bh_cfg_yaml
    mesh_cfg = common.config.load_config(input_data.mesh_cfg_yaml)
    bh_cfg = mesh_cfg["borehole_section"]
    borehole = bh_cfg["borehole"]
    section = bh_cfg["section"]

    # parse string values to enums
    try:
        borehole = input_data.Borehole(borehole)
        section = input_data.Section(int(section))
    except Exception:
        print(f"Unable to parse loaded string borehole and section: {traceback.print_exc()}")
        sys.exit(1)

    flow_series = get_flow_time_series(borehole, section)

    # calculate observe point
    # used point is in the middle of the section on the axis
    _, _, section_start, section_end = geometry_points(mesh_cfg)
    section_middle = (section_start + section_end) / 2

    # initial pressure
    # will be used for all regions, including outer pressure
    # event.yaml's origin is the start of pressure drop, start is a bit before that
    # TODO: vefify that all starts are before pressure rise, aka at borehole's steady state
    initial_pressure = get_initial_pressure(borehole, section)

    # fracture center
    fractures = borehole_fractures(mesh_cfg)
    fracture_centers = [fracture[1].tolist() for fracture in fractures]
    # fractures should be in same order as in config
    fracture_config = common.config.load_config(input_data.bh_cfg_yaml)["boreholes"]
    bh_data = {}
    for bh in fracture_config:
        if bh["name"] == borehole:
            bh_data = bh
            break
    fracture_radii = [fracture["width"] for fracture in bh_data["fractures"]]
    print(fracture_centers, fracture_radii)

    # compile all replacements
    replacements = {
        "flow_series": flow_series,
        "init_pressure": initial_pressure,
        "observe_point": section_middle.tolist(),
        # figure out a way to pass this without duplicating code
        "fracture_center_0": fracture_centers[0],
        "fracture_center_1": fracture_centers[1],
        "fracture_center_2": fracture_centers[2],
        "fracture_radius_0": fracture_radii[0],
        "fracture_radius_1": fracture_radii[1],
        "fracture_radius_2": fracture_radii[2],
    }

    print(replacements)

    run_model(cfg, replacements=replacements)
