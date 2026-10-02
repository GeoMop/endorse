"""Prepare and run the local hydro-mechanical Flow123d model."""

import argparse
import shutil
import sys
from pathlib import Path
import traceback
import math
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import input_data
from endorse import common
from mesh.create_mesh import borehole_fractures, geometry_points, make_mesh

DEFAULT_REPLACEMENTS = {
    "rock_conductivity": "1e-13",
    # Flow123d storativity is pressure-head storage [m^-1]:
    # S = rho_w*g*(beta_d + n*beta_w).  For the model's E=60 GPa,
    # nu=0.25 rock, beta_d=3*(1-2*nu)/E; with n=0.007 and
    # beta_w=4.6e-10 Pa^-1 (liquid water near 20 degC), S=2.77e-7 m^-1.
    "rock_storativity": "2.8e-7",
    "packer_conductivity": "1e-14",
    # Packers use the same mechanical material as rock in this model, so
    # their uncalibrated hydraulic storage uses the same estimate.
    "packer_storativity": "2.8e-7",
    "water_conductivity": "1e-5",
    # Water-filled chamber storage: rho_w*g*beta_w = 4.51e-6 m^-1.
    "water_storativity": "4.5e-6",
    "fracture_conductivity": "1e-6",
    # The lower-dimensional fracture is water-filled; its cross-section
    # scales the stored volume, while its material storage is that of water.
    "fracture_storativity": "4.5e-6",
    "fracture_cross_section": "1e-3",
    "rock_young": "60e9",
    "rock_poisson": "0.25",
    "packer_young": "30e9",
    # A conceptual model gives E_eff 6 - 30 GPa for 0.5 long packer
    # and 15 - 90GPa for 1m packer, taking into account lot of uncertainties in the packer design.
    "packer_poisson": "0.3",
    # taking into account 0.5 poisson ration of the rubber, but just in thin layer and
    # steel reinforced.
    "fracture_young": "1e7",
    "fracture_poisson": "0.25",
}


def machine_config(config_path: Path | None, flow_executable: str) -> common.dotdict:
    """Return Flow123d machine configuration."""
    if config_path is not None and config_path.exists():
        return common.load_config(config_path).machine_config
    return common.dotdict({"flow_executable": [flow_executable]})


def prepare_mesh_file(work_dir: Path) -> None:
    """Generate the mesh and make it available to the Flow123d work directory."""
    cfg = common.config.load_config(input_data.mesh_cfg_yaml)
    make_mesh(cfg, work_dir, split_pocket=False)
    expected_mesh = work_dir / "wpt_section.msh"
    generated_mesh = work_dir / "wpt_section.msh2"
    shutil.copy2(generated_mesh, expected_mesh)


def run_model(
    cfg: common.dotdict,
    work_dir: Path,
    replacements: dict[str, str] | None = None,
) -> common.FlowOutput:
    """Substitute YAML template placeholders and run Flow123d."""
    yaml_replacements = DEFAULT_REPLACEMENTS.copy()
    if replacements is not None:
        yaml_replacements.update(replacements)
    work_dir.mkdir(parents=True, exist_ok=True)
    prepare_mesh_file(work_dir)
    with common.workdir(work_dir):
        return common.call_flow(cfg, input_data.hm_sim_tmpl_yaml, yaml_replacements)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=APP_DIR / "input_data" / "config.yaml",
        help="Optional config file with machine_config.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=APP_DIR / "runs" / "workdir",
        help="Directory for generated Flow123d inputs and outputs.",
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

    # now length is just the difference of correct section end and start
    section_start, section_end = section_coordinates(borehole, section)
    section_length = section_end - section_start

    # compute water volume
    return math.pi * section_length * borehole_radius**2


def section_coordinates(borehole: input_data.Borehole, section: input_data.Section) -> tuple[float, float]:
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

    return section_starts[section.value], section_ends[section.value]

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
    # convert flow from mm^3/s to m^3/s (SI units)
    filtered[flow_column] = filtered[flow_column] * 1e-9

    # offset the data so that it starts on time 0
    # precompute time=0
    time_zero = pd.to_datetime(filtered.iloc[0]["Date"], format="%Y-%m-%d %H:%M:%S")
    # compute time offsets
    filtered["Time"] = filtered.apply(lambda row: (pd.to_datetime(row["Date"], format="%Y-%m-%d %H:%M:%S") - time_zero).total_seconds(), axis=1)
    # remove redundant column
    filtered.drop(columns="Date", inplace=True)
    # swap column order to match expected format in template
    filtered = filtered[filtered.columns[::-1]]

    # adjust values to water source density
    # by dividing flow by volume
    volume = compute_water_volume(borehole, section)
    assert volume != -1, "Unable to calculate volume"
    filtered[flow_column] = filtered[flow_column] / volume

    return filtered.values.tolist()

# TODO: adjust this to work with any WPT, not just 2025 ones
def get_initial_pressure(borehole: input_data.Borehole, section: input_data.Section) -> float:
    """ Read initial pressure from data file. Uses events.yaml (field "start" in borehole) to determine datetime to read.

    Arguments:
        borehole -- Borehole to read pressure of.
        section -- Section to read pressure of.

    Returns:
        Value of initial pressure in pascals.
    """
    # get start datetime
    events = common.config.load_config(input_data.events)["water_pressure_tests"]
    target_event = {}
    for event in events:
        # TODO: better way to identify year
        if event["borehole"] == borehole.value and event["section"] == section.value and event["start"][:2] == "25":
            target_event = event
            break

    assert target_event != {}, f"Could not find target event for borehole {borehole}, section {section}"

    # load data from appropriate .csv column
    pressure_column = borehole.value + "_" + str(section.value) + "_pressure"
    pressure_data = pd.read_csv(input_data.data_2025, usecols=["Date", pressure_column])

    # transform datetime to timedelta from target event's datetime
    target_datetime = pd.to_datetime(target_event["start"], format="%y/%m/%d %H:%M:%S")
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

    # time point where to set flow=0 onward
    # 1 minute after last specified flow rate
    flow_series_end = flow_series[-1][0] + 1 * 60

    # end of entire simulation, for now hardcoded
    # could probably be included in some config
    simulation_end = 3600 * 24 * 7

    # calculate observe point
    # used point is in the middle of the section on the axis
    _, _, section_start_mesh, section_end_mesh = geometry_points(mesh_cfg)
    section_middle_mesh = (section_start_mesh + section_end_mesh) / 2

    # initial pressure
    # will be used for all regions, including outer pressure
    # event.yaml's origin is the start of pressure drop, start is a bit before that
    # TODO: vefify that all starts are before pressure rise, aka at borehole's steady state
    initial_pressure = get_initial_pressure(borehole, section)

    fracture_config = common.config.load_config(input_data.bh_cfg_yaml)["boreholes"]
    fractures = []
    for bh in fracture_config:
        if bh["name"] == borehole:
            fractures = bh["fractures"]
            break
    
    # filter fractures to only ones interesecting the section
    # they have to be in ascending distance from section start
    # which boreholes.yaml already does
    section_start, section_end = section_coordinates(borehole, section)
    fractures_interesecting = []
    for fracture in fractures:
        if fracture["position"] <= section_end and fracture["position"] >= section_start:
            fractures_interesecting.append(fracture)

    # fracture config doesn't contain mesh-coordinate centers
    # borehole_fractures() returns mesh coordinates
    # but in the same order, so that can be used to append data
    fracture_centers = [fracture[1].tolist() for fracture in borehole_fractures(mesh_cfg)]
    for idx, _ in enumerate(fractures_interesecting):
        fractures_interesecting[idx]["mesh_center"] = fracture_centers[idx]

    # template file always expects 3 fractures
    # => fill out the fractures array to always have 3 elements
    fractures_interesecting += [{
        "mesh_center": [0, 0, 0],
        "width": 1
    }] * (3 - len(fractures_interesecting))

    # compile all replacements
    replacements = {
        "flow_series": flow_series,
        "init_pressure": initial_pressure,
        "observe_point": section_middle_mesh.tolist(),
        # figure out a way to pass this without duplicating code
        "fracture_center_0": fractures_interesecting[0]["mesh_center"],
        "fracture_center_1": fractures_interesecting[1]["mesh_center"],
        "fracture_center_2": fractures_interesecting[2]["mesh_center"],
        "fracture_radius_0": fractures_interesecting[0]["width"],
        "fracture_radius_1": fractures_interesecting[1]["width"],
        "fracture_radius_2": fractures_interesecting[2]["width"],
        "end_time": simulation_end,
        "flow_series_end": flow_series_end,
        "fracture_conductivity": 1e-14
    }

    print(replacements)

    run_model(cfg, args.work_dir, replacements=replacements)
