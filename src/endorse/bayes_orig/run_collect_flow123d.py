import os
import sys
import csv
import time
import yaml
import numpy as np

from measured_data import MeasuredData
from flow123d_simulation import generate_time_axis


def collect_flow123d(measured_data, output_dir, config_dict):

    times = np.array(generate_time_axis(config_dict))

    with open(os.path.join(output_dir, "flow_observe.yaml"), "r") as f:
        loaded_yaml = yaml.load(f, yaml.CSafeLoader)
        points = loaded_yaml['points']
        point_names = [p["name"] for p in points]
        data = loaded_yaml['data']
        values = np.array([d["pressure_p0"] for d in data])#.transpose()
        obs_times = np.array([d["time"] for d in data]).transpose().flatten()

        # check that observe data are computed at all times of defined time axis
        all_times_computed = np.alltrue(np.isin(times, obs_times))
        if not all_times_computed:
            raise Exception("Observe data not computed at all times as defined by input!")
        # skip the times not specified in input
        t_indices = np.isin(obs_times, times).nonzero()
        values = values[t_indices].transpose().flatten()

        measured_data.plot_comparison(values, output_dir, point_names)


if __name__ == "__main__":

    # RUN THE MCMC SIMULATION
    # default parameters
    output_dir = None
    input_yaml_file = None

    len_argv = len(sys.argv)
    assert len_argv > 2, "Specify input yaml file and output dir!"
    if len_argv > 1:
        input_yaml_file = sys.argv[1]
    if len_argv > 2:
        output_dir = sys.argv[2]

    with open(input_yaml_file, "r") as f:
        input_dict = yaml.load(f, Loader=yaml.BaseLoader)
        # input_dict = yaml.load(f, yaml.CSafeLoader)

    config_dict = {
        "work_dir": output_dir,
        "script_dir": os.path.dirname(os.path.abspath(__file__)),
        "end_time": input_dict["problem"]["flow_equation"]["time"]["end_time"],
        "output_times": input_dict["problem"]["flow_equation"]["flow_equation"]["output"]["times"]
    }

    # prepare measured data as observations
    md = MeasuredData(config_dict)
    md.initialize()

    # md.plot_all_data()
    # md.plot_interp_data()

    collect_flow123d(md, output_dir, config_dict)
