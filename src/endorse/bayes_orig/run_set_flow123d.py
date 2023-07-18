import os
import sys
import csv
import time
import ruamel.yaml as yaml
import numpy as np

import flow_wrapper
from measured_data import MeasuredData

from run_all import setup
from preprocess import preprocess

from surrDAMH.modules.raw_data import RawData
from surrDAMH.modules.analysis import Analysis


def just_run_flow123d(config_dict, measured_data, params_in, output_dir_in, solver_id):

    wrap = flow_wrapper.Wrapper(solver_id=solver_id, output_dir=output_dir_in, config_dict=config_dict)

    for idx, pars in enumerate(params_in):
        wrap.set_parameters(data_par=pars)
        t = time.time()
        res, obs_data = wrap.get_observations()
        print("Flow123d res: ", res)
        if res >= 0:
            print(obs_data)
            boreholes = config_dict["surrDAMH_parameters"]["observe_points"]
            measured_data.plot_comparison(obs_data, wrap.sim.sample_dir, boreholes)

        print("LEN:", len(obs_data))
        print("TIME:", time.time() - t)
        # if idx == 1:
        #     exit(0)


def get_best_accepted_params(config_dict_in, output_dir_in, count):
    no_parameters = len(config_dict_in['surrDAMH_parameters']['parameters'])

    conf_bayes_path = os.path.join(output_dir_in, "common_files", config_dict_in["surrDAMH_parameters"]["config_file"])
    with open(conf_bayes_path) as f:
        config_bayes = yaml.safe_load(f)
    observations = np.array(config_bayes["problem_parameters"]["observations"])
    basename = os.path.basename(conf_bayes_path)
    problem_name, fext = os.path.splitext(basename)
    output_dir_bayes = os.path.join(output_dir_in, 'saved_samples', problem_name)

    raw_data = RawData()
    raw_data.load(output_dir_bayes, no_parameters, len(observations))
    # type: 0-accepted, 1-prerejected, 2-rejected
    raw_data_filtered = raw_data.filter(types=[0, 2], stages=[0])

    analysis_pe = Analysis(config=config_bayes, raw_data=raw_data_filtered)
    fits, norms = analysis_pe.find_n_best_fits(observations, count=count, norm="L2")

    param_file = os.path.join(output_dir_in, "best_accepted_params.csv")
    params = []
    with open(param_file, 'w') as file:
        line = ','.join(analysis_pe.par_names)
        file.write(line + "\n")
        for sample in fits:
            params.append(sample.parameters())
            line = ','.join([str(s) for s in sample.parameters()])
            file.write(line + "\n")

    return params


def add_output_keys(config_dict):
    fname = config_dict["hm_params"]["in_file"]
    fname_output = fname + '_vtk'
    ftemplate = os.path.join(config_dict["common_files_dir"], fname + '_tmpl.yaml')
    ftemplate_output = os.path.join(config_dict["common_files_dir"], fname_output + '_tmpl.yaml')

    yaml_handler = yaml.YAML()
    with open(ftemplate, "r") as f:
        file_content = f.read()
    template = yaml_handler.load(file_content)

    flow_fields = [
       {"field": "conductivity", "interpolation": "P1_average"},
       "piezo_head_p0",
       "pressure_p0",
       "velocity_p0",
       "region_id"
    ]
    template["problem"]["flow_equation"]["flow_equation"]["output"]["fields"] = flow_fields

    mech_fields = [
        {"field": "displacement", "interpolation": "P1_average"},
        "stress",
        "displacement_divergence",
        "mean_stress",
        "von_mises_stress",
        "initial_stress",
        "region_id"
    ]
    template["problem"]["flow_equation"]["mechanics_equation"]["output"]["fields"] = mech_fields

    config_dict["hm_params"]["in_file"] = fname_output
    with open(ftemplate_output, 'w') as f:
        yaml_handler.dump(template, f)


if __name__ == "__main__":

    # default parameters
    output_dir = None
    csv_data = None
    n_best_params = 0
    sample_subdir = None
    solver_id = 0

    len_argv = len(sys.argv)
    assert len_argv > 2, "Specify output dir and parameters in csv file!"
    if len_argv > 1:
        output_dir = os.path.abspath(sys.argv[1])
    if len_argv > 2:
        file = sys.argv[2]
        if os.path.exists(file):
            csv_data = os.path.abspath(sys.argv[2])
        else:
            n_best_params = int(sys.argv[2])
    if len_argv > 3:
        if os.path.isabs(sys.argv[3]):
            sample_subdir = sys.argv[3]
        else:
            sample_subdir = os.path.join(output_dir, sys.argv[3])
    if len_argv > 4:
        solver_id = sys.argv[4]

    # setup paths and directories
    config_dict = setup(output_dir, can_overwrite=False, clean=False)
    if "vtk_output" in config_dict and config_dict["vtk_output"]:
        add_output_keys(config_dict)
    if sample_subdir is not None:
        config_dict["sample_subdir"] = sample_subdir

    # preprocess(config_dict)

    # prepare measured data as observations
    md = MeasuredData(config_dict)
    md.initialize()

    boreholes = config_dict["surrDAMH_parameters"]["observe_points"]
    times, values = md.generate_measured_samples(boreholes, [])

    if csv_data:
        print("Reading parameters from CSV: ", csv_data)
        with open(csv_data, newline='') as csvfile:
            parameters = list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
    else:
        print("Getting " + str(n_best_params) + " best parameters.")
        parameters = get_best_accepted_params(config_dict, output_dir, n_best_params)

    print(parameters)

    print(boreholes)
    # JUST RUN FLOW123D FOR TESTING
    just_run_flow123d(config_dict, md, parameters, output_dir, solver_id)
