import os
import sys
import csv
import time
import yaml
import numpy as np

import flow_wrapper
from measured_data import MeasuredData

from run_all import setup
from preprocess import preprocess

from surrDAMH.modules.raw_data import RawData
from surrDAMH.modules.analysis import Analysis


def just_run_flow123d(measured_data, params_in, output_dir_in, boreholes_in):

    wrap = flow_wrapper.Wrapper(solver_id=0, output_dir=output_dir_in)

    for idx, pars in enumerate(params_in):
        wrap.set_parameters(data_par=pars)
        t = time.time()
        res, obs_data = wrap.get_observations()
        print("Flow123d res: ", res)
        if res >= 0:
            print(obs_data)
            measured_data.plot_comparison(obs_data, wrap.sim.sample_dir, boreholes_in)

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
        for sample in fits:
            params.append(sample.parameters())
            line = ','.join([str(s) for s in sample.parameters()])
            file.write(line + "\n")

    return params


if __name__ == "__main__":

    # default parameters
    output_dir = None
    csv_data = None
    n_best_params = 0

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

    # setup paths and directories
    config_dict = setup(output_dir, can_overwrite=False, clean=False)

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
    just_run_flow123d(md, parameters, output_dir, boreholes)
