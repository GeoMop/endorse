import os
import sys
import csv
import time
import ruamel.yaml as yaml
import numpy as np
import pandas as pd

import flow_wrapper
from measured_data import MeasuredData

import aux_functions
from run_all import setup
from preprocess import preprocess

from surrDAMH.modules.raw_data import RawData
from surrDAMH.modules.analysis import Analysis

n_samples_per_diff = 33


def collect_flow123d_results(sensitivity_dir):
    # next(os.walk(sensitivity_dir))[1] : walk is generator that gives [0]-root, [1]-dirnames, [2]-filenames
    solver_dirs = [os.path.join(sensitivity_dir,d) for d in next(os.walk(sensitivity_dir))[1]]
    no_observations = len(config_bayes["problem_parameters"]["observations"])
    observations = np.zeros((0,no_observations))

    for solver_dir in solver_dirs:
        solver_id = os.path.basename(solver_dir).split("_")[-1]
        sample_dirs = [os.path.join(solver_dir, d) for d in os.listdir(solver_dir)]
        output_file = os.path.join(sensitivity_dir, 'output_' + solver_id + '.csv')
        n_samples = len(sample_dirs)

        wrap = flow_wrapper.Wrapper(solver_id=solver_id, output_dir=solver_dir, config_dict=config_dict)
        observations_part = np.zeros((n_samples, no_observations))

        for sample_dir in sample_dirs:
            idx = int(os.path.basename(sample_dir).split("_")[-1])
            # print(solver_id, idx)
            wrap.sim.sample_counter = idx
            wrap.sim.sample_dir = sample_dir
            wrap.sim.sample_output_dir = os.path.join(sample_dir, "output_" + config_dict["hm_params"]["in_file"])
            try:
                observations_part[idx] = wrap.sim.collect_results(config_dict)
            except:
                observations_part[idx] = np.inf * np.ones((1,no_observations))

        observations = np.vstack((observations, observations_part))

        with open(output_file, 'w') as file:
            for idx in range(observations_part.shape[0]):
                line = str(idx) + ',' + ','.join([str(s) for s in observations_part[idx]])
                file.write(line + "\n")
    # with open(os.path.join(sensitivity_dir, 'output.csv'), 'w') as file:
    #     for idx in range(n_samples):
    #         line = str(idx) + ',' + ','.join([str(s) for s in observations[idx]])
    #         file.write(line + "\n")


def collect_data(sensitivity_dir):
    files = os.listdir(sensitivity_dir)
    param_files = sorted([f for f in files if "params_" in f and ".csv" in f])
    data_files = sorted([f for f in files if "output_" in f and ".csv" in f])
    assert len(param_files) == len(data_files)

    no_parameters = config_bayes["no_parameters"]
    no_observations = config_bayes["no_observations"]

    parameters = np.zeros((0, no_parameters))
    observations = np.zeros((0, no_observations))
    for i, (pf, df) in enumerate(zip(param_files, data_files)):
        print("Reading parameters from CSV: ", pf)
        p_samples = pd.read_csv(os.path.join(sensitivity_dir, pf), header=0)
        d_samples = pd.read_csv(os.path.join(sensitivity_dir, df))
        n_collected = len(d_samples)

        # sort
        # obs_data = obs_data[observations_part[:, 0].argsort()]
        # cut unfinished samples at the end (not all diffs computed)
        cut = n_collected - int(n_collected / n_samples_per_diff) * n_samples_per_diff
        obs_data = np.array(d_samples.iloc[:-cut,1:])
        n_collected = obs_data.shape[0]

        # get corresponding parameters
        params = np.array(p_samples.iloc[:n_collected, :])

        observations = np.vstack((observations, obs_data))
        parameters = np.vstack((parameters, params))

    print(observations.shape)
    print(parameters.shape)
    return parameters, observations


def compute_differences(parameters, observations):
    tidx = 10
    npar = config_bayes["no_parameters"]
    n_points = int(parameters.shape[0]/n_samples_per_diff)

    diff_close = np.zeros((n_points, npar))
    diff_far = np.zeros((n_points, npar))
    points = np.zeros((n_points, npar))

    for i in range(n_points):
        sub_p = parameters[i*n_samples_per_diff:(i+1)*n_samples_per_diff]
        sub_o = observations[i*n_samples_per_diff:(i+1)*n_samples_per_diff, tidx]
        # diff nominator
        diff_o_close = sub_o[1:npar+1] + sub_o[npar+1:2*npar+1] - 2*sub_o[0]
        diff_o_far = sub_o[2*npar+1:3*npar+1] + sub_o[3*npar+1:] - 2*sub_o[0]
        # diff denominator
        diff_p_close = np.diag(sub_p[1:npar+1,:]) - np.diag(sub_p[npar+1:2*npar+1,:])
        diff_p_far = np.diag(sub_p[2*npar+1:3*npar+1,:]) - np.diag(sub_p[3*npar+1:,:])

        points[i] = sub_p[0]
        diff_close[i] = diff_o_close / diff_p_close**2
        diff_far[i] = diff_o_far / diff_p_far**2

    plot_diff(points, diff_close, "diff_close")
    plot_diff(points, diff_far, "diff_far")


def plot_diff(points, diffs, name):
    npoints, npar = points.shape
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(npar, npar, sharex=False, sharey=False, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    trans = config_bayes["transformations"]
    for j in range(npar):
        axis = axes[0, j]
        # determine parameter name
        label = trans[j]["name"]
        axis.set_title(label, x=0.5, rotation=45, multialignment='center')

    for i in range(npar):
        for j in range(npar):
            axis = axes[i, j]
            if i==j:
                axis.plot(points[:,i], diffs[:,i], '.')
            elif i>j:
                axis.scatter(points[:,i], points[:,j], c=diffs[:,i], cmap='hsv')
            else:
                axis.scatter(points[:,i], points[:,j], c=diffs[:,j], cmap='hsv')

    fig.savefig(os.path.join(sensitivity_dir, name + ".pdf"), bbox_inches="tight")


def read_sensitivity_config(work_dir):
    # test if config exists, copy from rep_dir if necessary
    sens_config_file = os.path.join(work_dir, "config_sensitivity.yaml")

    if not os.path.exists(sens_config_file):
        raise Exception("Main configuration file 'config.yaml' not found in workdir.")

    # read config file
    with open(sens_config_file, "r") as f:
        sens_config_dict = yaml.safe_load(f)

    sens_config_dict["script_dir"] = os.path.dirname(os.path.abspath(__file__))
    sens_config_dict["rep_dir"] = os.path.abspath(os.path.join(sens_config_dict["script_dir"], "../../.."))
    return sens_config_dict


if __name__ == "__main__":

    # default parameters
    output_dir = None

    len_argv = len(sys.argv)
    assert len_argv > 1, "Specify output dir & number of processes & number of best fits"
    if len_argv > 1:
        output_dir = os.path.abspath(sys.argv[1])

    # setup paths and directories
    config_dict = setup(output_dir, can_overwrite=False, clean=False)
    conf_bayes_path = os.path.join(output_dir, "common_files", config_dict["surrDAMH_parameters"]["config_file"])
    with open(conf_bayes_path) as f:
        config_bayes = yaml.safe_load(f)

    sens_config_dict = read_sensitivity_config(output_dir)

    sensitivity_dir = os.path.join(output_dir, "sensitivity")
    # collect_flow123d_results(sensitivity_dir)
    par, obs = collect_data(sensitivity_dir)
    compute_differences(par, obs)
