import os.path
import random
import shutil

import ruamel.yaml as yaml

from endorse.sa import sample, analyze
from endorse.bayes_orig import aux_functions
from endorse.bayes_orig import run_all as bayes_run_all
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import pytest
import multiprocessing
import math
import time
import csv


def calculate_and_format_ci(value, ci_width):
    # Calculate confidence interval as a percentage of the value
    lower_bound = value - 0.5 * ci_width
    upper_bound = value + 0.5 * ci_width

    # Format the result as a string
    result_str = f"{value:.4f} +/- {ci_width:.4f} ({100 * ci_width / value:.2f}%) [{lower_bound:.4f}, {upper_bound:.4f}]"

    return result_str


def print_sensitivity_results(problem, sobol_indices):
    # Print first-order Sobol indices with confidence intervals
    print("First-order Sobol indices with confidence intervals (95%):")
    for i, name in enumerate(problem['names']):
        S1_value = sobol_indices['S1'][i]
        S1_conf_width = sobol_indices['S1_conf'][i]

        result_str = calculate_and_format_ci(S1_value, S1_conf_width)
        print(f"{name}: {result_str}")

    # Print total-order Sobol indices with confidence intervals
    print("\nTotal-order Sobol indices with confidence intervals (95%):")
    for i, name in enumerate(problem['names']):
        ST_value = sobol_indices['ST'][i]
        ST_conf_width = sobol_indices['ST_conf'][i]

        result_str = calculate_and_format_ci(ST_value, ST_conf_width)
        print(f"{name}: {result_str}")

    # Print second-order Sobol indices with confidence intervals
    print("\nSecond-order Sobol indices:")
    for i, name1 in enumerate(problem['names']):
        for j, name2 in enumerate(problem['names']):
            if i < j:
                index_pair = (i, j)
                S2_value = sobol_indices['S2'][index_pair]
                S2_conf_width = sobol_indices['S2_conf'][index_pair]

                result_str = calculate_and_format_ci(S2_value, S2_conf_width)
                print(f"{name1}-{name2}: {result_str}")


def large_out_model(X, params):
    P1, P2, P3, P4 = params
    epsilon = np.random.normal(scale=0.1, size=len(X))  # Adding random noise

    Y = P1 + P2 * X + P3 * X**P4 + epsilon

    return Y


def large_model(X, params):
    """
    Somewhat inefficient to mimict performance of parallel execution with
    a simple model.
    :param X:
    :param params:
    :return:
    """
    np.random.seed(int(params[7] * 1000000))
    cond_eps = np.random.normal(0, 0.5)
    eps = np.random.normal(0, 0.1)
    stress = params[3] / (X + 2) + params[4]
    cond_x = params[0]  +  params[1] * X  + params[2] * 0.5 * (3 * X * X - 1)
    cond = np.exp( cond_x - params[5] * stress + cond_eps)
    flux = (stress + params[6]) * cond
    return flux * math.exp(eps)


class Analyze:
    def __init__(self, problem):
        self.problem = problem
    def single(self, output):
        return analyze.sobol(self.problem, output, calc_second_order=True, print_to_console=False)


def read_sensitivity_config(work_dir):
    # test if config exists, copy from rep_dir if necessary
    sens_config_file = os.path.join(work_dir, "config_sensitivity.yaml")

    if not os.path.exists(sens_config_file):
        raise Exception("Main configuration file 'config_sensitivity.yaml' not found in workdir.")

    # read config file
    with open(sens_config_file, "r") as f:
        sens_config_dict = yaml.safe_load(f)

    sens_config_dict["script_dir"] = os.path.dirname(os.path.abspath(__file__))
    sens_config_dict["rep_dir"] = os.path.abspath(os.path.join(sens_config_dict["script_dir"], "../../.."))
    return sens_config_dict


def prepare_pbs_scripts(sens_config_dict, output_dir, np):
    endorse_root = sens_config_dict["script_dir"]
    met = sens_config_dict["metacentrum"]

    def create_common_lines(id):
        name = met["name"] + "_" + id
        common_lines = [
            '#!/bin/bash',
            '#PBS -S /bin/bash',
            # '#PBS -l select=' + str(met["chunks"]) + ':ncpus=' + str(met["ncpus_per_chunk"]) + ':mem=' + met["memory"],
            '#PBS -l select=1:ncpus=1:mem=' + met["memory"],
            # '#PBS -l place=scatter',
            '#PBS -l walltime=' + str(met["walltime"]),
            '#PBS -q ' + met["queue"],
            '#PBS -N ' + name,
            '#PBS -o ' + os.path.join(output_dir, "sensitivity", name + '.out'),
            '#PBS -e ' + os.path.join(output_dir, "sensitivity", name + '.err'),
            '\n',
            'set -x',
            '\n# absolute path to output_dir',
            'output_dir="' + output_dir + '"',
            '\n',
            'sing_script="' + met["swrap"] + '"',
            '\n',
            'image="' + os.path.join(endorse_root, 'endorse.sif') + '"',
            'cd $output_dir'
        ]
        return common_lines

    pbs_file_list = []
    for n in range(np):
        id = str(n).zfill(2)
        csv_file = os.path.join(output_dir, "sensitivity", "params_" + id + ".csv")
        sample_subdir = os.path.join(output_dir, "sensitivity", "samples_" + id)
        # prepare PBS script
        common_lines = create_common_lines(id)
        lines = [
            *common_lines,
            '\n# finally gather the full command',
            os.path.join(endorse_root, "bin", "endorse-bayes") + " "
                + ' '.join(["-t", "set", "-o", output_dir, "-p", csv_file, "-x", sample_subdir, "-s", id]),
            # 'zip -r samples.zip solver_*', # avoid 'bash: Argument list too long'
            # 'find . -name "solver_*" -print0 | xargs -0 tar -zcvf samples.tar.gz',
            # 'find . -name "solver_*" -print0 | xargs -0 rm -r',
            # '\n' + ' '.join(['tar', '-zcvf', 'samples_' + id + '.tar.gz', sample_subdir]),
            # ' '.join(['rm', '-r', sample_subdir]),
            'echo "FINISHED"'
        ]
        pbs_file = os.path.join(output_dir, "sensitivity", "pbs_job_" + id + ".sh")
        with open(pbs_file, 'w') as f:
            f.write('\n'.join(lines))
        pbs_file_list.append(pbs_file)

    return pbs_file_list


def prepare_sets_of_params(parameters, output_dir_in, n_processes, par_names):
    no_samples, no_parameters = np.shape(parameters)
    rows_per_file = no_samples // n_processes + (no_samples % n_processes > 0)

    for i in range(n_processes):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, no_samples)
        subset_matrix = parameters[start_idx:end_idx, :]

        param_file = os.path.join(output_dir_in, "sensitivity", "params_" + str(i).zfill(2) + ".csv")
        with open(param_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(par_names)
            for row in subset_matrix:
                writer.writerow(row)

    # for i, mat in enumerate(sub_parameters):
    #     output_file = f"parameters_{str(i+1).zfill(2)}.npy"
    #     np.save(output_file, mat)
    #     print(f"Saved {output_file}")


def test_vtk_process():
    # Load the PVD file
    pvd_file_path = "JSmodel/output/flow.pvd"
    field_name = "pressure_p0"
    pvd_reader = pv.PVDReader(pvd_file_path)

    field_data_list = []
    for time_frame in range(len(pvd_reader.time_values)):
        pvd_reader.set_active_time_point(time_frame)
        mesh = pvd_reader.read()[0]  # MultiBlock mesh with only 1 block

        field_data = mesh[field_name]
        field_data_list.append(field_data)
        print(field_data.shape)

    sample_data = np.stack(field_data_list)
    # sample_data = np.concatenate(field_data_list)
    sample_data = sample_data.reshape((1,*sample_data.shape)) # axis 0 - sample
    print(sample_data.shape)

    file_path = 'test_bukov2.h5'
    # dataset_name = 'pressure'
    from endorse.Bukov2 import sample_storage

    n_samples = 10
    # sample_size = 90000
    # initial_shape = (K, M, N + extra_space)  # Pre-allocate extra space in the N dimension
    # chunks = (K, M, chunk_size)  # Define a suitable chunk size

    sample_storage.create_chunked_dataset(file_path, chunk_shape=sample_data.shape)

    # When you have new data to append
    # new_data = np.random.rand(K, M, n)  # Your new data
    for i in range(10):
        sample_storage.append_data(file_path, sample_data)




def test_salib_flow123d_2D():

    output_dir = os.path.abspath("../sandbox/salib_test_2D")
    n_processes = 10
    n_samples = 20

    aux_functions.force_mkdir(output_dir, force=True)
    shutil.copyfile("../test_data/config_sim_A04hm_V1_03.yaml", os.path.join(output_dir, "config.yaml"))
    # setup paths and directories
    config_dict = bayes_run_all.setup(output_dir, can_overwrite=False, clean=False)

    # Define the problem for SALib
    # Bayes Inversion borehole_V1/sim_A04hm_V1_04_20230713a
    problem = {
        'num_vars': 8,
        'names': ['storativity',
                  'young_modulus',
                  'initial_stress_x',
                  'initial_stress_y',
                  'perm_kr',
                  'perm_km',
                  'perm_beta',
                  'perm_gamma'],
        'dists': ['lognorm',
                  'lognorm',
                  'lognorm',
                  'lognorm',
                  'lognorm',
                  'lognorm',
                  'lognorm',
                  'lognorm'],
        # available distributions:
        # unif - interval given by bounds
        # logunif,
        # triang - [lower_bound, upper_bound, mode_fraction]
        # norm,  bounds : [mean, std]
        # truncnorm, bounds : [lower_bound, upper_bound, mean, std_dev]
        # lognorm, bounds: [mean, std]  # mean and std of the log(X)
        'bounds': [[-14.5662279378421, 2.0],
                   [23.2670944048714, 2.0],
                   [17.8553760319809, 2.0],
                   [16.2134058307626, 2.0],
                   [-49.4456937078649, 3.0],
                   [-33.8402223378873, 3.0],
                   [-13.1451669487322, 2.0],
                   [-13.007626024022, 2.0]]
    }

    # Generate Saltelli samples
    param_values = sample.saltelli(problem, 100, calc_second_order=True)

    sensitivity_dir = os.path.join(output_dir, "sensitivity")
    aux_functions.force_mkdir(sensitivity_dir, force=True)

    # plan sample parameters a prepare them in CSV
    prepare_sets_of_params(param_values, output_dir, n_processes, problem["names"])

    # plan parallel sampling, prepare PBS jobs
    pbs_file_list = prepare_pbs_scripts(config_dict, output_dir, n_processes)

    # print(param_values)
    # X = np.linspace(-1, 1, 1000)
    #
    # t = time.perf_counter()
    # # Evaluate the model for each set of parameters
    # combined_values  = ( (X, p) for p in param_values )
    # #with multiprocessing.Pool(4) as pool:
    # #    Y_samples = pool.starmap(large_model, combined_values)
    # Y_samples = [large_model(*c) for c in combined_values]
    # print(f"Sample time: {time.perf_counter() - t}")
    #
    # t = time.perf_counter()
    # # Perform Sobol sensitivity analysis
    # print("\n")
    # analyze = Analyze(problem)
    # with multiprocessing.Pool(4) as pool:
    #     indices = pool.map(analyze.single, np.array(Y_samples).T)
    # print(f"Analyze time: {time.perf_counter() - t}")

    # TODO: Plot indices spatial variability, using histograms
    # Use max of variability over time.
    # Unfortunately analysis takes quite some time:
    # Assuming linear dependence on number of outputs and number of samples we have:




