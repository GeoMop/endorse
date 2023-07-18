import os
import sys
import csv
import time
import ruamel.yaml as yaml
import numpy as np

import flow_wrapper
from measured_data import MeasuredData

import aux_functions
from run_all import setup
from preprocess import preprocess

from surrDAMH.modules.raw_data import RawData
from surrDAMH.modules.analysis import Analysis


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

def prepare_sets_of_params(config_dict_in, output_dir_in, n_processes, count):
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

    bin = np.int(count/n_processes)
    rest_bin = count - n_processes*bin
    bins = []
    for i in range(n_processes):
        bins.append(bin)
        if rest_bin > 0:
            bins[i] = bins[i] + 1
            rest_bin = rest_bin - 1

    sample_indices = np.array([s.idx for s in fits])
    offset = 0
    print(bins)
    for n, b in enumerate(bins):
        parameters = raw_data.parameters[sample_indices[offset:offset+b], :]
        param_file = os.path.join(output_dir_in, "sensitivity", "params_" + str(n).zfill(2) + ".csv")

        with open(param_file, 'w') as file:
            line = ','.join(analysis_pe.par_names)
            file.write(line + "\n")

            print(np.shape(parameters))
            for i in range(np.shape(parameters)[0]):
                line = ','.join([str(s) for s in parameters[i,:]])
                file.write(line + "\n")

        offset = offset + b

def prepare_pbs_scripts(sens_config_dict, output_dir, np):
    endorse_root = sens_config_dict["rep_dir"]
    met = sens_config_dict["metacentrum"]
    common_lines = [
        '#!/bin/bash',
        '#PBS -S /bin/bash',
        '#PBS -l select=' + str(met["chunks"]) + ':ncpus=' + str(met["ncpus_per_chunk"]) + ':mem=' + met["memory"],
        '#PBS -l place=scatter',
        '#PBS -l walltime=' + str(met["walltime"]),
        '#PBS -q ' + met["queue"],
        '#PBS -N ' + met["name"],
        '#PBS -o ' + os.path.join(output_dir, met["name"] + '.out'),
        '#PBS -e ' + os.path.join(output_dir, met["name"] + '.err'),
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

    for n in range(np):
        id = str(n).zfill(2)
        csv_file = os.path.join(output_dir, "sensitivity", "params_" + id + ".csv")
        sample_subdir = os.path.join(output_dir, "sensitivity", "samples_" + id)
        # prepare PBS script
        lines = [
            *common_lines,
            '\n# finally gather the full command',
            '\n.' + os.path.join(endorse_root, "bin", "endorse-bayes") + " "
                + ' '.join(["-t", "set", "-o", output_dir, "-p", csv_file, "-x", sample_subdir, "-s", id]),
            # 'zip -r samples.zip solver_*', # avoid 'bash: Argument list too long'
            # 'find . -name "solver_*" -print0 | xargs -0 tar -zcvf samples.tar.gz',
            # 'find . -name "solver_*" -print0 | xargs -0 rm -r',
            'echo "FINISHED"'
        ]
        with open(os.path.join(output_dir, "sensitivity", "pbs_job_" + id + ".sh"), 'w') as f:
            f.write('\n'.join(lines))


if __name__ == "__main__":

    # default parameters
    output_dir = None
    n_processes = 1
    n_best_params = 1

    len_argv = len(sys.argv)
    assert len_argv > 3, "Specify output dir & number of processes & number of best fits"
    if len_argv > 1:
        output_dir = os.path.abspath(sys.argv[1])
    if len_argv > 2:
        n_processes = int(sys.argv[2])
    if len_argv > 3:
        n_best_params = int(sys.argv[3])

    # setup paths and directories
    config_dict = setup(output_dir, can_overwrite=False, clean=False)
    # if config_dict["vtk_output"]:
        # add_output_keys(config_dict)

    # preprocess(config_dict)
    sens_config_dict = read_sensitivity_config(output_dir)

    sensitivity_dir = os.path.join(output_dir, "sensitivity")
    aux_functions.force_mkdir(sensitivity_dir, force=True)

    prepare_sets_of_params(config_dict, output_dir, n_processes, n_best_params)

    prepare_pbs_scripts(sens_config_dict, output_dir, n_processes)
    # if csv_data:
    #     print("Reading parameters from CSV: ", csv_data)
    #     with open(csv_data, newline='') as csvfile:
    #         parameters = list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
    # else:
    #     print("Getting " + str(n_best_params) + " best parameters.")
    #     parameters = get_best_accepted_params(config_dict, output_dir, n_best_params)

    # JUST RUN FLOW123D FOR TESTING
    # just_run_flow123d(config_dict, md, parameters, output_dir)
