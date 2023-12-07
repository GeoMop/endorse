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
        raise Exception("Main configuration file 'config_sensitivity.yaml' not found in workdir.")

    # read config file
    with open(sens_config_file, "r") as f:
        sens_config_dict = yaml.safe_load(f)

    sens_config_dict["script_dir"] = os.path.dirname(os.path.abspath(__file__))
    sens_config_dict["rep_dir"] = os.path.abspath(os.path.join(sens_config_dict["script_dir"], "../../.."))
    return sens_config_dict


def prepare_sets_of_params(config_dict_in, output_dir_in, n_processes, n_best_fits):
    no_parameters = len(config_dict_in['surrDAMH_parameters']['parameters'])

    conf_bayes_path = os.path.join(output_dir_in, "common_files", config_dict_in["surrDAMH_parameters"]["config_file"])
    with open(conf_bayes_path) as f:
        config_bayes = yaml.safe_load(f)
    observations = np.array(config_bayes["problem_parameters"]["observations"])
    basename = os.path.basename(conf_bayes_path)
    problem_name, fext = os.path.splitext(basename)
    output_dir_bayes = os.path.join(output_dir_in, 'saved_samples', problem_name)

    # read posterior estimates for std
    with open(os.path.join(output_dir_bayes, "output.yaml")) as f:
        output_yaml = yaml.safe_load(f)
        estimates = output_yaml["estimated_distributions"]
        estimates_std = np.array([float(s["sigma"]) for s in estimates])

    raw_data = RawData()
    raw_data.load(output_dir_bayes, no_parameters, len(observations))
    # type: 0-accepted, 1-prerejected, 2-rejected
    raw_data_filtered = raw_data.filter(types=[0, 2], stages=[0])

    analysis_pe = Analysis(config=config_bayes, raw_data=raw_data_filtered)
    fits, norms = analysis_pe.find_n_best_fits(observations, count=n_best_fits, norm="L2")

    bin = int(n_best_fits/n_processes)
    rest_bin = n_best_fits - n_processes*bin
    bins = []
    for i in range(n_processes):
        bins.append(bin)
        if rest_bin > 0:
            bins[i] = bins[i] + 1
            rest_bin = rest_bin - 1

    sample_indices = np.array([s.idx for s in fits])
    offset = 0
    # close diff: 0.05 sigma of posterior
    # distant diff: 0.15 sigma of posterior
    diff = 0.05*np.diag(estimates_std)
    # print(diff)
    # print(bins)
    for n, b in enumerate(bins):
        parameters = raw_data.parameters[sample_indices[offset:offset+b], :]
        param_file = os.path.join(output_dir_in, "sensitivity", "params_" + str(n).zfill(2) + ".csv")

        with open(param_file, 'w') as file:
            line = ','.join(analysis_pe.par_names)
            file.write(line + "\n")

            # print(np.shape(parameters))
            shape = np.shape(parameters)
            for i in range(shape[0]):
                n_diffs = 4*no_parameters+1
                par_matrix = np.tile(parameters[i,:], (n_diffs,1))
                diffs = np.vstack((0 * parameters[i, :],    # orig parameters
                                   diff, -diff,             # close forward, backward diff
                                   3*diff, -3*diff))        # distant forward, backward diff
                par_matrix = par_matrix + diffs
                for k in range(n_diffs):
                    line = ','.join([str(s) for s in par_matrix[k,:]])
                    file.write(line + "\n")

        offset = offset + b


def prepare_pbs_scripts(sens_config_dict, output_dir, np):
    endorse_root = sens_config_dict["rep_dir"]
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
            '\n' + ' '.join(['tar', '-zcvf', 'samples_' + id + '.tar.gz', sample_subdir]),
            ' '.join(['rm', '-r', sample_subdir]),
            'echo "FINISHED"'
        ]
        pbs_file = os.path.join(output_dir, "sensitivity", "pbs_job_" + id + ".sh")
        with open(pbs_file, 'w') as f:
            f.write('\n'.join(lines))
        pbs_file_list.append(pbs_file)

    return pbs_file_list


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

    # plan sample parameters a prepare them in CSV
    prepare_sets_of_params(config_dict, output_dir, n_processes, n_best_params)

    # plan parallel sampling, prepare PBS jobs
    pbs_file_list = prepare_pbs_scripts(sens_config_dict, output_dir, n_processes)

