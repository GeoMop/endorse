import sys
import os.path

from endorse.sa import sample, analyze
from endorse.bayes_orig import aux_functions
from endorse.bayes_orig import run_all as bayes_run_all
import endorse.Bukov2.sample_storage as sample_storage

import numpy as np
import csv


class Analyze:
    def __init__(self, problem):
        self.problem = problem

    def single(self, output):
        return analyze.sobol(self.problem, output, calc_second_order=True, print_to_console=False)


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

    sample_idx = 0
    for i in range(n_processes):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, no_samples)
        subset_matrix = parameters[start_idx:end_idx, :]

        param_file = os.path.join(output_dir_in, "sensitivity", "params_" + str(i).zfill(2) + ".csv")
        with open(param_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['idx', *par_names])
            for row in subset_matrix:
                writer.writerow([sample_idx, *row])
                sample_idx = sample_idx+1

    # for i, mat in enumerate(sub_parameters):
    #     output_file = f"parameters_{str(i+1).zfill(2)}.npy"
    #     np.save(output_file, mat)
    #     print(f"Saved {output_file}")


if __name__ == "__main__":

    # default parameters
    output_dir = None

    len_argv = len(sys.argv)
    assert len_argv > 1, "Specify output dir."
    if len_argv > 1:
        output_dir = os.path.abspath(sys.argv[1])

    # aux_functions.force_mkdir(output_dir, force=True)
    # shutil.copyfile("../test_data/config_sim_A04hm_V1_03.yaml", os.path.join(output_dir, "config.yaml"))
    # setup paths and directories
    config_dict = bayes_run_all.setup(output_dir, can_overwrite=False, clean=False)
    # add repository dir
    config_dict["rep_dir"] = os.path.abspath(os.path.join(config_dict["script_dir"], "../../.."))

    # Define the problem for SALib
    # Bayes Inversion borehole_V1/sim_A04hm_V1_04_20230713a
    params = config_dict["parameters"]
    problem = {
        'num_vars': len(params),
        'names': [p["name"] for p in params],
        'dists': [p["type"] for p in params],
        # available distributions:
        # unif - interval given by bounds
        # logunif,
        # triang - [lower_bound, upper_bound, mode_fraction]
        # norm,  bounds : [mean, std]
        # truncnorm, bounds : [lower_bound, upper_bound, mean, std_dev]
        # lognorm, bounds: [mean, std]  # mean and std of the log(X)
        'bounds': [p["bounds"] for p in params]
    }

    # Generate Saltelli samples
    param_values = sample.saltelli(problem, config_dict["n_samples"], calc_second_order=True)
    # param_values = sample.sobol(problem, n_samples, calc_second_order=True)
    print(param_values.shape)

    sensitivity_dir = os.path.join(output_dir, "sensitivity")
    aux_functions.force_mkdir(sensitivity_dir, force=True)

    # plan sample parameters a prepare them in CSV
    prepare_sets_of_params(param_values, output_dir, config_dict["n_processes"], problem["names"])

    # plan parallel sampling, prepare PBS jobs
    pbs_file_list = prepare_pbs_scripts(config_dict, output_dir, config_dict["n_processes"])

    # Prepare HDF, write parameters
    output_file = os.path.join(output_dir, 'sampled_data.h5')
    sample_storage.create_chunked_dataset(output_file, shape=(param_values.shape[0], *config_dict["sample_shape"]))
    sample_storage.append_new_dataset(output_file, "parameters", param_values)
