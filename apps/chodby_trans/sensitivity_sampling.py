import logging
import shutil
from typing import *
import csv
import os
from pathlib import Path

import numpy as np

from endorse import common
from endorse.common import dotdict, File, report, memoize
from endorse.sa import sample, analyze

import chodby_trans.sample_storage as sample_storage
import chodby_trans.input_data as input_data
input_dir = input_data.input_dir
work_dir = input_data.work_dir


def solver_id(i):
    return str(i).zfill(2)

def sampled_data_hdf(i):
    return 'sampled_data_' + solver_id(i) + '.h5'


def prepare_pbs_scripts(cfg, output_dir_in):
    np = cfg.sensitivity.n_processes
    # endorse_root = cfg["rep_dir"]
    endorse_root = (input_dir / "../../../../").resolve()
    pbs = cfg.machine_config.pbs

    pbs_dir = output_dir_in / input_data.pbs_job_dirname
    pbs_dir.mkdir()

    def create_common_lines(id):
        name = pbs.name + "_" + id
        common_lines = [
            '#!/bin/bash',
            '#PBS -S /bin/bash',
            # '#PBS -l select=' + str(met["chunks"]) + ':ncpus=' + str(met["ncpus_per_chunk"]) + ':mem=' + met["memory"],
            # scratch: charon ssd: 346/20 = 17,3 GB
            '#PBS -l select=1:ncpus=1:mem=' + pbs.mem + ":scratch_local=17gb",
            # '#PBS -l place=scatter',
            '#PBS -l walltime=' + str(pbs.walltime),
            '#PBS -q ' + pbs.queue,
            '#PBS -N ' + name,
            '#PBS -j oe',
            '#PBS -o ' + pbs_dir / (name + '.out'),
            #'#PBS -e ' + os.path.join(pbs_dir, name + '.err'),
            '\n',
            'set -x',
            'export TMPDIR=$SCRATCHDIR',
            '\n# absolute path to output_dir',
            'output_dir="' + work_dir + '"',
            'workdir=$SCRATCHDIR',
            #'\n',
            #'SWRAP="' + met["swrap"] + '"',
            #'IMG_MPIEXEC="/usr/local/mpich_4.0.3/bin/mpirun"',
            #'SING_IMAGE="' + os.path.join(endorse_root, 'endorse.sif') + '"',
            #'\n',
            'cd $output_dir',
            #'SCRATCH_COPY=$output_dir',
            #'python3 $SWRAP/smpiexec_prepare.py -i $SING_IMAGE -s $SCRATCH_COPY -m $IMG_MPIEXEC'
        ]
        return common_lines

    pbs_file_list = []
    for n in range(np):
        id = solver_id(n)
        sensitivity_dir = Path("$workdir") / input_data.sensitivity_dirname
        csv_file = "params_" + id + ".csv"

        sample_subdir = sensitivity_dir /("samples_" + id)
        sampled_data_out = Path("$workdir") / sampled_data_hdf(n)
        # prepare PBS script
        common_lines = create_common_lines(id)
        rsync_cmd = " ".join(["rsync -av",
                              #"--include " + os.path.join(sensitivity_dirname, empty_hdf_dirname, sampled_data_hdf(n)),
                              #"--include " + os.path.join(sensitivity_dirname, param_dirname, "params_" + id + ".csv"),
                              "--exclude *.h5",
                              "--exclude *.pdf",
                              "--exclude " + os.path.join(input_data.sensitivity_dirname, input_data.empty_hdf_dirname),
                              "--exclude " + os.path.join(input_data.sensitivity_dirname, input_data.param_dirname),
                              "--exclude " + os.path.join(input_data.sensitivity_dirname, input_data.pbs_job_dirname),
                              "$output_dir" + "/",
                              "$workdir"])
        lines = [
            *common_lines,
            '\n',
            rsync_cmd,
            ' '.join(['cp',
                      os.path.join(input_data.sensitivity_dirname, input_data.empty_hdf_dirname, sampled_data_hdf(n)),
                      "$workdir"]),
            ' '.join(['cp',
                      os.path.join(input_data.sensitivity_dirname, input_data.param_dirname, csv_file),
                      "$workdir"]),
            'cd $workdir',
            'pwd',
            'ls -la',
            '\n# finally gather the full command',
            os.path.join(endorse_root, "bin", "endorse-bayes") + " "
                + ' '.join(["-t", "set", "-o", "$workdir", "-p", csv_file, "-x", sample_subdir, "-s", id]),
            # 'zip -r samples.zip solver_*', # avoid 'bash: Argument list too long'
            # 'find . -name "solver_*" -print0 | xargs -0 tar -zcvf samples.tar.gz',
            # 'find . -name "solver_*" -print0 | xargs -0 rm -r',
            # '\n' + ' '.join(['tar', '-zcvf', 'samples_' + id + '.tar.gz', sample_subdir]),
            # ' '.join(['rm', '-r', sample_subdir]),
            'ls -la',
            'mkdir -p $output_dir/sampled_data',
            'time cp ' + str(sampled_data_out) + ' $output_dir/sampled_data',
            #'time cp -r sensitivity $output_dir',
            'clean_scratch',
            'echo "FINISHED"'
        ]
        pbs_file = os.path.join(pbs_dir, "pbs_job_" + id + ".sh")
        with open(pbs_file, 'w') as f:
            f.write('\n'.join(lines))
        pbs_file_list.append(pbs_file)

    return pbs_file_list


def prepare_sets_of_params(cfg_sens: dotdict, parameters, output_dir_in, par_names):
    n_processes = cfg_sens.n_processes
    no_samples, no_parameters = np.shape(parameters)
    rows_per_file = no_samples // n_processes
    rem = no_samples % n_processes

    param_dir = output_dir_in / input_data.param_dirname
    param_dir.mkdir()
    empty_hdf_dir = output_dir_in / input_data.empty_hdf_dirname
    empty_hdf_dir.mkdir()

    sample_idx = 0
    off_start = 0
    off_end = 0
    for i in range(n_processes):
        off_start = off_end
        off_end = off_end + rows_per_file
        # add sample while there is still remainder after rows_per_file division
        if rem > 0:
            off_end = off_end + 1
            rem = rem - 1
        subset_matrix = parameters[off_start:off_end, :]

        param_file = param_dir / ("params_" + solver_id(i) + ".csv")
        with open(param_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['idx', *par_names])
            for row in subset_matrix:
                writer.writerow([sample_idx, *row])
                sample_idx = sample_idx+1

        # Prepare HDF, write parameters
        output_file = empty_hdf_dir / sampled_data_hdf(i)
        n_params = parameters.shape[0]
        n_times = cfg_sens.sample_shape[0]
        n_elements = cfg_sens.sample_shape[1]
        sample_storage.create_chunked_dataset(output_file,
                                              shape=(n_params, n_times, n_elements),
                                              chunks=(1, 1, n_elements))
        sample_storage.append_new_dataset(output_file, "parameters", parameters)

    # for i, mat in enumerate(sub_parameters):
    #     output_file = f"parameters_{str(i+1).zfill(2)}.npy"
    #     np.save(output_file, mat)
    #     print(f"Saved {output_file}")


def salib_samples(cfg: dotdict, seed):
    cfg_sens = cfg.sensitivity
    # Define the problem for SALib
    # Bayes Inversion borehole_V1/sim_A04hm_V1_04_20230713a
    problem = sample.prepare_problem_defition(cfg_sens.parameters)
    print(problem)

    # Generate Saltelli samples
    param_values = sample.saltelli(problem, cfg_sens.n_samples, calc_second_order=cfg_sens.second_order_sa)
    # param_values = sample.sobol(problem, n_samples, calc_second_order=True)
    print(param_values.shape)

    # # plot requires LaTeX installed
    # plot_conductivity(cfg, param_values)
    # # exit(0)

    sensitivity_dir = Path(input_data.sensitivity_dirname)
    if sensitivity_dir.exists():
        shutil.rmtree(sensitivity_dir)
    sensitivity_dir.mkdir()

    # plan sample parameters a prepare them in CSV
    prepare_sets_of_params(cfg_sens, param_values, sensitivity_dir, problem["names"])
    # exit(0)

    # plan parallel sampling, prepare PBS jobs
    if cfg.machine_config.pbs is not None:
        pbs_file_list = prepare_pbs_scripts(cfg, sensitivity_dir)
    else:
        pass

def main():
    # common.EndorseCache.instance().expire_all()

    conf_file = input_data.transport_config
    cfg = common.config.load_config(str(conf_file))

    seed = 101
    with common.workdir(str(work_dir), clean=False):

        shutil.copytree(input_dir, work_dir / input_dir.name, dirs_exist_ok=True)
        salib_samples(cfg, seed)


if __name__ == '__main__':
    main()
