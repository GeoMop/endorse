import os
import sys
import shutil
import yaml

import aux_functions
from preprocess import preprocess

from surrDAMH.configuration import Configuration

# this script is supposed to be dependent only on python packages present on any machine
# all other python scripts are later run inside docker container

def setup(output_dir, can_overwrite, clean):
    # create and cd workdir
    rep_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = output_dir

    # Files in the directory are used by each simulation at that level
    common_files_dir = os.path.join(work_dir, "common_files")
    # Create working directory if necessary
    aux_functions.force_mkdir(common_files_dir, force=clean)
    os.chdir(work_dir)

    # test if config exists, copy from rep_dir if necessary
    config_file = os.path.join(work_dir, "config.yaml")
    if not os.path.exists(config_file):
        # to enable processing older results
        config_file = os.path.join(common_files_dir, "config.yaml")
        if not os.path.exists(config_file):
            raise Exception("Main configuration file 'config.yaml' not found in workdir.")
        else:
            import warnings
            warnings.warn("Main configuration file 'config.yaml' found in 'workdir/common_files'.",
                          category=DeprecationWarning)

    # read config file and setup paths
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict["work_dir"] = work_dir
    config_dict["script_dir"] = rep_dir

    config_dict["common_files_dir"] = common_files_dir
    config_dict["bayes_config_file"] = os.path.join(common_files_dir,
                                                    config_dict["surrDAMH_parameters"]["config_file"])

    # copy common files
    for f in config_dict["copy_files"]:
        filepath = os.path.join(common_files_dir, f)
        if not os.path.isfile(filepath) or can_overwrite:
            shutil.copyfile(os.path.join(rep_dir, f), filepath)

    return config_dict

def create_bash_python_script(venv_path, filename, command):
    with open(filename, 'w') as f:
        f.write('\n'.join(['#!/bin/bash',
                           'source ' + os.path.join(venv_path, 'bin', 'activate'),
                           command]))

    abs_filename = os.path.abspath(filename)
    os.popen('chmod +x ' + abs_filename)
    return abs_filename

if __name__ == "__main__":

    # resolve root dir of Endorse repository
    script_dir = os.path.dirname(os.path.abspath(__file__))
    endorse_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    venv_path = os.path.join(endorse_root, "venv_bayes")

    # default parameters
    output_dir = "flow123d_sim"
    N = 2  # default number of sampling processes
    oversubscribe = False  # if there are not enough slots available
    visualize = False  # True = only visualization
    clean = False

    # read parameters
    len_argv = len(sys.argv)
    assert len_argv > 1, "Specify configuration yaml file!"
    if len_argv > 1:
        output_dir = os.path.abspath(sys.argv[1])
    if len_argv > 2:
        N = int(sys.argv[2])  # number of MH/DAMH chains
    if len_argv > 3:
        clean = sys.argv[3] == "clean"
    if len_argv > 4:
        oversubscribe = sys.argv[4] == "oversubscribe"
        visualize = sys.argv[4] == "visualize"


    # setup paths and directories
    config_dict = setup(output_dir, can_overwrite=(not visualize), clean=clean)
    problem_path = config_dict["bayes_config_file"]

    surrDAMH_path = os.path.join(endorse_root, "submodules/surrDAMH")
    # run sampling
    # paths are relative to repository dir
    # paths passed to surrDAMH are absolute
    command = None
    if visualize:
        # os.error("Visualization not implemented.")
        # os.chdir(script_dir)
        if not os.path.isfile(problem_path):
            raise Exception("Missing problem configuration '" + problem_path + "'."
                            + " Call simulation with 'run' command first!")
        print(problem_path, flush=True)
        C = Configuration(N, problem_path)
        args = [str(N), problem_path, output_dir]
        if os.path.exists(surrDAMH_path + "/examples/visualization/" + C.problem_name + ".py"):
            command = "python3 " + surrDAMH_path + "/examples/visualization/" + C.problem_name + ".py " + " ".join(args)
        else:
            command = "python3 " + surrDAMH_path + "/examples/visualization/general_visualization.py " + " ".join(args)
    else:
        if oversubscribe:
            opt = " --oversubscribe "
        else:
            opt = " "
        sampler = "python3 -m mpi4py " + surrDAMH_path + "/surrDAMH/process_SAMPLER.py " + output_dir
        solver = "python3 -m mpi4py " + surrDAMH_path + "/surrDAMH/process_SOLVER.py " + problem_path + " " + output_dir
        collector = "python3 -m mpi4py " + surrDAMH_path + "/surrDAMH/process_COLLECTOR.py"

        # prepare running command for local run
        # or prepare PBS script for running on Metacentrum
        if not config_dict["run_on_metacentrum"]:
            # possibly change to particular mpirun for testing
            # mpirun = "/usr/local/mpich_3.4.2/bin/mpirun"
            # mpirun = "mpirun"
            # mpirun = "mpiexec -envnone"
            mpirun = "mpiexec"
            command = mpirun + " -n " + str(N) + opt + sampler \
                      + " : " + "-n 1" + opt + solver + " : " + "-n 1" + opt + collector
        else:
            sampler_bash = create_bash_python_script(venv_path, "sampler.sh", sampler)
            solver_bash = create_bash_python_script(venv_path, "solver.sh", solver)
            collector_bash = create_bash_python_script(venv_path, "collector.sh", collector)

            met = config_dict["metacentrum"]
            common_lines = [
                'set -x',
                '\n# absolute path to output_dir',
                'output_dir="' + output_dir + '"',
                '\n# Endorse root directory',
                # 'endorse_root="' + endorse_root + '"',
                # '\n# command for running correct docker image',
                # 'image_name="$(./endorse_fterm image)"',
                # '\n',
                # 'image=$( echo "$image_name.sif" | tr /: _ )'
                '\n',
                'sing_script="' + met["swrap"] + '"',
                '\n',
                'image="' + os.path.join(endorse_root, 'endorse.sif') + '"',
                'cd $output_dir'
            ]

            # prepare PBS script
            lines = [
                '#!/bin/bash',
                '#PBS -S /bin/bash',
                '#PBS -l select=' + str(met["chunks"]) + ':ncpus=' + str(met["ncpus_per_chunk"]) + ':mem=' + met["memory"],
                '#PBS -l place=scatter',
                '#PBS -l walltime=' + str(met["walltime"]),
                '#PBS -q ' + met["queue"],
                '#PBS -N ' + met["name"],
                '#PBS -o ' + os.path.join(output_dir,met["name"] + '.out'),
                '#PBS -e ' + os.path.join(output_dir,met["name"] + '.err'),
                '\n',
                *common_lines,
                '\n# finally gather the full command',
                'command="python3 $sing_script -i $image -- '
                        + ' '.join(['-n', str(N),sampler_bash, ':',
                                    '-n', str(1),solver_bash, ':',
                                    '-n', str(1),collector_bash]) + '"',
                'echo $command', 'eval $command', '\n',
                'command="' + ' '.join([os.path.join(script_dir,'run_visualize.sh'), '-n', str(N), '-o', output_dir, '-t', 'visualize', '-s']) + '"',
                'echo $command', 'eval $command', '\n',
                # 'zip -r samples.zip solver_*', # avoid 'bash: Argument list too long'
                'find . -name "solver_*" -print0 | xargs -0 tar -zcvf samples.tar.gz',
                # 'rm -r solver_*',
                'find . -name "solver_*" -print0 | xargs -0 rm -r',
                'command="' + ' '.join([os.path.join(script_dir,'run_set.sh'), output_dir, str(config_dict["run_best_n_accepted"]), 'sing']) + '"',
                'echo $command', 'eval $command'
            ]
            with open("pbs_job.sh", 'w') as f:
                f.write('\n'.join(lines))

        print("preprocess", flush=True)
        preprocess(config_dict)

    # final command call
    if not config_dict["run_on_metacentrum"] or visualize:
        # local command call
        os.chdir(script_dir)
        print(command)
        os.system(command)
    else:
        # PBS script
        # os.system("qsub " + os.path.join(config_dict["work_dir"], "pbs_job.sh"))
        exit(0)
