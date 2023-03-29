import os
import sys
import yaml
import time

import flow_wrapper
from measured_data import MeasuredData
import numpy as np

# OBSOLETE FILE

def just_run_flow123d(measured_data):
    wrap = flow_wrapper.Wrapper(solver_id=1)
    wrap.set_parameters(data_par=[7.794869596611009e-15, 0.14035455365537208])
    t = time.time()
    res = wrap.get_observations()
    md.plot_comparison(res, wrap.sim.sample_dir)
    print(res)
    print("LEN:", len(res[1]))
    print("TIME:", time.time() - t)
    exit(0)


if __name__ == "__main__":

    # setup paths and directories
    config_dict = flow_wrapper.setup_config()
    flow_wrapper.setup_dirs(config_dict)

    # RUN THE MCMC SIMULATION
    # default parameters
    N = 4  # number of sampling processes
    oversubscribe = False  # if there are not enough slots available
    visualize = False  # True = only visualization
    problem_path = None

    len_argv = len(sys.argv)
    assert len_argv > 1, "Specify configuration yaml file!"
    if len_argv > 1:
        problem_path = sys.argv[1]
    if len_argv > 2:
        N = int(sys.argv[2])  # number of MH/DAMH chains
    if len_argv > 3:
        oversubscribe = sys.argv[3] == "oversubscribe"
        visualize = sys.argv[3] == "visualize"

    basename = os.path.basename(problem_path)
    problem_name, fext = os.path.splitext(basename)

    # prepare measured data as observations
    md = MeasuredData(config_dict)
    md.initialize()

    md.plot_all_data()
    md.plot_interp_data()

    boreholes = ["HGT1-5", "HGT1-4", "HGT2-4", "HGT2-3"]
    times, values = md.generate_measured_samples(boreholes)

    # JUST RUN FLOW123D FOR TESTING
    # just_run_flow123d(md)

    with open(problem_path) as f:
        text = f.read()
  
    Y = yaml.YAML()
    conf = Y.load(text)
    conf["problem_parameters"]["observations"] = np.array(values).tolist()
    conf["no_observations"] = len(values)
    conf["noise_type"] = "Gaussian_process"
    conf["noise_grid"] = np.array(times).tolist()
    conf["noise_parameters"] = [[30, 50]] * len(boreholes)
    conf["solver_module_path"] = os.path.join(config_dict["script_dir"], "flow_wrapper.py")

    with open(problem_path, 'w') as f:
        Y.dump(conf, f)

    # run sampling
    command = None
    if visualize:
        # os.error("Visualization not implemented.")
        # if os.path.exists("examples/visualization/" + problem_name + ".py"):
        #     command = "python3 examples/visualization/" + problem_name + ".py " + str(N)
        # else:
        #     command = "python3 examples/visualization/general_visualization.py " + str(N) + " " + problem_name
        command = "python3 examples/visualization/general_visualization.py " + str(N) + " " + problem_path
    else:
        if oversubscribe:
            opt = " --oversubscribe "
        else:
            opt = " "
        sampler = "python3 -m mpi4py surrDAMH/process_SAMPLER.py "
        solver = "python3 -m mpi4py surrDAMH/process_SOLVER.py " + problem_path + " "
        collector = "python3 -m mpi4py surrDAMH/process_COLLECTOR.py "

        # possibly change to particular mpirun for testing
        # mpirun = "/usr/local/mpich_3.4.2/bin/mpirun"
        mpirun = "mpirun"
        if "surrogate_type" in conf.keys():
            command = mpirun + " -n " + str(N) + opt + sampler + ":" + " -n 1" + opt + solver + ":" + " -n 1" + opt + collector
        else:
            command = mpirun + " -n " + str(N) + opt + sampler + ":" + " -n 1" + opt + solver

        if "metacentrum" in config_dict.keys():
            met = config_dict["metacentrum"]
            ncpus = met["ncpus"]

            # bash -c "source ./venv/bin/activate && "
            lines = [
                '#!/bin/bash',
                '#PBS -S /bin/bash',
                '#PBS -l select=1:ncpus=' + str(ncpus+2) + ':mem=12gb',
                '#PBS -l walltime=' + met["walltime"],
                '#PBS -q ' + met["queue"],
                '#PBS -N ' + met["name"],
                '#PBS -j oe',
                '\n',
                'cd "' + config_dict["script_dir"] + '"',
                'image=$(./endorse_fterm image)',
                'sing_command="singularity exec -B ' + met["workspace_rel"] + '/:/' + met["workspace_rel"]
                        + ' docker://$image"',
                '\n',
                'sampler="' + sampler + '"',
                'solver="' + solver + '"',
                'collector="' + collector + '"',
                '\n',
                'bash_venv() {bash - c "source ./venv/bin/activate && ${1}"}',
                'command="mpirun '
                        + '-n ' + str(ncpus) + ' $sing_command bash_env($sampler) : '
                        + '-n 1 $sing_command bash_env($solver) : '
                        + '-n 1 $sing_command bash_env($collector)"',
                'echo $command',
                'eval $command'
            ]
            with open("shell_process.sh", 'w') as f:
                f.write('\n'.join(lines))

    if "metacentrum" in config_dict.keys():
        os.system("qsub " + os.path.join(config_dict["work_dir"], "shell_process.sh"))
    else:
        rep_dir = config_dict["script_dir"]
        os.chdir(os.path.join(rep_dir, "surrDAMH"))
        print(command)
        # exit(0)
        os.system(command)
