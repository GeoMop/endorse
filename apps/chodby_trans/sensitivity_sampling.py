import logging
import shutil
from typing import *
from pathlib import Path

import numpy as np

from endorse import common
from endorse.common import dotdict, File, report, memoize
from endorse.sa import sample, analyze

import chodby_trans.input_data as input_data
input_dir = input_data.input_dir
work_dir = input_data.work_dir


def salib_samples(cfg: dotdict, seed):
    cfg_sens = cfg.sensitivity
    # Define the problem for SALib
    # Bayes Inversion borehole_V1/sim_A04hm_V1_04_20230713a
    problem = sample.prepare_problem_defition(cfg_sens["parameters"])
    print(problem)

    # Generate Saltelli samples
    param_values = sample.saltelli(problem, cfg_sens["n_samples"], calc_second_order=cfg_sens["second_order_sa"])
    # param_values = sample.sobol(problem, n_samples, calc_second_order=True)
    print(param_values.shape)

    # # plot requires LaTeX installed
    # plot_conductivity(cfg, param_values)
    # # exit(0)
    #
    # sensitivity_dir = os.path.join(output_dir, sensitivity_dirname)
    # common.force_mkdir(sensitivity_dir, force=True)
    #
    # # plan sample parameters a prepare them in CSV
    # prepare_sets_of_params(param_values, sensitivity_dir, config_dict["n_processes"], problem["names"])
    # # exit(0)
    #
    # # plan parallel sampling, prepare PBS jobs
    # pbs_file_list = prepare_pbs_scripts(config_dict, sensitivity_dir, config_dict["n_processes"])

def main():
    # common.EndorseCache.instance().expire_all()

    conf_file = input_data.transport_config
    cfg = common.config.load_config(str(conf_file))

    seed = 101
    with common.workdir(str(work_dir), clean=False):
        salib_samples(cfg, seed)


if __name__ == '__main__':
    main()
