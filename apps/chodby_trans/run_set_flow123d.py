import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import time
import ruamel.yaml as yaml
import numpy as np
import logging

from endorse import common
# from endorse.Bukov2 import sample_storage, flow_wrapper
import chodby_trans.sample_storage as sample_storage
import chodby_trans.input_data as input_data
import chodby_trans.transport_wrapper as transport_wrapper


def setup(output_dir, can_overwrite, clean):
    # create and cd workdir
    rep_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = output_dir

    # Files in the directory are used by each simulation at that level
    common_files_dir = os.path.join(work_dir, "common_files")
    # Create working directory if necessary
    common.force_mkdir(common_files_dir, force=clean)
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
        yaml_reader = yaml.YAML(typ='safe', pure=True)
        config_dict = yaml_reader.load(f)

    config_dict["work_dir"] = work_dir
    config_dict["script_dir"] = rep_dir

    config_dict["common_files_dir"] = common_files_dir
    # config_dict["bayes_config_file"] = os.path.join(common_files_dir,
    #                                                 config_dict["surrDAMH_parameters"]["config_file"])

    # copy common files
    for f in config_dict["copy_files"]:
        filepath = os.path.join(common_files_dir, f)
        if not os.path.isfile(filepath) or can_overwrite:
            shutil.copyfile(os.path.join(rep_dir, f), filepath)

    return config_dict

def run_samples(cfg, params_in, output_dir_in, solver_id):

    wrap = transport_wrapper.Wrapper(cfg=cfg)
    if not output_dir_in.exists():
        output_dir_in.mkdir(mode=0o775, exist_ok=True)

    for pars in params_in:
        idx = int(pars[0])

        # create sample dir
        sample_dir = output_dir_in / ("solver_" + str(solver_id).zfill(2) +
                                           "_sample_" + str(idx).zfill(3))
        sample_dir.mkdir(mode=0o775, exist_ok=True)

        logging.info("=========================== RUNNING CALCULATION " +
                     "solver {} ".format(solver_id).zfill(2) +
                     "sample {} ===========================".format(idx).zfill(3))
        logging.info(sample_dir)

        with common.workdir(str(sample_dir), clean=False):

            wrap.set_parameters(data_par=pars[1:])
            t = time.time()
            res, sample_data = wrap.get_observations()

            print("Flow123d res: ", res)

            # print("LEN:", len(obs_data))
            print("TIME:", time.time() - t)

            # # write output
            # if config_dict["sample_subdir"] is not None:
            #     output_file = os.path.join(config_dict["sample_subdir"], 'output_' + str(solver_id) + '.csv')
            # else:
            #     output_file = os.path.join(output_dir_in, 'output_' + str(solver_id) + '.csv')
            # with open(output_file, 'a') as file:
            #     line = str(idx) + ',' + ','.join([str(s) for s in obs_data])
            #     file.write(line + "\n")

            output_file = os.path.join(output_dir_in, 'sampled_data_' + str(solver_id) + '.h5')
            sample_storage.set_sample_data(output_file, sample_data, idx)


def add_output_keys(config_dict):
    fname = config_dict["hm_params"]["in_file"]
    fname_output = fname + '_vtk'
    ftemplate = os.path.join(config_dict["common_files_dir"], fname + '_tmpl.yaml')
    ftemplate_output = os.path.join(config_dict["common_files_dir"], fname_output + '_tmpl.yaml')

    yaml_handler = yaml.YAML()
    with open(ftemplate, "r") as f:
        file_content = f.read()
    template = yaml_handler.load(file_content)

    flow_fields = [
       {"field": "conductivity", "interpolation": "P1_average"},
       "piezo_head_p0",
       "pressure_p0",
       "velocity_p0",
       "region_id"
    ]
    template["problem"]["flow_equation"]["flow_equation"]["output"]["fields"] = flow_fields

    mech_fields = [
        {"field": "displacement", "interpolation": "P1_average"},
        "stress",
        "displacement_divergence",
        "mean_stress",
        "von_mises_stress",
        "initial_stress",
        "region_id"
    ]
    template["problem"]["flow_equation"]["mechanics_equation"]["output"]["fields"] = mech_fields

    config_dict["hm_params"]["in_file"] = fname_output
    with open(ftemplate_output, 'w') as f:
        yaml_handler.dump(template, f)


def main(work_dir, solver_id):
    sensitivity_dir = work_dir / input_data.sensitivity_dirname
    csv_file = sensitivity_dir / input_data.param_dirname / ("params_" + solver_id + ".csv")
    sample_subdir = sensitivity_dir / ("samples_" + solver_id)

    # conf_file = input_data.transport_config
    conf_file = work_dir / input_data.transport_config.parent.name / input_data.transport_config.name
    cfg = common.config.load_config(str(conf_file))

    # seed = 101
    # with common.workdir(str(work_dir), clean=False):
    #     shutil.copytree(input_dir, work_dir / input_dir.name, dirs_exist_ok=True)
    #     salib_samples(cfg, seed)

    # setup paths and directories
    # config_dict = setup(output_dir, can_overwrite=False, clean=False)
    # if "vtk_output" in config_dict and config_dict["vtk_output"]:
    #     add_output_keys(config_dict)
    if sample_subdir is not None:
        cfg["sample_subdir"] = sample_subdir

    # preprocess(config_dict)

    print("Reading parameters from CSV: ", csv_file)
    pd_samples = pd.read_csv(csv_file, header=0)
    parameters = np.array(pd_samples.iloc[:, :])

    # print(parameters)
    run_samples(cfg, parameters, sample_subdir, solver_id)


if __name__ == "__main__":

    # common.CallCache.instance(expire_all=True)

    # default parameters
    work_dir = None
    solver_id = 0

    len_argv = len(sys.argv)
    assert len_argv > 2, "Specify output dir and parameters in csv file!"
    if len_argv > 1:
        work_dir = Path(sys.argv[1]).resolve()
    if len_argv > 2:
        solver_id = sys.argv[2]

    main(work_dir, solver_id)
