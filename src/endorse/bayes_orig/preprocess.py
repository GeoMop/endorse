import os
import sys
import ruamel.yaml as yaml
import numpy as np

import flow_wrapper
from measured_data import MeasuredData
from mesh_factory import MeshFactory


def preprocess(config_dict):
    # prepare measured data as observations
    md = MeasuredData(config_dict)
    md.initialize()

    md.plot_all_data()
    md.plot_interp_data()

    conf_bayes = config_dict["surrDAMH_parameters"]

    pressure_obs_points = conf_bayes["observe_points"]
    conductivity_obs_points = conf_bayes["conductivity_observe_points"]
    times, values = md.generate_measured_samples(pressure_obs_points)

    config_bayes_file = config_dict["bayes_config_file"]
    yaml_handler = yaml.YAML()
    with open(config_bayes_file) as f:
        file_content = f.read()
    conf = yaml_handler.load(file_content)
    # print(conf.ca)

    npob = len(pressure_obs_points)
    ncob = len(conductivity_obs_points)
    npar = len(conf_bayes["parameters"])
    conf["no_parameters"] = npar
    # not necessary due to conf["noise_parameters"]
    # conf["problem_parameters"]["noise_std"] = [noise_std] # * len(values)
    conf["problem_parameters"]["observations"] = np.array(values).tolist()
    conf["problem_parameters"]["prior_mean"] = [0.0] * npar
    conf["problem_parameters"]["prior_std"] = [1.0] * npar
    conf["no_observations"] = len(values)

    noise_model_list = []
    # noise for pressure head in boreholes
    dict_01 = dict()
    dict_01["time_grid"] = np.array(times).tolist()
    dict_01["corr_length"] = 30
    dict_01["std"] = 20
    dict_01["cov_type"] = "default"
    offset = 0
    for i in range(npob):
        d = dict_01.copy()
        length = len(times)
        d["range"] = [offset, offset + length]
        noise_model_list.append(d)
        offset = offset + length
    # noise for conductivity
    dict_02 = dict_01.copy()
    dict_02["time_grid"] = np.array(times[-1]).tolist()
    dict_02["corr_length"] = 0
    dict_02["std"] = 1.0
    for i in range(ncob):
        d = dict_02.copy()
        length = 1
        d["range"] = [offset, offset + length]
        noise_model_list.append(d)
        offset = offset + length

    conf["noise_model"] = noise_model_list

    conf["solver_module_path"] = os.path.join(config_dict["script_dir"], "flow_wrapper.py")
    conf["transformations"] = conf_bayes["parameters"]
    conf["observe_points"] = pressure_obs_points

    for i, par in enumerate(conf_bayes["parameters"]):
        if par["type"] is None:
            conf["problem_parameters"]["prior_mean"][i] = par["options"]["mu"]
            conf["problem_parameters"]["prior_std"][i] = par["options"]["sigma"]

    conf["no_solvers"] = int(np.round(0.5*(config_dict["metacentrum"]["chunks"] * config_dict["metacentrum"]["ncpus_per_chunk"]-2)))

    with open(config_bayes_file, 'w') as f:
        yaml_handler.dump(conf, f)

    MeshFactory.prepare_mesh(config_dict["geometry"], config_dict["common_files_dir"], cut_tunnel=True)


if __name__ == "__main__":

    output_dir = None
    len_argv = len(sys.argv)
    assert len_argv > 1, "Specify output directory!"
    if len_argv > 1:
        output_dir = sys.argv[1]

    config_dict = flow_wrapper.setup_config(output_dir)
    preprocess(config_dict)


