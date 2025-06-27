import os
import subprocess
import time

import logging
import numpy as np
import scipy as scp
import pyvista as pv
import itertools
import collections
import shutil
import csv
import yaml
from typing import List
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

from endorse import common
import chodby_trans.input_data as input_data
import chodby_trans.fullscale_transport as transport


class Wrapper:

    def __init__(self, cfg: common.dotdict):

        # if "sample_subdir" in config:
        #     self.work_dir = Path(config["sample_subdir"])
        #     print(config["sample_subdir"])
        # else:
        #     self.work_dir = Path(config["work_dir"])
        self._config = cfg
        self.sample_dir = Path(".")
        self.sample_output_dir = "output"

    def set_parameters(self, data_par):
        cfg = self._config
        param_list = cfg.sensitivity.parameters
        assert(len(data_par) == len(param_list))

        fr_families = cfg.fractures.population
        # set parameters according to a type
        for idx, param in enumerate(param_list):
            if param.type == 'seed':
                # hash the seed to make the following distribution independent
                par_seed = hash(data_par[idx])

                # setup seed for scipy distributions
                if param.name.startswith('dfn') and 'parameters' in param:
                    par_randomGen = np.random.Generator(np.random.PCG64(par_seed))
                    for par in param.parameters:
                        for fr_fam in fr_families:
                            match par["type"]:
                                case "norm":
                                    mean = fr_fam[par.name]
                                    std = mean * par.bounds[1]
                                    val = scp.stats.norm.rvs(loc=mean, scale=std, random_state=par_randomGen)
                                    fr_fam[par.name] = val
                                case "lognorm":
                                    mean = np.log10(fr_fam[par.name])
                                    std = par.bounds[1]
                                    norm_val = scp.stats.norm.rvs(loc=mean, scale=std, random_state=par_randomGen)
                                    fr_fam[par.name] = 10**norm_val
                else:
                    # currently only `dfn_macro` seed for fractures
                    cfg.transport_fullscale[param.name] = par_seed
            else:
                pname = param.name
                assert pname in cfg.transport_fullscale, pname + " not in transport_fullscale"
                cfg.transport_fullscale[pname] = data_par[idx]

    def get_observations(self):
        try:
            print("get observations from transport_wrapper")
            res = self.calculate(self._config)
            return res
        except ValueError:
            print("transport_wrapper failed for unknown reason.")
            return -1000, []

    def calculate(self, cfg):
        """
        The program changes to <work_dir> directory.
        does all the data preparation, passing
        running simulation
        extracting results
        """

        # create sample dir
        # self.sample_counter = self.sample_counter + 1
        # self.sample_dir = self.work_dir/("solver_" + str(cfg["solver_id"]).zfill(2) +
        #                                  "_sample_" + str(self.sample_counter).zfill(3))
        # self.sample_dir.mkdir(mode=0o775, exist_ok=True)
        #
        # logging.info("=========================== RUNNING CALCULATION " +
        #       "solver {} ".format(cfg["solver_id"]).zfill(2) +
        #       "sample {} ===========================".format(self.sample_counter).zfill(3))
        # logging.info(self.sample_dir)

        # with common.workdir(str(self.sample_dir), clean=False):
        transport.transport_run(cfg, cfg.transport_fullscale.dfn_macro)

        # os.chdir(self.sample_dir)
        #
        # # collect only
        # # not used in any project (was used for dev in WGC2020)
        # # if config_dict["collect_only"]:
        # #     return 2, self.collect_results(config_dict)
        #
        # logging.info("Creating mesh...")
        # comp_mesh = self.prepare_mesh(config_dict, cut_tunnel=False)
        #
        # mesh_bn = os.path.basename(comp_mesh)
        # config_dict["hm_params"]["mesh"] = mesh_bn
        #
        # # endorse_2Dtest.read_physical_names(config_dict, comp_mesh)
        #
        # if config_dict["mesh_only"]:
        #     return -10, None  # tag, value_list
        #
        # # endorse_2Dtest.prepare_hm_input(config_dict)
        # hm_succeed, fo = self.call_flow(config_dict, 'hm_params', result_files=["flow_observe.yaml"])
        #
        # if not hm_succeed:
        #     # raise Exception("HM model failed.")
        #     # "Flow123d failed (wrong input or solver diverged)"
        #     logging.warning("Flow123d failed.")
        #     # still try collect results
        #     try:
        #         collected_values = self.collect_results(config_dict, fo)
        #         logging.info("Sample results collected.")
        #         return 3, collected_values  # tag, value_list
        #     except:
        #         logging.error("Collecting sample results failed:")
        #         traceback.print_exc()
        #         return -3, None
        #     # return -1, None  # tag, value_list
        # print("Running Flow123d - HM...finished")
        #
        # if self._config["make_plots"]:
        #     try:
        #         self.observe_time_plot(config_dict)
        #     except:
        #         logging.error("Making plot of sample results failed:")
        #         traceback.print_exc()
        #         return -2, None
        #
        # logging.info("Finished computation")
        #
        # # collected_values = self.collect_results(config_dict)
        # # print("Sample results collected.")
        # # return 1, collected_values  # tag, value_list
        #
        # try:
        #     collected_values = self.collect_results(config_dict, fo)
        #     logging.info("Sample results collected.")
        #     return 1, collected_values  # tag, value_list
        # except:
        #     logging.error("Collecting sample results failed:")
        #     traceback.print_exc()
        #     return -3, None


    def collect_results(self, cfg, fo: common.FlowOutput):
        # Load the PVD file
        # pvd_file_path = os.path.join(self.sample_output_dir, "flow.pvd")
        field_name = "pressure_p0"
        pvd_reader = pv.PVDReader(fo.hydro.spatial_file.path)

        field_data_list = []
        for time_frame in range(len(pvd_reader.time_values)):
            pvd_reader.set_active_time_point(time_frame)
            mesh = pvd_reader.read()[0]  # MultiBlock mesh with only 1 block

            field_data = mesh[field_name]
            field_data_list.append(field_data)

        sample_data = np.stack(field_data_list)
        sample_data = sample_data.reshape((1, *sample_data.shape))  # axis 0 - sample

        if cfg.sensitivity.clean_sample_dir:
            shutil.rmtree(self.sample_dir)

        return sample_data

    def get_from_observe(self, observe_dict, point_names, field_name, select_times=None):
        points = observe_dict['points']
        all_point_names = [p["name"] for p in points]
        # print('all_point_names', all_point_names)
        # print('point_names', point_names)
        points2collect_indices = []
        for p2c in point_names:
            tmp = [i for i, pn in enumerate(all_point_names) if pn == p2c]
            assert len(tmp) == 1
            points2collect_indices.append(tmp[0])

        # print("Collecting results for observe points: ", point_names)
        data = observe_dict['data']
        data_values = np.array([d[field_name] for d in data])
        values = data_values[:, points2collect_indices]
        obs_times = np.array([d["time"] for d in data]).transpose()

        if select_times is not None:
            # check that observe data are computed at all times of defined time axis
            all_times_computed = np.alltrue(np.isin(select_times, obs_times))
            if not all_times_computed:
                raise Exception("Observe data not computed at all times as defined by input!")
            # skip the times not specified in input
            t_indices = np.isin(obs_times, select_times).nonzero()
            values = values[t_indices]
        values = values.transpose()

        # if "smooth_factor" in config_dict.keys():
        #     smooth_factor = config_dict["smooth_factor"]
        #     for i in range(len(values)):
        #         values[i] = self.smooth_ode(times, values[i], smooth_factor)

        return values

    def call_flow(self, config_dict, param_key, result_files) -> (bool, common.FlowOutput):
        """
        Redirect sstdout and sterr, return true on succesfull run.
        :param result_files: Files to be computed - skip computation if already exist.
        :param param_key: config dict parameters key
        :param config_dict:
        :return:
        """

        params = config_dict[param_key]
        arguments = config_dict["local"]["flow_executable"].copy()

        if all([os.path.isfile(os.path.join(self.sample_output_dir, f)) for f in result_files]):
            status = True
            completed_process = subprocess.CompletedProcess(args=arguments, returncode=0)
        else:
            fname = params["in_file"]
            input_template = common.File(Path(config_dict["common_files_dir"])/(fname + '_tmpl.yaml'))
            completed_process, stdout, stderr = common.flow_call(self.sample_dir, arguments, input_template, params)
        status, fo = common.flow_check(self.sample_dir, completed_process, result_files)

        return status, fo


    def prepare_mesh(self, config_dict, cut_tunnel):
        mesh_name = config_dict["geometry"]["mesh_name"]
        if cut_tunnel:
            mesh_name = mesh_name + "_cut"
        # mesh_file = mesh_name + ".msh"
        # mesh_healed = mesh_name + "_healed.msh"
        mesh_healed = mesh_name + ".msh"
        print(mesh_healed)

        # suppose that the mesh was created/copied during preprocess
        print(os.path.join(config_dict["common_files_dir"]))
        print(mesh_healed)

        assert os.path.isfile(os.path.join(config_dict["common_files_dir"], mesh_healed))
        shutil.copyfile(os.path.join(config_dict["common_files_dir"], mesh_healed), mesh_healed)
        return mesh_healed

