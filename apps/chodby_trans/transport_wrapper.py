import os, sys
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
import traceback
from typing import List
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

from endorse import common
import chodby_trans.fullscale_transport as transport
from chodby_trans import ot_sa

from endorse.fullscale_transport import output_times
import zarr_fuse as zf
import chodby_trans.input_data as input_data
import xarray as xr
import zarr

from dask.distributed import Lock


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
        print(f"salib parameters: {data_par}")

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
                    print(f"seed parameter: {param.name} = {par_seed}")
                    cfg.transport_fullscale[param.name] = par_seed
            else:
                pname = param.name
                assert pname in cfg.transport_fullscale, pname + " not in transport_fullscale"
                cfg.transport_fullscale[pname] = data_par[idx]

    def get_observations(self, tags, parameters):
        try:
            t = time.time()
            logging.info(f"transport_wrapper: get observations tags={tags}")
            #self.set_parameters(parameters)
            sa = ot_sa.SensitivityAnalysis.from_cfg(self._config.ot_sensitivity)
            param_dict = sa.param_vec_to_dict(parameters)
            rc, slice_array = transport.transport_run(
                self._config, 
                tags, param_dict)
            
            # test random results
            # cfg = self._config
            # param_names = [p.name for p in cfg.sensitivity.parameters]
            # times = output_times(cfg.transport_fullscale)
            # ng = 20
            # slice_array = np.random.rand(len(times), ng, ng, 2)

            sample_time = time.time() - t
            logging.info(f"SIMULATION TIME: {sample_time}")

            # ZARR FUSE
            # kwargs =  {"WORKDIR": str(input_data.zarr_store_path), "STORE_URL": str(input_data.zarr_store_path)}
            # data_schema = zf.schema.deserialize(input_data.data_schema_yaml) # read data scheme
            # root_node = zf.open_store(data_schema, **kwargs)
            # current_node = root_node[cfg.data_schema_key]

            # sample_dict = dict(
            #     iid=[tags[0]],  # coords
            #     qmc=[tags[1]],
            #     param_name=param_names,
            #     sim_time=times,
            #     X=np.linspace(2.0, 3.0, num=ng),
            #     Y=np.linspace(5.0, 8.0, num=ng),
            #     Z=np.array([1.0,3.5]),
            #     block=np.full(len(param_names), tags[2])[np.newaxis, :],  #values
            #     parameter= parameters[np.newaxis, np.newaxis, :], # coords: [ "iid", "qmc", "param_name"]
            #     conc=slice_array[np.newaxis, np.newaxis, ...] # coords: [ "iid", "qmc", "time", "X", "Y", "Z"]
            # )
            # logging.info(sample_dict)
            # current_node.update_dense(sample_dict)

            def _chunk_ranges(coord_slice: slice, chunk_len: int):
                start, stop = coord_slice.start, coord_slice.stop
                first = start // chunk_len
                last = (stop - 1) // chunk_len
                return range(first, last + 1)

            # DIRECT ZARR
            # Open the existing Zarr store as an Xarray dataset
            store_path = str(input_data.zarr_store_path)
            ds = xr.open_zarr(store_path, consolidated=False)

            # Validate slice_array shape
            expected_shape = (ds.sizes['sim_time'], ds.sizes['X'], ds.sizes['Y'], ds.sizes['Z'])
            if slice_array.shape != expected_shape:
                raise ValueError(f"slice_array must have shape {expected_shape}, got {slice_array.shape}")

            sample_idx = tags[0]
            qmc_idx = tags[1]
            iid_idx = tags[2]
            region = {'iid': slice(iid_idx, iid_idx + 1),
                      'qmc': slice(qmc_idx, qmc_idx + 1)}

            # Lock for every chunk being accessed by the current write
            # iid_chunk = ds.chunksizes["iid"][0]
            # qmc_chunk = ds.chunksizes["qmc"][0]
            # chunk_ids = list(_chunk_ranges(region['iid'], iid_chunk))
            # logging.info("lock chunk_ids: ", chunk_ids)
            # lock = Lock(f"zarr-write-iid-{iid_idx}")  # or per-chunk naming if you prefer
            # locks = [Lock(f"zarr-chunk-iid-{cid}") for cid in sorted(chunk_ids)]

            lock_names = []
            for var, chunkshape in ds.chunksizes.items():  # 'iid', 'qmc'
                for ciid in _chunk_ranges(region['iid'], chunkshape[0]):
                    for cqmc in _chunk_ranges(region['qmc'], chunkshape[0]):
                            lock_names.append(f"zarr-{var}-{ciid}-{cqmc}")
            lock_names = sorted(set(lock_names))  # deterministic ordering avoids deadlocks
            logging.info("lock_names: ", lock_names)
            locks = [Lock(name) for name in lock_names]

            for L in locks:
                L.acquire()

            try:
                ds_coords = xr.Dataset({
                    'conc': (['iid', 'qmc', 'sim_time', 'X', 'Y', 'Z'],
                              slice_array[np.newaxis, np.newaxis, ...]),
                    'return_code': (['iid', 'qmc'], np.array(rc)[np.newaxis, np.newaxis, ...]),
                    'sample_time': (['iid', 'qmc'], np.array(sample_time)[np.newaxis, np.newaxis, ...])
                })
                ds_coords.to_zarr(store_path, mode='a', region=region)
            except Exception as e:
                rc = -1001
                raise Exception('zarr error') from e
            finally:
                for L in reversed(locks): L.release()

            # ds_pars = xr.Dataset(
            #     {'parameter': (['iid', 'qmc', 'param_name'], parameters[np.newaxis, np.newaxis, ...])}
            # )
            # ds_pars.to_zarr(store_path, mode='a', region={
            #     'iid': slice(tags[0], tags[0] + 1),
            #     'qmc': slice(tags[1], tags[1] + 1)})

            # ds_coords = xr.Dataset(
            #     {'conc': (['iid', 'qmc', 'sim_time', 'X', 'Y', 'Z'],
            #               slice_array[np.newaxis, np.newaxis, ...])}
            # )
            # ds_coords.to_zarr(
            #     store_path,
            #     mode='a',
            #     region={
            #         'iid': slice(tags[0], tags[0]+1),
            #         'qmc': slice(tags[1], tags[1]+1),
            #         'sim_time': 'auto',
            #         'X': 'auto',
            #         'Y': 'auto',
            #         'Z': 'auto',
            #     }
            # )

            # ds = xr.Dataset(
            #     {
            #       # 'conc': (['iid', 'qmc'], slice_array), # add sample, qmc and block dims
            #       # 'parameter': (['iid', 'qmc'], parameters)
            #       'conc': (['iid', 'qmc', 'sim_time', 'X', 'Y', 'Z'], slice_array[np.newaxis, np.newaxis, ...]), # add sample, qmc and block dims
            #       # 'parameter': (['sim_time', 'X', 'Y', 'Z'], parameters)
            #     },
            #     # dims=('iid', 'qmc', 'param_name', 'sim_time', 'X', 'Y', 'Z'),
            #     coords={
            #         'iid': [tags[0]],
            #         'qmc': [tags[1]],
            #         # 'param_name': param_names,
            #         'sim_time': times,
            #         'X': np.arange(20),
            #         'Y': np.arange(20),
            #         'Z': np.arange(2),
            #         # 'X': np.linspace(2.0, 3.0, num=ng),
            #         # 'Y': np.linspace(5.0, 8.0, num=ng),
            #         # 'Z': np.array([1.0,3.5])
            #     }
            # )
            #
            # # ds.to_zarr(store_path, mode='a')
            # # Write the slice by specifying the region to overwrite
            # ds.to_zarr(
            #     store_path,
            #     mode='a',
            #     region={
            #         'iid': slice(tags[0], tags[0]+1),
            #         'qmc': slice(tags[1], tags[1]+1),
            #         # 'param_name': slice(block_idx, block_idx + 1),
            #         'sim_time': 'auto',
            #         'X': 'auto',
            #         'Y': 'auto',
            #         'Z': 'auto',
            #     }
            # )
            #
            # ds_pars = xr.Dataset(
            #     {
            #         'parameter': (['iid', 'qmc', 'param_name'], parameters[np.newaxis, np.newaxis, ...])
            #     },
            #     coords={
            #         'iid': [tags[0]],
            #         'qmc': [tags[1]],
            #         'param_name': param_names,
            #     }
            # )
            # # Write the slice by specifying the region to overwrite
            # ds_pars.to_zarr(
            #     store_path,
            #     mode='a',
            #     region={
            #         'iid': slice(tags[0], tags[0] + 1),
            #         'qmc': slice(tags[1], tags[1] + 1),
            #         'param_name': 'auto',
            #     }
            # )

            # Wrap slice_array into a DataArray with new sample and qmc coords
            # da = xr.DataArray(
            #     {
            #       'conc': (['iid', 'qmc'], slice_array), # add sample, qmc and block dims
            #       'parameter': (['iid', 'qmc'], parameters)
            #     },
            #     # dims=('iid', 'qmc', 'param_name', 'sim_time', 'X', 'Y', 'Z'),
            #     coords={
            #         'iid': [tags[0]],
            #         'qmc': [tags[1]],
            #         'param_name': param_names,
            #         'time': times,
            #         'X': np.linspace(2.0, 3.0, num=ng),
            #         'Y': np.linspace(5.0, 8.0, num=ng),
            #         'Z': np.array([1.0,3.5])
            #     }
            # )

            # Write the slice by specifying the region to overwrite
            # da.to_dataset(name='conc').to_zarr(
            #     store_path,
            #     mode='a',
            #     region={
            #         'iid': slice(sample_idx, sample_idx + 1),
            #         'qmc': slice(qmc_idx, qmc_idx + 1),
            #         'param_name': slice(block_idx, block_idx + 1),
            #         'sim_time': 'auto',
            #         'X': 'auto',
            #         'Y': 'auto',
            #         'Z': 'auto',
            #     }
            # )

            # return 0, []

            return rc, slice_array
        except Exception as e:
            sys.stdout.write("-"*60)
            sys.stdout.write(f"Traceback sample tags:{tags}")
            sys.stdout.write(f"transport_wrapper failed with exception: {e}")
            traceback.print_exc()
            sys.stdout.write("-"*60)
            sys.stdout.flush()
            # empty_block = np.zeros(18, 20, 20, 2)
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
        # transport.transport_run(cfg, cfg.transport_fullscale.dfn_macro)

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

