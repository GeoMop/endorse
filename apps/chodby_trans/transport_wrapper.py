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
import chodby_trans.job as job

import xarray as xr
import zarr

from dask.distributed import Lock

import chodby_trans.exception_wrapper as exp

class Wrapper:

    def __init__(self, cfg: common.dotdict):

        self._config = cfg
        self.sample_dir = Path(".")
        self.sample_output_dir = "output"

    def get_observations(self, tags, param_dict):
        t = time.time()
        logging.info(f"transport_wrapper: get observations tags={tags}")
        cfg = self._config

        def print_exp(_e: Exception, _tags: list):
            sys.stdout.write("-\n" * 60)
            sys.stdout.write(f"Traceback sample tags:{_tags}\n")
            sys.stdout.write(f"transport_wrapper failed with exception: {_e}\n")
            traceback.print_exc()
            sys.stdout.write("-" * 60)
            sys.stdout.flush()

        try:
            if cfg.test_random_data:
                # test random results
                times = output_times(cfg.transport_fullscale)
                ng = 20
                slice_array = np.random.rand(len(times), ng, ng, 2)
                rc = 42
            else:
                rc, slice_array = transport.transport_run(
                    self._config, 
                    tags, param_dict)

        except Exception as e:
            print_exp(e, tags)
            slice_array = np.array([])

            if isinstance(e, exp.WrapperException):
                rc = e.code
            else:
                rc = exp.ReturnCode.UNKNOWN_ERROR

        sample_time = time.time() - t
        logging.info(f"SIMULATION TIME: {sample_time}")
        logging.info(f"slice_array: max {np.max(slice_array)}")

        try:
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
            store_path = str(job.output.zarr_store_path)
            ds = xr.open_zarr(store_path, consolidated=False)

            i_eval_idx = tags[0]
            isample_idx = tags[1]
            isaltelli_idx = tags[2]
            region = {'i_sample': slice(isample_idx, isample_idx + 1),
                      'i_saltelli': slice(isaltelli_idx, isaltelli_idx + 1)}

            # Validate slice_array shape
            expected_shape = (ds.sizes['sim_time'], ds.sizes['X'], ds.sizes['Y'], ds.sizes['Z'])
            if slice_array.shape != expected_shape:
                # raise ValueError(f"slice_array must have shape {expected_shape}, got {slice_array.shape}")
                slice_array = np.zeros(expected_shape)
                logging.info(f"sample {i_eval_idx} return code {rc} => create empty slice for zarr")

            # Lock for every chunk being accessed by the current write
            lock_names = []
            for var, chunkshape in ds.chunksizes.items():  # 'i_sample', 'i_saltelli'
                for cisample in _chunk_ranges(region['i_sample'], chunkshape[0]):
                    for cisaltelli in _chunk_ranges(region['i_saltelli'], chunkshape[0]):
                            lock_names.append(f"zarr-{var}-{cisample}-{cisaltelli}")
            lock_names = sorted(set(lock_names))  # deterministic ordering avoids deadlocks
            # logging.info("lock_names: {}".format(' '.join(map(str, lock_names))))
            logging.info(f"lock_names: {lock_names}")
            locks = [Lock(name) for name in lock_names]

            for L in locks:
                L.acquire()

            try:
                ds_coords = xr.Dataset({
                    'conc': (['i_sample', 'i_saltelli', 'sim_time', 'X', 'Y', 'Z'],
                              slice_array[np.newaxis, np.newaxis, ...]),
                    'return_code': (['i_sample', 'i_saltelli'], np.array(rc)[np.newaxis, np.newaxis, ...]),
                    'eval_time': (['i_sample', 'i_saltelli'], np.array(sample_time)[np.newaxis, np.newaxis, ...])
                })
                ds_coords.to_zarr(store_path, mode='r+', region=region)
            except Exception as e:
                print_exp(e, tags)
                rc = exp.ReturnCode.ZARR_ERROR
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
            print_exp(e, tags)
            return -1000, slice_array

