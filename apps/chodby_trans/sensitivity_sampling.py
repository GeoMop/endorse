import shutil
from typing import *
import csv
import os, sys, socket
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import copy
import subprocess
import json

import yaml
# from scoop import futures
# from mpi4py.futures import MPIPoolExecutor

# ---- Dask (over TCP) ----
from dask.distributed import Client, wait
import dask.array as da

import numpy as np

from endorse import common
from endorse.common import dotdict, File, report, memoize
# from endorse.sa import sample, analyze
import endorse.sa as sa

import xarray as xr
import zarr
import zarr_fuse as zf

# import chodby_trans.sample_storage as sample_storage
import chodby_trans.input_data as input_data
import chodby_trans.transport_wrapper as transport_wrapper

# from worker_logging import submit_logged, map_logged, setup_worker_logging

import logging
def setup_logging(name="driver"):
    fmt = f"%(asctime)s [{name}] {socket.gethostname()}:%(process)d %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[logging.StreamHandler(sys.stdout)], force=True)
setup_logging(name="trans")


input_dir = input_data.input_dir
work_dir = input_data.work_dir
output_dir = input_data.work_dir
script_path = Path(__file__).absolute()


def solver_id(i):
    return str(i).zfill(2)

def sampled_data_hdf(i):
    return 'sampled_data_' + solver_id(i) + '.h5'

def atomic_write_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


@memoize
def salib_samples(cfg: dotdict, seed):
    cfg_sens = cfg.sensitivity
    # Define the problem for SALib
    # Bayes Inversion borehole_V1/sim_A04hm_V1_04_20230713a
    problem = sa.sample.prepare_problem_defition(cfg_sens.parameters)
    print(problem)

    # Generate Saltelli samples
    # param_values = sa.sample.saltelli(problem, cfg_sens.n_samples, calc_second_order=cfg_sens.second_order_sa)
    param_values = sa.sample.sobol(problem, cfg_sens.n_samples, calc_second_order=cfg_sens.second_order_sa)
    print(param_values.shape)

    # # plot requires LaTeX installed
    # plot_conductivity(cfg, param_values)
    # # exit(0)

    sensitivity_dir = Path(input_data.sensitivity_dirname)
    if sensitivity_dir.exists():
        shutil.rmtree(sensitivity_dir)
    sensitivity_dir.mkdir()

    return param_values
    # TODO: write parameters
    # N = param_values.shape[0]  # number of rows
    # row_idx = np.arange(N).reshape(-1, 1)  # column vector [[0], [1], …]
    # params_with_idx = np.hstack((row_idx, param_values))
    # return params_with_idx

    # plan sample parameters a prepare them in CSV
    # prepare_sets_of_params(cfg_sens, param_values, sensitivity_dir, problem["names"])
    # exit(0)

    # plan parallel sampling, prepare PBS jobs
    # if cfg.machine_config.pbs is not None:
    #     pbs_file_list = prepare_pbs_scripts(cfg, sensitivity_dir)
    # else:
    #     pass


def single_sample(args):
    # sample_dir, data_schema_key, tags, parameters = args
    data_schema_key, tags, parameters = args

    # host = socket.gethostname()
    pid  = os.getpid()
    # logging.info(f"[worker {pid} on {host}] starting with tags={tags}", flush=True)

    # create sample dir
    # do we have to do this independently or is the "import" done in each process??
    # scoop required total independency
    workdir, inputdir, outputdir = input_data.resolve_dirs()

    sensitivity_dir = workdir / input_data.sensitivity_dirname
    sample_subdir = sensitivity_dir / "samples"

    flag_file = sample_subdir / f"sample_{str(tags[0]).zfill(3)}.done.json"
    if flag_file.exists():
        # Already finished—return cached result
        with open(flag_file) as f:
            logging.info(f"SAMPLE already done: {flag_file}")
            return json.load(f)["res"]

    sample_dir = sample_subdir / (f"sample_{str(tags[0]).zfill(3)}_{pid}")
    sample_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    # read config file
    conf_file = inputdir / input_data.transport_cfg_path.name
    cfg = common.config.load_config(str(conf_file))
    cfg["data_schema_key"] = data_schema_key
    cfg["input_dir"] = inputdir

    logging.info("=========================== RUNNING CALCULATION " +
                 "sample {} ===========================".format(tags[0]).zfill(3))
    logging.info(f"tags={tags}, parameters={parameters}, sample_dir={str(sample_dir)}")

    wrap = transport_wrapper.Wrapper(cfg=cfg)
    with common.workdir(str(sample_dir), clean=False):
        res, sample_data = wrap.get_observations(tags, parameters)
        logging.info(f"Flow123d res: {res, np.shape(sample_data)}")
        # print("LEN:", len(obs_data))
    
    result = {"res": res, "sample": str(sample_dir), "ts": time.time()}
    atomic_write_json(flag_file, result)
    # tmp =  Path(f"{flag_file}.tmp")
    # with open(tmp, "w") as f: json.dump(result, f)
    # os.replace(tmp, flag_file)   # atomic rename

    if res < 0:
      logging.info(f"Moving failed sample {sample_dir.name}")
      failed_subdir = sensitivity_dir / "failed_samples"
      failed_subdir.mkdir(mode=0o775, parents=True, exist_ok=True)
      sample_dir.rename(failed_subdir / sample_dir.name)
    else:
        if cfg["sensitivity"]["clean_sample_dir"]:
            shutil.rmtree(sample_dir)

    return res


def prepare_sample_args(cfg, seed):
    data_schema_key, data_schema = initialize_data_schema()
    if input_data.zarr_store_path.exists() and cfg.sensitivity.recompute_failed:
        tags, parameters = read_failed_parameters()
    else:
        parameters = salib_samples(cfg, seed)
        tags = setup_data_storage(cfg, str(input_data.zarr_store_path), data_schema, parameters)
    # data_schema_key = "run_XXX"

    n_samples, n_params = parameters.shape
    sample_args = [(data_schema_key, tags[idx], parameters[idx]) for idx in range(n_samples)]
    logging.info(f"sample_args: {sample_args}")
    return sample_args


def all_samples(cfg, sample_args, client=None):
    # Set directories to avoid NFS IO errors

    # # create sample dir
    # sensitivity_dir = workdir / input_data.sensitivity_dirname
    # sample_subdir = sensitivity_dir / "samples"

    # OPTION A: submit pattern
    # futs = [submit_logged(futures, single_sample, a, label="fem") for a in args]
    # results = [f.result() for f in futs]
    # return results

    # OPTION B: map pattern
    # return list(map_logged(futures, single_sample, args, label="fem"))

    # mapped = map_fn(single_sample, bh_args)  # iterator over results
    # mapped = map_logged(futures, single_sample, bh_args, label="trans")
    # results = [r for r in mapped]  # forces full consumption
    # results = list(map_fn(single_sample, bh_args))
    # results = [f.result() for f in scoop_list]  # consume all

    # with MPIPoolExecutor() as ex:
        # results = list(ex.map(single_sample, bh_args))

    # Dask
    t0 = time.time()    
    futures = client.map(single_sample, sample_args, pure=False)
    results = client.gather(futures)
    logging.info("Completed %d tasks in %.2fs", len(results), time.time() - t0)
    logging.info(f"Results collected: {results[:100]}")
    # bcommon.pkl_write(workdir, results, "sample_results.pkl")
    # zarr.consolidate_metadata(str(input_data.zarr_store_path))


def initialize_data_schema():
    # add current scheme for current sampling run
    data_schema_path = input_data.data_schema_yaml
    if not data_schema_path.exists():
        shutil.copy2(input_data.data_schema_empty_yaml, data_schema_path)

    with data_schema_path.open("r", encoding="utf-8") as file:
        content = file.read()
        data_schema = yaml.safe_load(content)

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    data_schema_key = f"run_{now}"
    data_schema[data_schema_key] = copy.deepcopy(data_schema["run_timestamp"])
    # data_schema.pop("run_timestamp", None)

    with data_schema_path.open("w", encoding="utf-8") as file:
        yaml.dump(data_schema, file, sort_keys=False)

    return data_schema_key, data_schema[data_schema_key]


def setup_data_storage(cfg: dotdict,
                       store_path: str,
                       data_schema: dict,
                       parameters):
    """
    Initialize an empty Zarr store with the specified dimensions and chunking.

    Parameters
    ----------
    cfg : dotdict
        Configuration
    store_path : str
        Path to the Zarr store (directory or Zarr URL).
    data_schema : dict
        Data schema (zarr fuse alike)
    parameters : np.array
        Parameter matrix NxP
    """
    # prepare data scheme for zarr storage

    # kwargs =  {"WORKDIR": str(workdir), "S3_ENDPOINT_URL": "https://s3.cl4.du.cesnet.cz"}
     # Local storage logic
    # store_path = (workdir / fname).with_suffix(".zarr")
    # ZARR FUSE
    # kwargs =  {"WORKDIR": str(input_data.zarr_store_path), "STORE_URL": str(input_data.zarr_store_path)}
    # data_schema = zf.schema.deserialize(data_schema_path) # read data scheme
    # zf.remove_store(data_schema, **kwargs)      # start from scratch
    # node = zf.open_store(data_schema, **kwargs) # initialize zarr_fuse storage

    # DIRECT ZARR
    # temporary shortcut for direct zarr

    cfg_sens = cfg.sensitivity
    n_samples, n_params = parameters.shape
    n_qmc = cfg_sens.n_samples
    n_blocks = sa.sample._num_blocks(n_params, second_order=cfg.sensitivity.second_order_sa)

    print(f"N={n_qmc}, D={n_params}")
    print(f"n_blocks={n_blocks}")
    print(f"n_samples (N*n_blocks)={n_samples}")
    assert n_samples == n_qmc*n_blocks

    # saltelli_base_idx = sa.sample.saltelli_base_idx(N=cfg_sens.n_samples,
    #                                                 D=n_params,
    #                                                 second_order=cfg_sens.second_order_sa)
    # print("saltelli_base_idx:  ", saltelli_base_idx)
    qmc = sa.sample.saltelli_qmc_idx(N=cfg_sens.n_samples,
                                     D=n_params,
                                     second_order=cfg_sens.second_order_sa)
    print("saltelli_qmc_idx:   ", qmc)
    block = sa.sample.saltelli_block_idx(N=cfg_sens.n_samples,
                                         D=n_params,
                                         second_order=cfg_sens.second_order_sa)
    print("saltelli_block_idx: ", block)
    A_sample = sa.sample.saltelli_ab_mask(N=cfg_sens.n_samples,
                                          D=n_params,
                                          second_order=cfg_sens.second_order_sa)
    print("A_sample_idx:\n", A_sample)

    param_names = [p.name for p in cfg_sens.parameters]
    grid_size = data_schema["ATTRS"]["grid_step"]

    from endorse.fullscale_transport import output_times
    otimes = output_times(cfg.transport_fullscale)

    # ALL coords
    coords = data_schema["COORDS"]
    coords_names = list(coords.keys())
    shapes = (n_blocks, n_qmc, n_params, len(otimes), *grid_size)

    # set coords once
    # default coords are set to index their range: 0,1,2,...
    ds_coords = {k: np.arange(v) for k, v in zip(coords_names, shapes)}
    ds_coords['sim_time'] = otimes          # actual simulation times
    ds_coords['param_name'] = param_names   # actual parameter names
    # 'X', 'Y', 'Z' -> indices in the regular grid axes

    # concentration - prepare empty
    conc_coords = data_schema['VARS']['conc']['coords']
    conc_shapes = (n_blocks, n_qmc, len(otimes), *grid_size)
    conc_chunks = [coords[c]["chunk_size"] for c in coords_names if c in conc_coords]

    # sample_id
    sid_coords = data_schema['VARS']['sample_id']['coords']
    sid_chunks = [coords[c]["chunk_size"] for c in coords_names if c in sid_coords]
    sid_matrix = np.full((n_blocks, n_qmc), -1, dtype=int)  # or (U, V) if your ranges are [0, U) and [0, V)
    sid_matrix[block, qmc] = np.arange(n_samples)
    res_shapes = (n_blocks, n_qmc)
    print(f"sid_matrix shape: {sid_matrix.shape}, coords: {sid_coords}")
    print("sid_matrix:\n", sid_matrix)

    # parameters
    par_coords = data_schema['VARS']['parameter']['coords']
    par_chunks = [coords[c]["chunk_size"] for c in coords_names if c in par_coords]
    par_matrix = np.zeros((n_blocks, n_qmc, n_params), dtype=parameters.dtype)
    par_matrix[block, qmc, :] = parameters
    print(f"par_matrix shape: {par_matrix.shape}, coords: {par_coords}")
    print("par_matrix:\n", par_matrix)

    # A_sample
    A_coords = data_schema['VARS']['A_sample']['coords']
    A_chunks = [coords[c]["chunk_size"] for c in coords_names if c in A_coords]
    A_matrix = np.zeros((n_blocks, n_qmc, n_params), dtype=parameters.dtype)
    A_matrix[block, qmc, :] = A_sample
    print(f"A_matrix shape: {A_matrix.shape}, coords: {A_coords}")
    print("A_matrix:\n", A_matrix)

    ds = xr.Dataset(
        data_vars={
            'sample_id': (tuple(sid_coords), da.from_array(sid_matrix, chunks=sid_chunks)),
            'sample_time': (tuple(sid_coords), da.zeros(res_shapes, chunks=sid_chunks, dtype=int)),
            'return_code': (tuple(sid_coords), da.zeros(res_shapes, chunks=sid_chunks, dtype=int)),
            'conc': (tuple(conc_coords), da.zeros(conc_shapes, chunks=conc_chunks)),
            'parameter': (tuple(par_coords), da.from_array(par_matrix, chunks=par_chunks)),
            'A_sample': (tuple(A_coords), da.from_array(A_matrix, chunks=A_chunks)),
        },
        coords=ds_coords
    )

    # Write to Zarr, overwrite if exists
    ds.to_zarr(store_path, mode='w')

    print("=========== READ ZARR ==============")
    print("Control read of created Zarr storage")
    read_ds = xr.open_zarr(store_path)
    print(read_ds)
    # print(read_ds['A_sample'].to_numpy())
    print("=========== END READ ZARR ==============")

    tags = np.column_stack((range(n_samples), block, qmc))
    return tags


def read_failed_parameters():

    print("=========== READ ZARR ==============")
    ds = xr.open_zarr(str(input_data.zarr_store_path))
    print(ds)
    # print(read_ds['A_sample'].to_numpy())
    # print("sample_id:\n", ds['sample_id'].to_numpy())
    # print(ds['parameter'].to_numpy())
    # print("return_code:\n", ds['return_code'].to_numpy())
    print("=========== END READ ZARR ==============")

    # n_samples = ds['sample_time'].max()
    # mask = np.isclose(ds['sample_time'].isel(iid=slice(7)).to_numpy(), 0, rtol=0, atol=1e-12)
    # params = ds['parameter'].to_numpy()[mask]  # 1D np.ndarray of matches

    # subset to the first 7 along iid ("rows")
    param_sub = ds['parameter'].isel(iid=slice(6))
    time_sub = ds['sample_time'].isel(iid=slice(6))
    sample_id_sub = ds['sample_id'].isel(iid=slice(6))

    mask = np.isclose(time_sub.to_numpy(), 0, rtol=0, atol=1e-12)

    param_vec = param_sub.to_numpy()[mask]
    sample_id_vec = sample_id_sub.to_numpy()[mask]

    i_idx, q_idx = np.where(mask)  # integer indices
    iid_vec = ds['iid'].isel(iid=i_idx).to_numpy()  # coordinate values of iid
    qmc_vec = ds['qmc'].isel(qmc=q_idx).to_numpy()  # coordinate values of qmc

    tags = np.column_stack((sample_id_vec, iid_vec, qmc_vec))
    return tags, param_vec


pbs_script_template = """
#!/bin/bash
#PBS -S /bin/bash
#PBS -l select={n_chunks}:ncpus={n_cpus}:mem={mem}:scratch_local=10gb
#PBS -l place=scatter
#PBS -l walltime={walltime}
#PBS -q {queue}
#PBS -N {job_name}
#PBS -o {out_log}
#PBS -j oe
#PBS -m e

set -x
env | grep PBS_
export TMPDIR=$SCRATCHDIR

output_dir={outputdir}
# work_dir={workdir}

SCRIPT_PATH={script_path}
PROJECT_DIR=$(dirname -- $SCRIPT_PATH)
WORK_DIRNAME="workdir"

# # A node list with each host once
UNIQ_HOSTS=$(sort -u "$PBS_NODEFILE")
NCPUS=$(wc -l < "$PBS_NODEFILE")

# copy to scratch
echo "Staging data to scratch ..."
# copy to scratchdir on all nodes (unique line in pbs nodefile)
for node in $UNIQ_HOSTS; do
    pbsdsh -vh "$node" -- rsync -a --delete "$output_dir/" "$SCRATCHDIR/" &
done
wait

bash $PROJECT_DIR/dask_cluster.sh start

cd "$SCRATCHDIR"
echo "START SAMPLING"
bash $PROJECT_DIR/dask_cluster.sh run
echo "FINISHED SAMPLING"

echo "Copying results back ..."
uname -n
for node in $UNIQ_HOSTS; do
    pbsdsh -vh "$node" -- rsync -a "$SCRATCHDIR/$WORK_DIRNAME/" "$output_dir/workdir_$node" &
done
wait

bash $PROJECT_DIR/dask_cluster.sh stop

# just dry-run while compressing logs and failed samples
bash $PROJECT_DIR/cleanup_workdir.sh $output_dir

clean_scratch
echo "FINISHED"
"""

def submit_pbs(cfg):
    cfg_pbs = cfg.machine_config.pbs
    # n_workers = min(n_boreholes + 1, cfg.pbs.n_workers)
    pbs_path = output_dir / "sensitivity_sampling.pbs"
    n_workers = int(cfg_pbs.n_nodes * (cfg_pbs.n_cores-1)-2)# Not sure if we need reserve for the master scoop process
    parameters = dict(
        n_chunks=cfg_pbs.n_nodes,
        n_cpus=cfg_pbs.n_cores,
        queue=cfg_pbs.queue,
        mem=cfg_pbs.mem,
        walltime=cfg_pbs.walltime,
        job_name=cfg_pbs.pbs_name,

        python=sys.executable,
        n_workers=n_workers,
        script_path=script_path,
        workdir=work_dir,
        outputdir=output_dir,
        out_log=(output_dir / (cfg_pbs.pbs_name + '.out'))
    )
    print(parameters['python'])
    pbs_script = pbs_script_template.format(**parameters)
    with open(pbs_path, "w") as f:
        f.write(pbs_script)

    cmd = ['qsub', pbs_path]
    logging.info(f"submit pbs: '{cmd}'")
    subprocess.run(cmd, check=True)


def main():
    # common.EndorseCache.instance().expire_all()
    # setup_worker_logging(name="trans")

    if len(sys.argv) == 2:
        cmd = sys.argv[1]
    elif len(sys.argv) == 3:
        cmd = sys.argv[1]
        scheduler = sys.argv[2]
    # elif len(sys.argv) == 3:
    #     cmd = sys.argv[1]
    #     work_dir = Path(sys.argv[2]).absolute()
    else:
      sys.exit("Provide command (submit|meta|).")

    logging.info(f"main: work_dir = {work_dir}")
    logging.info(f"main: input_dir = {input_dir}")
    
    cfg_path = input_data.transport_cfg_path
    cfg = common.config.load_config(str(cfg_path))

    seed = 101

    if cmd == 'submit':
        # with common.workdir(str(work_dir), clean=False):
        shutil.copytree(input_dir, output_dir / input_dir.name, dirs_exist_ok=True)
        submit_pbs(cfg)
    # elif cmd == 'local':
    #     with common.workdir(str(work_dir), clean=False):
    #         shutil.copytree(input_dir, work_dir / input_dir.name, dirs_exist_ok=True)
    #         parameters = salib_samples(cfg, seed)
    #         all_samples(cfg=cfg, parameters=parameters)
    elif cmd == 'read':
        # zarr_path = sys.argv[2]
        read_failed_parameters()
    elif cmd == 'meta':
        # optional: cap hidden threading for your FEM libs
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        
        client = Client(scheduler)
        logging.info(f"Connected to: {client}")
        
        with common.workdir(str(work_dir), clean=False):
            sample_args = prepare_sample_args(cfg, seed)
            all_samples(cfg=cfg, sample_args=sample_args, client=client)
    else:
        sys.exit(f"Unkown command provided: '{cmd}'!")

if __name__ == '__main__':
    main()
