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

import matplotlib.pyplot as plt
from scipy.stats import norm

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
from endorse.fullscale_transport import output_times

import xarray as xr
import zarr
import zarr_fuse as zf

import chodby_trans.job as job
import chodby_trans.transport_wrapper as transport_wrapper
from chodby_trans import ot_sa
#from chodby_trans.sa import vector_sa_plot as vsp
from chodby_trans import postprocess as pp


import logging
def setup_logging(name="driver"):
    fmt = f"%(asctime)s [{name}] {socket.gethostname()}:%(process)d %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[logging.StreamHandler(sys.stdout)], force=True)
setup_logging(name="trans")

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



   
def prepare_sampling(cfg: dotdict, seed):
    """
    Clean samplig directory + return samples array.
    """
    
    sensitivity_dir = Path(job.scratch.sensitivity_dir)
    if sensitivity_dir.exists():
        shutil.rmtree(sensitivity_dir)
    sensitivity_dir.mkdir()

    return pp.ot_samples(cfg, seed)

def single_sample(args):
    # sample_dir, data_schema_key, tags, parameters = args
    workdir, data_schema_key, tags, parameters = args
    job.set_workdir(workdir)
    setup_logging(name=f"T{tags[0]}")

    # host = socket.gethostname()
    pid  = os.getpid()

    sensitivity_dir = job.scratch.sensitivity_dir
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
    conf_file = job.input.transport_cfg_path
    cfg = common.config.load_config(str(conf_file))
    cfg["data_schema_key"] = data_schema_key

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
      failed_subdir = sensitivity_dir / "failed_samples"
      failed_subdir.mkdir(mode=0o775, parents=True, exist_ok=True)

      num_dirs = sum(1 for p in failed_subdir.iterdir() if p.is_dir())

      if num_dirs <= 10:
          logging.info(f"Moving failed sample {sample_dir.name}")
          sample_dir.rename(failed_subdir / sample_dir.name)
      else:
        if cfg["ot_sensitivity"]["clean_sample_dir"]:
            shutil.rmtree(sample_dir)
    else:
        if cfg["ot_sensitivity"]["clean_sample_dir"]:
            shutil.rmtree(sample_dir)

    return res


def prepare_sample_args(cfg, seed):
    data_schema_key, data_schema = initialize_data_schema()
    if job.output.zarr_store_path.exists() and cfg.ot_sensitivity.recompute_failed:
        tags, parameters = read_failed_parameters()
    else:
        # parameters = salib_samples(cfg, seed)
        # tags = setup_data_storage(cfg, str(input_data.zarr_store_path), data_schema, parameters)
        input_design = prepare_sampling(cfg, seed)
        parameters = input_design.param_mat
        tags = setup_data_storage(cfg, str(job.output.zarr_store_path), data_schema, input_design)
    # data_schema_key = "run_XXX"

    n_samples, n_params = parameters.shape
    sample_args = [(job.output.dir_path, data_schema_key, tags[idx], parameters[idx]) for idx in range(n_samples)]
    logging.info(f"eval_args:\n{sample_args[:10]}\n... n_evals={len(sample_args)}")
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
    data_schema_path = job.input.data_schema_yaml
    if not data_schema_path.exists():
        shutil.copy2(job.input.data_schema_empty_yaml, data_schema_path)

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
                       input_design):
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
    input_design : np.array
        Input Design
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

    param_names = input_design.param_names
    n_params = len(param_names)
    n_samples = input_design.n_samples
    n_saltelli = input_design.n_saltelli

    parameters = input_design.param_mat
    i_sample, i_saltelli, A_sample = input_design.saltelli_layout

    print("i_sample:   ", i_sample)
    print("i_saltelli: ", i_saltelli)
    # print("A_mask:\n", input_design.A_mask)
    # print("A_sample:\n", A_sample)

    grid_size = data_schema["ATTRS"]["grid_step"]
    otimes = output_times(cfg.transport_fullscale)

    # ALL coords
    coords = data_schema["COORDS"]
    coords_names = list(coords.keys())
    shapes = (n_samples, n_saltelli, n_params, len(otimes), *grid_size)

    # set coords once
    # default coords are set to index their range: 0,1,2,...
    ds_coords = {k: np.arange(v) for k, v in zip(coords_names, shapes)}
    ds_coords['sim_time'] = otimes          # actual simulation times
    ds_coords['param_name'] = param_names   # actual parameter names
    # 'X', 'Y', 'Z' -> indices in the regular grid axes

    # concentration - prepare empty
    conc_coords = data_schema['VARS']['conc']['coords']
    conc_shapes = (n_samples, n_saltelli, len(otimes), *grid_size)
    conc_chunks = [coords[c]["chunk_size"] for c in coords_names if c in conc_coords]

    # i_eval
    sid_coords = data_schema['VARS']['i_eval']['coords']
    sid_chunks = [coords[c]["chunk_size"] for c in coords_names if c in sid_coords]
    sid_matrix = np.full((n_samples, n_saltelli), -1, dtype=int)  # or (U, V) if your ranges are [0, U) and [0, V)
    sid_matrix[i_sample, i_saltelli] = np.arange(input_design.n_evals)
    res_shapes = (n_samples, n_saltelli)
    print(f"i_eval_matrix shape: {sid_matrix.shape}, coords: {sid_coords}")
    print("i_eval_matrix:\n", sid_matrix)

    # parameters
    par_coords = data_schema['VARS']['parameter']['coords']
    par_chunks = [coords[c]["chunk_size"] for c in coords_names if c in par_coords]
    par_matrix = np.zeros((n_samples, n_saltelli, n_params), dtype=parameters.dtype)
    par_matrix[i_sample, i_saltelli, :] = parameters
    print(f"par_matrix shape: {par_matrix.shape}, coords: {par_coords}")
    print("par_matrix:\n", par_matrix)

    # A_sample
    # A_coords = data_schema['VARS']['A_sample']['coords']
    # A_chunks = [coords[c]["chunk_size"] for c in coords_names if c in A_coords]
    # A_matrix = np.zeros((n_blocks, n_qmc, n_params), dtype=parameters.dtype)
    # # A_matrix[:, :, :] = A_sample[None, :, :]
    # print(f"A_matrix shape: {A_matrix.shape}, coords: {A_coords}")
    # print("A_matrix:\n", A_matrix)

    ds = xr.Dataset(
        data_vars={
            'i_eval': (tuple(sid_coords), da.from_array(sid_matrix, chunks=sid_chunks)),
            'eval_time': (tuple(sid_coords), da.full(res_shapes, fill_value=-1, chunks=sid_chunks, dtype=float)),
            'return_code': (tuple(sid_coords), da.full(res_shapes, fill_value=-2000, chunks=sid_chunks, dtype=int)),
            'conc': (tuple(conc_coords), da.zeros(conc_shapes, chunks=conc_chunks)),
            'parameter': (tuple(par_coords), da.from_array(par_matrix, chunks=par_chunks)),
            # 'A_sample': (tuple(A_coords), da.from_array(A_matrix, chunks=A_chunks)),
        },
        coords=ds_coords
    )

    # Write to Zarr, overwrite if exists
    ds.to_zarr(store_path, mode='w')

    print("=========== READ ZARR ==============")
    print("Control read of created Zarr storage")
    read_ds = xr.open_zarr(store_path, consolidated=False)
    print(read_ds)
    # print(read_ds['A_sample'].to_numpy())
    print("=========== END READ ZARR ==============")

    tags = np.column_stack((range(input_design.n_evals), i_sample, i_saltelli))
    return tags


def read_failed_parameters():

    print("=========== READ ZARR ==============")
    ds = xr.open_zarr(str(job.output.zarr_store_path))
    print(ds)
    # print(read_ds['A_sample'].to_numpy())
    # print("sample_id:\n", ds['sample_id'].to_numpy())
    # print(ds['parameter'].to_numpy())
    # print("return_code:\n", ds['return_code'].to_numpy())
    print("=========== END READ ZARR ==============")
    logging.info("plotting eval time histogram...")
    plot_sample_time_hist(ds['eval_time'].to_numpy().ravel())

    logging.info("getting failed samples...")
    v_param = ds['parameter'].to_numpy()
    v_time = ds['eval_time'].to_numpy()
    v_ieval = ds['i_eval'].to_numpy()
    v_rc = ds['return_code'].to_numpy()

    mask = v_rc < 0
    f_param = v_param[mask]
    f_ieval = v_ieval[mask]

    i_idx, q_idx = np.where(mask)  # integer indices
    f_isample = ds['i_sample'].isel(i_sample=i_idx).to_numpy()  # coordinate values of i_sample
    f_isaltelli = ds['i_saltelli'].isel(i_saltelli=q_idx).to_numpy()  # coordinate values of i_saltelli

    tags = np.column_stack((f_ieval, f_isample, f_isaltelli))
    return tags, f_param

def plot_sample_time_hist(st):
    upper_limit = 2000
    n_removed = np.sum(st > upper_limit)
    print(f"count st > {upper_limit}: {n_removed}")
    cst = st[st <= upper_limit]

    cst_std = cst.std()
    cst_mean = cst.mean()
    cst_median = np.median(cst)
    cst_q95 = np.quantile(cst, 0.95)
    n_bins = 50

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(cst, bins=n_bins)
    for v, label, style in [(cst_mean, f"Mean [{int(cst_mean)}]", "-"),
                            (cst_median, f"Median [{int(cst_median)}]", "--"),
                            (cst_q95, f"95th pct [{int(cst_q95)}]", ":")]:
        if np.isfinite(v):
            ax.axvline(v, linestyle=style, linewidth=1, label=label, color="k")
    # ax.plot(x_axis, norm.pdf(x_axis, cst_mean, cst_std))
    ax.text(0.7, 0.65,
            f"st > {upper_limit}: {n_removed}",
            transform=plt.gca().transAxes,
            ha="left", va="top",
            fontsize=12, bbox=dict(facecolor="yellow", alpha=0.7, edgecolor="b"))
    ax.set_title('Sample time histogram')
    ax.set_ylabel('count')
    ax.legend()
    fig.tight_layout()
    fig.savefig(job.output.plots / 'sample_time_hist.pdf', format='pdf')


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
    pbs_path = job.output.pbs_script
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
        workdir=job.scratch.dir_path,
        outputdir=job.output.dir_path,
        out_log=(job.output.dir_path / (cfg_pbs.pbs_name + '.out'))
    )
    print(parameters['python'])
    pbs_script = pbs_script_template.format(**parameters)
    with open(pbs_path, "w") as f:
        f.write(pbs_script)

    cmd = ['qsub', pbs_path]
    logging.info(f"submit pbs: '{cmd}'")
    subprocess.run(cmd, check=True)

#@memoize
def compute_raw_sobol(cfg, seed):
    input_design = prepare_sampling(cfg, seed)
    store_path = str(job.output.zarr_store_path)
    ds = xr.open_zarr(store_path, consolidated=True)
    rc = ds['return_code'].to_numpy()
    print('Return code:\n', rc)
    print(f"Number of failed runs: {(rc < 0).sum()} / {rc.size}")
    valid_isample = (ds['return_code'] >= 0).all(dim='i_saltelli').to_numpy()  # mask failed runs
    n_valid = int(valid_isample.sum())
    print(f"Valid i_sample: {n_valid} ")
    conc = ds['conc'].isel(sim_time=slice(1, None), i_sample=valid_isample)  # skip t=0
    conc_max_space = conc.quantile(0.99, dim=('X', 'Y', 'Z')).compute()
    print("Max concentration (99% quantile) over space and time:\n", conc_max_space.to_numpy())
    conc_mean = conc.mean(dim=('i_sample', 'i_saltelli'))
    conc_vars = conc.var(dim=('i_sample', 'i_saltelli')).compute()
    assert np.all(conc_vars > 0.0), f"Some concentration variance is zero: {np.sum(conc_vars == 0.0)}"
    center_conc = conc - conc_mean
    conc_2D = center_conc.stack(sample=('i_sample', 'i_saltelli'), output=('sim_time', 'X', 'Y', 'Z'))
    
    si_ds = input_design.compute_sobol(conc_2D.transpose('sample', 'output').compute())
    si_ds = si_ds.unstack('output')  
    assert 'sim_time' in si_ds.dims, "Expected spatial dimensions in the output"
    return si_ds, conc_vars

def make_plots(cfg, seed):
    si_ds, conc_vars = compute_raw_sobol(cfg, seed)

    si_space_agg = si_ds.copy()
    space_dims = ('X', 'Y', 'Z')
    for var in ['S1', 'ST', 'S2']:
        si_space_agg[var] = si_space_agg[var].weighted(conc_vars).mean(dim=space_dims)
    for d in space_dims:
        si_space_agg = si_space_agg.drop_vars(d)

    filtered_ds = vsp.select_sobol_terms_with_others(si_space_agg, 0.9, output_dim='sim_time', si_threshold=0.05)
    vsp.plot_sobol_stacked(filtered_ds, figsize=(10, 5), out_path=job.output.plots / "sobol_stacked.pdf")


def main():
    # common.EndorseCache.instance().expire_all()

    if len(sys.argv) == 3:
        work_dir = Path(sys.argv[1]).absolute()
        cmd = sys.argv[2]
    elif len(sys.argv) == 4:
        work_dir = Path(sys.argv[1]).absolute()
        cmd = sys.argv[2]
        scheduler = sys.argv[3]
    else:
      sys.exit("Provide <workdir> <command: (submit|local|meta|plots|read)> <command_args>.")


    # resolve job dirs
    job.set_workdir(work_dir)
    if cmd == 'submit' or cmd == 'local':
        if job.input.dir_path.exists():
            while True:
                user_input = input("Do you want to rewrite INPUT DATA? (yes/no): ")
                if user_input.lower() in ["yes", "y"]:
                    print("Continuing...")
                    break
                elif user_input.lower() in ["no", "n"]:
                    print("Exiting...")
                    break
                else:
                    print("Invalid input. Please enter yes/no.")
        else:
            shutil.copytree(script_path.parent / job.input.dir_path.name, job.input.dir_path, dirs_exist_ok=True)

    logging.info(job.to_str())
    if not job.input.dir_path.exists():
        raise Exception(f"Input data '{job.input.dir_path}' not found in workdir '{work_dir}'")

    cfg_path = job.input.transport_cfg_path
    cfg = common.config.load_config(str(cfg_path))

    seed = 101

    if not job.output.plots.exists():
        job.output.plots.mkdir()

    if cmd == 'submit':
        submit_pbs(cfg)
    elif cmd == 'read':
        # zarr_path = sys.argv[2]
        read_failed_parameters()
    elif cmd == 'meta' or cmd == 'local':
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
    elif cmd == 'plots':
        with common.workdir(str(job.output.plots), clean=False):
            pp.make_transport_plots(cfg, seed)

    else:
        sys.exit(f"Unkown command provided: '{cmd}'!")

if __name__ == '__main__':
    main()
