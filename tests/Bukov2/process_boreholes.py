import sys
import subprocess
from scoop import futures
import logging
import h5py
import numpy as np
from pathlib import Path
from endorse.Bukov2.sample_storage import dataset_name, failed_ids_name, done_ids_name
from endorse import common
from endorse.Bukov2 import plot_boreholes, sa_problem, sobol_fast, boreholes, bh_chambers, bukov_common as bcommon
from endorse.Bukov2.bukov_common import memoize, file_result
params_name="parameters"
from pathlib import Path
script_path = Path(__file__).absolute()

def single_borehole(args):
    workdir, i_bh = args
    return _single_borehole(workdir, i_bh)

@bcommon.memoize
def _single_borehole(bh_workdir, i_bh):
    workdir, cfg = bcommon.load_cfg(bh_workdir.parent / "Bukov2_mesh.yaml")
    bh_set = boreholes.make_borehole_set(workdir, cfg)

    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    sobol_fn = sobol_fast.vec_sobol_total_only
    chambers = bh_chambers.Chambers.from_bh_set(workdir, cfg, bh_set, i_bh, problem, sobol_fn)

    #best_packer_configs = bh_chambers.optimize_packers(cfg, chambers)
    
    # shuld not be necessary as the whole funcion result is memoized
    #bcommon.pkl_write(bh_workdir, best_packer_configs, "best_packer_configs.pkl")

    param_names = list(problem['names'])
    plots = plot_boreholes.PlotCfg(
        bh_workdir, cfg, bh_set, chambers,
        i_bh, param_names, show=False)
    plots.all()
    
    return best_packer_configs

def borehole_dir(workdir, i_bh):
    bh_dir = workdir / f"process_bh_{i_bh:03d}" 
    bh_dir.mkdir(parents=True, exist_ok=True)    
    return bh_dir

def all_boreholes(workdir):
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    n_boreholes = (0,50,5)
    #n_boreholes = (bh_set.n_boreholes,)
    bh_args = [(borehole_dir(workdir, i_bh), i_bh) for i_bh in range(*n_boreholes)]

    results = list(futures.map(single_borehole, bh_args))
    bcommon.pkl_write(workdir, results, "all_bh_configs.pkl")
    
pbs_script_template = """
#!/bin/bash
#PBS -j oe
#PBS -m e
set -x 
env | grep PBS_
echo "===="
#{python} -m scoop --hostfile $PBS_NODEFILE -vv -n {n_workers} {script_path} {workdir} 
{python} {script_path} {workdir} 23
"""
def submit_pbs(workdir):
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    n_boreholes = bh_set.n_boreholes    
    
    n_workers = min(n_boreholes + 1, cfg.pbs.n_workers)   # Not sure if we need reserve for the master scoop process
    queue = cfg.pbs.queue
    pbs_filename = workdir / "process_boreholes.pbs"
    parameters = dict(
        python=sys.executable,
        n_workers=n_workers - 1,
        script_path=script_path,
        workdir=workdir
    )
    print(parameters['python'])
    pbs_script = pbs_script_template.format(**parameters)
    with open(pbs_filename, "w") as f:
        f.write(pbs_script)
    cmd = ['qsub', '-q', queue, '-l', f'select={str(n_workers)}:ncpus=1', pbs_filename]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    workdir = Path(sys.argv[1]).absolute()
    if len(sys.argv) > 2 and (sys.argv[2] == 'submit'):
        submit_pbs(workdir)
    elif len(sys.argv) > 2:
        i_bh = int(sys.argv[2])
        single_borehole( (borehole_dir(workdir, i_bh), i_bh) )
    else:
        all_boreholes(workdir)
