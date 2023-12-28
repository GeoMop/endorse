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
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    bh_set = boreholes.make_borehole_set(workdir, cfg)

    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    sobol_fn = sobol_fast.vec_sobol_total_only
    chambers = bh_chambers.Chambers.from_bh_set(workdir, cfg, bh_set, i_bh, problem, sobol_fn)

    best_packer_configs = bh_chambers.optimize_packers(cfg, chambers)
    bcommon.pkl_write(workdir, best_packer_configs, "best_packer_configs.pkl")

    param_names = list(problem['names'])
    bh_workdir = workdir / f"plot_bh_{i_bh:03d}"
    bh_workdir.mkdir(parents=True, exist_ok=True)
    plots = plot_boreholes.PlotCfg(
        bh_workdir, cfg, bh_set, chambers,
        i_bh, param_names, show=False)
    plots.all()


def all_boreholes(workdir):
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    bh_args = [(workdir, i_bh) for i_bh in range(bh_set.n_boreholes)]

    results = futures.map(single_borehole, bh_args)
    bcommon.pkl_write(workdir, results, "all_bh_configs.pkl")
    
pbs_script_template = """
#!/bin/bash
#PBS -j oe
#PBS -m e
set -x 
env | grep PBS_
echo "===="
{sys.executable} -m scoop --hostfile $PBS_NODEFILE -vv -n 4 {script_path} {workdir} 
"""
def submit_pbs(workdir):
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    n_workers = cfg.pbs.n_workers
    queue = cfg.pbs.queue
    pbs_filename = workdir / "process_boreholes.pbs"
    parameters = dict(
        python=sys.executable,
        n_workers=n_workers,
        script_path=script_path,
        workdir=workdir
    )
    pbs_script = pbs_script_template.format(parameters)
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
        single_borehole( (workdir, i_bh) )
    else:
        all_boreholes(workdir)
