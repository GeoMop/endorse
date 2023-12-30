import sys
import subprocess
from scoop import futures
from endorse.Bukov2 import boreholes, bukov_common as bcommon, process_bh
params_name="parameters"
from pathlib import Path
script_path = Path(__file__).absolute()

def single_borehole(args):
    workdir, i_bh = args
    return process_bh.process_borehole(workdir, i_bh)

# def borehole_dir(workdir, i_bh):
#     bh_dir = workdir / f"process_bh_{i_bh:03d}"
#     bh_dir.mkdir(parents=True, exist_ok=True)
#     return bh_dir

def all_boreholes(workdir):
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    #n_boreholes = (0,50,5)
    n_boreholes = (bh_set.n_boreholes,)
    bh_args = [(workdir, i_bh) for i_bh in range(*n_boreholes)]

    results = list(futures.map(single_borehole, bh_args))
    bcommon.pkl_write(workdir, results, "all_bh_configs.pkl")
    
pbs_script_template = """
#!/bin/bash
#PBS -j oe
#PBS -m e
set -x 
env | grep PBS_
echo "===="
{python} -m scoop --hostfile $PBS_NODEFILE -vv -n {n_workers} {script_path} {workdir} 
#{python} {script_path} {workdir} 23
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
