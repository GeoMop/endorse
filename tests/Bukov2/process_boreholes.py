import sys
import subprocess
import time
from scoop import futures
from endorse.Bukov2 import boreholes, bukov_common as bcommon, process_bh
params_name="parameters"
from pathlib import Path
script_path = Path(__file__).absolute()

def single_borehole(args):
    workdir, i_bh = args
    result = process_bh.process_borehole(workdir, i_bh)
    print(f"[{i_bh}] done")
    return result

# def borehole_dir(workdir, i_bh):
#     bh_dir = workdir / f"process_bh_{i_bh:03d}"
#     bh_dir.mkdir(parents=True, exist_ok=True)
#     return bh_dir

def all_boreholes(workdir, map_fn):
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    #n_boreholes = (0,50,5)
    # bh_keys = list(bh_set.boreholes.keys())
    # print(bh_keys)
    # print(list(range(bh_set.n_boreholes)))
    # assert bh_keys == list(range(bh_set.n_boreholes))
    #
    # Set directories to avoid NFS IO errors
    for i_bh in range(bh_set.n_boreholes):
        bh_dir = workdir / "processed_bh" / f"bh_{i_bh:03d}"
        bh_dir.mkdir(parents=True, exist_ok=True)
    bh_args = [(workdir, i_bh) for i_bh in range(bh_set.n_boreholes)]
    #time.sleep(10)

    results = list(map_fn(single_borehole, bh_args))
    print("Results collected: ", str(results)[:200])
    bcommon.pkl_write(workdir, results, "all_bh_configs.pkl")
    
pbs_script_template = """
#!/bin/bash
#PBS -j oe
#PBS -m e
set -x 
env | grep PBS_
#umask 0007
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
    
    per_chunk = 2
    n_chunks = n_workers // (per_chunk+1) 
    
    cmd = ['qsub', '-q', queue, '-l', f'select={n_chunks}:ncpus={per_chunk}:mem=90gb', '-l', 'place=scatter', pbs_filename]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    workdir = Path(sys.argv[1]).absolute()
    if len(sys.argv) > 2 and (sys.argv[2] == 'submit'):
        submit_pbs(workdir)
    elif len(sys.argv) > 2 and (sys.argv[2] == 'local'):
        all_boreholes(workdir, map_fn = map)
    elif len(sys.argv) > 2:
        i_bh = int(sys.argv[2])
        single_borehole( (workdir, i_bh) )
    else:
        all_boreholes(workdir, map_fn=futures.map)
