"""
Prepare sampled pressure data on individual boreholes.
Write to separate files in order to paralelize the chamber optimization.
Slice paralelization of this process is possible but it is bound by the IO of
the disk storage.

We need dask for better performance when the large computations would be split into smaller pieces
allowing better overleap of calculation and communication and also keeping files distributed.
"""
import sys
import logging

import h5py
import numpy as np
from pathlib import Path
from endorse.Bukov2.sample_storage import dataset_name, failed_ids_name, done_ids_name
from endorse.common import load_config
from endorse.Bukov2 import sa_problem, boreholes
from endorse.Bukov2.bukov_common import memoize, file_result, load_cfg
params_name="parameters"
from pathlib import Path

def main(workdir, bh_range):
    cfg_file = workdir / "Bukov2_mesh.yaml"
    workdir, cfg = load_cfg(cfg_file)
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    updated_files_dict = bh_set.project_field(workdir, cfg, bh_range)
    print("Updated: ", updated_files_dict)

if __name__ == '__main__':
    workdir = Path(sys.argv[1]).absolute()
    if len(sys.argv) > 2:
        bh_range = int(sys.argv[2]), int(sys.argv[3])
    else:
        bh_range = None
    main(workdir, bh_range)
