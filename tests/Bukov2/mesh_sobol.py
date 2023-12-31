"""
Collect HDF files from workers.

Usage:

python collect_hdf.py   workdir/<pattern>

e.g.
python collect_hdf.py   workdir/sampled_data_*.h5
"""
import sys
import logging

import h5py
import numpy as np
from pathlib import Path
from endorse import common
from endorse.Bukov2.sample_storage import dataset_name, failed_ids_name, done_ids_name
from endorse.Bukov2.bukov_common import memoize, file_result, load_cfg
from endorse.Bukov2 import sa_problem, sobol_fast
params_name="parameters"

#####################################š

def chracterize_samples(in_dset):
    n_samples, n_times, n_els = in_dset.shape
    s_max = np.empty(n_samples)
    s_min = np.empty(n_samples)
    for i_sample in range(n_samples):
        sample = np.array(in_dset[i_sample, :, :])
        s_max[i_sample] = np.max(sample)
        s_min[i_sample] = np.min(sample)
    return s_max, s_min

@file_result("mesh_sobol_st.h5")
def mesh_sobol_st(workdir, out_file, in_file):
    workdir, cfg = load_cfg(workdir / "Bukov2_mesh.yaml")
    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    n_params = problem['num_vars']
    with h5py.File(workdir / in_file, mode='r') as in_f:
        in_dset = in_f[dataset_name]
        n_samples, n_times, n_els = in_dset.shape
        with h5py.File(workdir / out_file, mode='w') as out_f:
            out_sobol_t = out_f.create_dataset('sobol_indices', (n_times, n_els, n_params), dtype='float64')
            out_mean = out_f.create_dataset('mean', (n_times, n_els), dtype='float64')
            out_std = out_f.create_dataset('std', (n_times, n_els), dtype='float64')
            out_max_sample = out_f.create_dataset('max_sample', (n_times, n_els), dtype='float64')
            out_med_sample = out_f.create_dataset('med_sample', (n_times, n_els), dtype='float64')
            s_max, s_min = chracterize_samples(in_dset)
            samples_sorted = np.argsort(s_max - s_min)
            out_max_sample[:,:]  = in_dset[samples_sorted[-1], :, :]
            out_med_sample[:, :] = in_dset[samples_sorted[n_samples//2], :, :]
            for i_time in range(n_times):
                time_frame = np.array(in_dset[:, i_time, :])
                sobol_samples = time_frame.transpose([1,0])
                out_sobol_t[i_time, :, :] = sobol_fast.vec_sobol_total_only(sobol_samples, problem)
                out_mean[i_time, :] = np.mean(time_frame, axis=0)
                out_std[i_time, :] = np.std(time_frame, axis=0)
        return out_file





def main(workdir):
    in_file = "sampled_fixed.h5"
    mesh_sobol_st(workdir, in_file)


if __name__ == '__main__':
    workdir = Path(sys.argv[1]).absolute()
    main(workdir)
