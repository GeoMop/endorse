"""
Collect HDF files from workers.

Usage:

python collect_hdf.py   workdir/<pattern>

e.g.
python collect_hdf.py   workdir/sampled_data_*.h5
"""
import sys
import time

import h5py
import numpy as np
from pathlib import Path
from endorse.Bukov2.sample_storage import dataset_name, failed_ids_name, done_ids_name

def detect_done_ids(dset):
    nonzero_mask = np.zeros((dset.shape[0]), dtype=np.int32)
    for chunk_slice in dset.iter_chunks():
        result_slice = chunk_slice[0]
        chunk_data = dset[chunk_slice]
        nonzero_mask[result_slice] += np.all(chunk_data != 0.0, axis=(1, 2))
    return np.arange(dset.shape[0], dtype=np.int32)[nonzero_mask > 0]

def fill_single_hdf(out_f, in_f):
    # main dataset
    in_dset = in_f[dataset_name]
    out_dset = out_f.require_dataset(dataset_name, in_dset.shape, chunks=True, dtype='float64')

    if done_ids_name in in_f:
        done_ids = np.array(in_f[done_ids_name], dtyp=np.int32)
    else:
        done_ids = detect_done_ids(in_dset)
    out_dset[done_ids] = in_dset[done_ids]
    # failed dataset
    in_failed = np.array(in_f[failed_ids_name], dtype=np.int32).ravel()
    out_failed = out_f.require_dataset(failed_ids_name, shape=(1,), maxshape=(in_dset.shape[0],), dtype='int32')
    failed = np.concatenate((out_failed, in_failed))
    failed = np.unique(failed)
    out_failed.resize(failed.shape)
    out_failed[:] = failed


def collect(workdir, pattern, outfile):
    with h5py.File(outfile, mode='w') as out_f:
        for f_name in workdir.glob(pattern):
            print(f"  {f_name.name}")
            with h5py.File(f_name, mode='r') as in_f:
                fill_single_hdf(out_f, in_f)

def rm_axis(tup, axis):
    return tup[:axis] + tup[axis+1:]

def chunked_mean(dset, axis):
    if not dset.chunks:
        raise ValueError("Dataset is not chunked")

    # Initialize variables for mean calculation
    new_shape = rm_axis(dset.shape, axis)
    total_sum = np.zeros(new_shape, dtype=np.float64)
    total_count = np.zeros(new_shape, dtype=np.float64)

    # Iterate and sum over chunks
    for chunk_slice in dset.iter_chunks():
        res_slice = rm_axis(chunk_slice, axis)
        chunk_data = dset[chunk_slice]
        total_sum[res_slice] += np.sum(chunk_data, axis=axis)
        total_count[res_slice] += 1
    # Calculate the overall mean
    return total_sum / total_count


def impute_nans_by_mean(hdf_file):
    with h5py.File(hdf_file, mode='a') as f :
        dset = f[dataset_name]
        # (n_samples, n_times, n_elements)
        mean = chunked_mean(dset, axis=0)
        failed_ids = np.array(f[failed_ids_name], dtype=np.int32)
        percent = int(float(failed_ids.shape[0]) / dset.shape[0] * 100)
        print(f"Imputing {failed_ids.shape[0]} failed samples [{percent:3d} %] by the mean.")
        for i in failed_ids:
            dset[i, :, :] = mean




def main():
    path = Path(sys.argv[1]).absolute()
    workdir = path.parent
    pattern = path.name
    basename = str(pattern)
    basename = basename[:basename.find('_*')]
    out_file = workdir / (basename + ".h5")
    print("Collectiong hdf files ...\n")
    start = time.process_time_ns()
    collect(workdir, pattern, out_file)
    sec = (time.process_time_ns() - start) / 1e9
    print(f"Done at {sec} s.")

    print(f"Imputting NaNs with means in {out_file} ...\n")
    start = time.process_time_ns()
    impute_nans_by_mean(out_file)
    sec = (time.process_time_ns() - start) / 1e9
    print(f"Done at {sec} s.")


if __name__ == '__main__':
    main()
