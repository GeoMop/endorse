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
from endorse.Bukov2.sample_storage import dataset_name, failed_ids_name, done_ids_name
from endorse.Bukov2.bukov_common import memoize, file_result
params_name="parameters"




#####################################š

def detect_done_ids(dset):
    nonzero_mask = np.zeros((dset.shape[0]), dtype=np.int32)
    for chunk_slice in dset.iter_chunks():
        result_slice = chunk_slice[0]
        chunk_data = dset[chunk_slice]
        nonzero_mask[result_slice] += np.all(chunk_data != 0.0, axis=(1, 2))
    return np.arange(dset.shape[0], dtype=np.int32)[nonzero_mask > 0]

def merge_ids(out_f, ids_name, max_size, in_ids):
    out_ids = out_f.require_dataset(ids_name, shape=(0,), maxshape=(max_size,), dtype=np.int32)
    #out_failed_mask |= failed_mask
    merged = np.concatenate((out_ids, in_ids))
    merged = np.sort(np.unique(merged))
    out_ids.resize(merged.shape)
    out_ids[:] = merged
    return merged


def fill_single_hdf(out_f, in_f):
    # Input datasets
    in_dset = in_f[dataset_name]
    in_params = in_f[params_name]
    if done_ids_name in in_f:
        in_done_ids = np.array(in_f[done_ids_name], dtype=np.int32)
    else:
        #raise AttributeError("No done IDs.")
        in_done_ids = detect_done_ids(in_dset)
    in_failed_ids = np.array(in_f[failed_ids_name], dtype=np.int32)
    #done_mask = np.zeros(in_dset.maxshape[0])
    #done_mask[in_done_ids] = 1
    #failed_mask = np.zeros(in_dset.maxshape[0])
    #failed_mask[in_failed_ids] = 1

    n_samples = in_dset.shape[0]
    n_params = in_params.shape[1]
    chunks = (1, 1, in_dset.shape[2])
    out_dset = out_f.require_dataset(dataset_name, in_dset.shape, chunks=chunks, dtype='float64')
    chunks = (1, in_params.shape[1])
    out_params = out_f.require_dataset(params_name, (n_samples, n_params), chunks=chunks, dtype='float64')


    out_dset[in_done_ids] = in_dset[in_done_ids]
    merged_ids = np.concatenate((in_failed_ids, in_done_ids))
    #out_params[merged_ids] = in_params[merged_ids]
    
    # failed dataset
    failed_ids = merge_ids(out_f, failed_ids_name, in_dset.shape[0], in_failed_ids)
    done_ids = merge_ids(out_f, done_ids_name, in_dset.shape[0], in_done_ids)

    print(f"    {len(done_ids)} done and  {len(failed_ids)} failed samples out of {out_dset.shape[0]}")

@file_result("sampled_collected.h5")
def collect(workdir, out_file):
    pattern = "sampled_data_*.h5"
    #print(list(workdir.glob(pattern)))
    with h5py.File(workdir / out_file, mode='a') as out_f:
        in_files = list(workdir.glob(pattern))
        for f_name in sorted(in_files):
            print(f"  {f_name.name}")
            with h5py.File(f_name, mode='r') as in_f:
                fill_single_hdf(out_f, in_f)
    return out_file

############################################
@memoize
def chunked_mean(workdir, dataset, axis, mask):
    # Ensure the mask length matches the size of the dataset along the specified axis
    if mask.shape[0] != dataset.shape[axis]:
        raise ValueError("Length of mask must match size of dataset along specified axis")

    # Initialize arrays to store the sum and count
    sum_shape = list(dataset.shape)
    del sum_shape[axis]
    masked_sum = np.zeros(sum_shape)
    masked_count = np.zeros(sum_shape)

    # Process each chunk
    for chunk_slice in dataset.iter_chunks():
        data_chunk = dataset[chunk_slice]

        chunk_mask = mask[chunk_slice[axis]]

        # Apply the mask
        masked_data = np.compress(chunk_mask, data_chunk, axis=axis)

        # Calculate the sum and count for the chunk
        chunk_sum = masked_data.sum(axis=axis)
        chunk_count = masked_data.shape[axis]

        # Store the sum and count
        reduced_chunk_slice = list(chunk_slice)
        del reduced_chunk_slice[axis]
        masked_sum[tuple(reduced_chunk_slice)] += chunk_sum
        masked_count[tuple(reduced_chunk_slice)] += chunk_count

    ## Calculate the mean
    masked_mean = masked_sum / masked_count
    return masked_mean

#############################################################

def in_dset_done_mask(in_f):
    done_ids = np.array(in_f[done_ids_name], dtype=np.int32)
    n_samples = in_f[dataset_name].shape[0]
    mask = np.zeros((n_samples,), dtype=np.int32)
    mask[done_ids] = 1
    return mask


def _get_completed(out_f, in_f, mean, mask):
    n_params = in_f[params_name].shape[1]
    group_size = 2 * (n_params + 1)     # second order
    in_dset = in_f[dataset_name]
    n_samples = in_dset.shape[0]
  
    orig_ids = np.arange(n_samples, dtype=np.int32)
    orig_ids[np.logical_not(mask)] = -1
    orig_ids = orig_ids.reshape(-1, group_size)
    mask = mask.reshape(-1, group_size)

    # Determine how many samples necessary for Total Sobol index are done.
    #
    min_group_size = n_params - 1
    mask_st = mask[:, :n_params+1]
    done_nums = np.sum(mask_st, axis=1)
    done_groups = done_nums > min_group_size
    mask_done = np.zeros((n_samples,), dtype=np.int32).reshape(-1, group_size)
    mask_done[done_groups, : ] = 1
    orig_ids = orig_ids[done_groups, :].ravel()
    print(f"    Imputing {sum(orig_ids == -1)} from {len(orig_ids)}")

    growth_size = group_size

    # For unknown, we get error on file closing if the dataset is created at its final size.
    # It is faster and works without error if the dataset is resized as the samples are added.
    new_shape = [len(done_groups) * n_params + 1, *in_dset.shape[1:]]
    max_shape = list(new_shape)
    new_shape[0] = 0
    chunks = (1, 1, in_dset.shape[2]) 
    out_dset = out_f.create_dataset(dataset_name, new_shape, chunks=chunks, maxshape=in_dset.shape, dtype='float64')
    print("    New dataset: ", new_shape)
    print("    Max shape dataset: ", new_shape)

    for row, orig in enumerate(orig_ids):
        if row >= out_dset.shape[0]:            
            out_dset.resize( (out_dset.shape[0] + growth_size, *new_shape[1:]) )
        if orig == -1:
            print(f"    {(row, orig)}   mean shape ", mean.shape)
            out_dset[row, :, :]  = mean
        else:
            print(f"    {(row, orig)}   assign shape ", (in_dset[orig, :, :]).shape)
            out_dset[row, :, :]  = in_dset[orig, :, :]


    
@file_result("sampled_fixed.h5")
def get_completed_groups(workdir, out_file, in_file):
    # in_file ... passed from file_result

    with h5py.File(workdir / in_file, mode='r') as in_f:
        mask = in_dset_done_mask(in_f)
        in_dset = in_f[dataset_name]
        mean = chunked_mean(workdir, in_dset, axis=0, mask=mask)

        with h5py.File(workdir / out_file, mode='w') as out_f:
            _get_completed(out_f, in_f, mean, mask)
    return out_file


@file_result("sampled_reduced.h5")
def reduced_dataset(workdir, out_file, in_file):
    with h5py.File(workdir / in_file, mode='r') as in_f:
        dset = in_f[dataset_name] 
        print("Input dset shape: ", dset.shape)
        out_shape = list(dset.shape)
        out_shape[0] = 96
        out_shape[1] = 5 
        print("Output dset shape: ", out_shape)
            
        # extract smaller subset
        with h5py.File(workdir / out_file, mode='w') as out_f:
            out_dset = out_f.create_dataset(dataset_name, shape=out_shape)
            out_dset[...] = dset[0:out_shape[0], 0:2*out_shape[1]:2, :]    

#
# def impute_nans_by_mean(hdf_file):
#     print(f"Imputting NaNs with means in {out_file} ...\n")
#     start = time.process_time_ns()
#     with h5py.File(hdf_file, mode='a') as f :
#         dset = f[dataset_name]
#         # (n_samples, n_times, n_elements)
#         mean = chunked_mean(dset, axis=0)
#         failed_ids = np.array(f[failed_ids_name], dtype=np.int32)
#         percent = int(float(failed_ids.shape[0]) / dset.shape[0] * 100)
#         print(f"Imputing {failed_ids.shape[0]} failed samples [{percent:3d} %] by the mean.")
#         for i in failed_ids:
#             dset[i, :, :] = mean




def main(workdir):
    collected_hdf5 = collect(workdir)
    fixed = get_completed_groups(workdir, collected_hdf5)
    reduced_dataset(workdir, fixed)


if __name__ == '__main__':
    workdir = Path(sys.argv[1]).absolute()
    main(workdir)
