"""
Collect HDF files from workers.

Usage:

python collect_hdf.py   workdir/<pattern>

e.g.
python collect_hdf.py   workdir/sampled_data_*.h5
"""
import sys
import time
import pickle

import h5py
import numpy as np
from pathlib import Path
from endorse.Bukov2.sample_storage import dataset_name, failed_ids_name, done_ids_name



def pkl_write(workdir, data, name):
    with open(workdir / name, 'wb') as f:
        pickle.dump(data, f)

def pkl_read(workdir, name):
    try:
        with open(workdir / name, 'rb') as f:
            opt_results = pickle.load(f)
    except Exception:
        opt_results = None
    return opt_results

def memoize(func):
    def wrapper(workdir, *args, **kwargs):
        
    
        start = time.process_time_ns()
        fname = f"{func.__name__}.pkl"
        
        val = pkl_read(workdir, fname)
        if val is None:
            print(f"Execute {func.__name__}  ...", ne)
            val = func(*args, **kwargs)
            pkl_write(workdir, val, fname)
        else:
            print(f"Skip existing {basename}.h5 ...")
        
        return val
    return wrapper


def file_result(func):
    def decorator(filename):
        def wrapper(workdir, *args, **kwargs):
            fname = f"{func.__name__}.pkl"
            val = pkl_read(workdir, fname)
            if val is None:
                val = func(*args, **kwargs)
                pkl_write(workdir, val, fname)
            return val
        return wrapper
    return decorator



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
        done_ids = np.array(in_f[done_ids_name], dtype=np.int32)
    else:
        raise AttributeError("No done IDs.")
        #done_ids = detect_done_ids(in_dset)
    out_dset[done_ids] = in_dset[done_ids]
    
    # failed dataset
    in_failed = np.array(in_f[failed_ids_name], dtype=np.int32).ravel()
    out_failed = out_f.require_dataset(failed_ids_name, shape=(1,), maxshape=(in_dset.shape[0],), dtype='int32')
    failed = np.concatenate((out_failed, in_failed))
    failed = np.unique(failed)
    out_failed.resize(failed.shape)
    out_failed[:] = failed
    
    # done dataset
    out_done = out_f.require_dataset(done_ids_name, shape=(1,), maxshape=(in_dset.shape[0],), dtype='int32')
    done = np.concatenate((out_done, done_ids))
    done = np.unique(done)    
    out_done.resize(done.shape)
    out_done[:] = done
    print(f"    {len(done)} done samples out of {out_dset.shape[0]}")

def collect(workdir, pattern):
    
    basename = "sampled_data_collected"
    out_file = workdir / (basename + ".h5")
    if out_file.exists():
        print(f"Skip existing {basename}.h5 ...")
        
    
    print(f"Forming {basename}.h5 ...")
    start = time.process_time_ns()
    #print(list(workdir.glob(pattern)))
    with h5py.File(outfile, mode='a') as out_f:
        for f_name in workdir.glob(pattern):
            print(f"  {f_name.name}")
            with h5py.File(f_name, mode='r') as in_f:
                fill_single_hdf(out_f, in_f)
    sec = (time.process_time_ns() - start) / 1e9
    print(f"Done at {sec} s.")

def in_dset_done_mask(in_f):
    done_ids = np.array(in_f[done_ids_name], dtype=np.int32)
    n_samples = in_f[dataset_name].shape[0]
    mask = np.zeros((n_samples,), dtype=np.int32)
    mask[done_ids] = 1
    return mask

def _get_completed(out_f, in_f, mean, mask):
    n_params = 11
    group_size = 2 * (n_params + 1)
    in_dset = in_f[dataset_name]
    n_samples = in_dset.shape[0]
  
    orig_ids = np.arange(n_samples, dtype=np.int32)
    orig_ids[np.logical_not(mask)] = -1
    orig_ids = orig_ids.reshape(-1, group_size)
    mask = mask.reshape(-1, group_size)
    
    mask_st = mask[:, :n_params+1]
    done_nums = np.sum(mask_st, axis=1)
    done_groups = done_nums > 10
    mask_done = np.zeros((n_samples,), dtype=np.int32).reshape(-1, group_size)
    mask_done[done_groups, : ] = 1
    orig_ids = orig_ids[done_groups, :].ravel()
    print(f"Imputing {sum(orig_ids == -1)} from {len(orig_ids)}") 
    
    
    growth_size = 32
    
    new_shape = [len(done_groups) * n_params + 1, *in_dset.shape[1:]]
    max_shape = list(new_shape)
    new_shape[0] = 32
    chunks = (1, 1, in_dset.shape[2]) 
    out_dset = out_f.create_dataset(dataset_name, new_shape, chunks=chunks, maxshape=max_shape, dtype='float64')
    print("New dataset: ", new_shape)
    print("Max shape dataset: ", new_shape)
    #out_dset[0, :, :]  = in_dset[0, :, :]
    #out_dset[1, :, :]  = in_dset[1, :, :]
    
    for row, orig in enumerate(orig_ids):
        if row >= out_dset.shape[0]:            
            out_dset.resize( (out_dset.shape[0] + growth_size, *new_shape[1:]) )
        print((row, orig))
        if orig == -1:
            print("  mean shape ", mean.shape) 
            out_dset[row, :, :]  = mean
        else:
            print("  assign shape ", (in_dset[orig, :, :]).shape)
            out_dset[row, :, :]  = in_dset[orig, :, :]
        #if row ==0:
            #print("DONE.")
            #return
    

    

def get_completed_groups(workdir, in_file, out_file):
    #mean = np.zeros(in_dset.shape[1:]) 

    print("Completed groups hdf files ...\n")
    #print(list(workdir.glob(pattern)))
    with h5py.File(workdir / in_file, mode='r') as in_f:
        start = time.process_time_ns()
        mask = in_dset_done_mask(in_f)
        in_dset = in_f[dataset_name]
        mean = chunked_mean(workdir, in_dset, axis=0, mask=mask)

        with h5py.File(workdir / out_file, mode='w') as out_f:
            _get_completed(out_f, in_f, mean, mask)
            print("Output to file done.")
            
    sec = (time.process_time_ns() - start) / 1e9
    print(f"Done at {sec} s.")


def rm_axis(tup, axis):
    return tup[:axis] + tup[axis+1:]


@memoize
def chunked_mean(dataset, axis, mask):
    # Ensure the mask length matches the size of the dataset along the specified axis
    print("Computing mean .. ")
    if mask.shape[0] != dataset.shape[axis]:
        raise ValueError("Length of mask must match size of dataset along specified axis")

    # Initialize arrays to store the sum and count
    sum_shape = list(dataset.shape)
    del sum_shape[axis]
    masked_sum = np.zeros(sum_shape)
    masked_count = np.zeros(sum_shape)

    # Process each chunk
    for chunk_slice in dataset.iter_chunks():
        print(chunk_slice)
        data_chunk = dataset[chunk_slice]

        chunk_mask = mask[chunk_slice[axis]]
        # Apply the mask
        print(data_chunk.shape)
        masked_data = np.compress(chunk_mask, data_chunk, axis=axis)

        # Calculate the sum and count for the chunk
        chunk_sum = masked_data.sum(axis=axis)
        chunk_count = masked_data.shape[axis]

        # Store the sum and count
        #reduced_chunk_slice = tuple(slice(None) if i != axis else None for i in range(len(chunk_slice)))
        reduced_chunk_slice = list(chunk_slice)
        del reduced_chunk_slice[axis]
        masked_sum[tuple(reduced_chunk_slice)] += chunk_sum
        masked_count[tuple(reduced_chunk_slice)] += chunk_count

    ## Calculate the mean
    masked_mean = masked_sum / masked_count
    return masked_mean
    
    
    #if not dset.chunks:
        #raise ValueError("Dataset is not chunked")

    ## Initialize variables for mean calculation
    #new_shape = rm_axis(dset.shape, axis)
    #total_sum = np.zeros(new_shape, dtype=np.float64)
    #total_count = np.zeros(new_shape, dtype=np.float64)

    ## Iterate and sum over chunks
    #for chunk_slice in dset.iter_chunks():
        #res_slice = rm_axis(chunk_slice, axis)
        #l = list(chunk_slice)
        #l[axis] = mask
        #chunk_data = dset[chunk_slice]
        #total_sum[res_slice] += np.sum(chunk_data, axis=axis)
        #total_count[res_slice] += 1
    ## Calculate the overall mean
    #return total_sum / total_count


def impute_nans_by_mean(hdf_file):
    print(f"Imputting NaNs with means in {out_file} ...\n")
    start = time.process_time_ns()
    with h5py.File(hdf_file, mode='a') as f :
        dset = f[dataset_name]
        # (n_samples, n_times, n_elements)
        mean = chunked_mean(dset, axis=0)
        failed_ids = np.array(f[failed_ids_name], dtype=np.int32)
        percent = int(float(failed_ids.shape[0]) / dset.shape[0] * 100)
        print(f"Imputing {failed_ids.shape[0]} failed samples [{percent:3d} %] by the mean.")
        for i in failed_ids:
            dset[i, :, :] = mean
    sec = (time.process_time_ns() - start) / 1e9
    print(f"Done at {sec} s.")




def main():
    
    path = Path(sys.argv[1]).absolute()
    workdir = path.parent
    pattern = path.name
    collect(workdir, pattern)
    
    get_completed_groups(workdir, pattern, "sampled_done.h5")
    
    

    #impute_nans_by_mean(out_file)


if __name__ == '__main__':
    main()
