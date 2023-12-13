import h5py
import numpy as np
import time

dataset_name="pressure"

class FileSafe(h5py.File):
    """
    Context manager for openning HDF5 files with some timeout
    amd retrying of getting acces.creation and usage of a workspace dir.

    Usage:
    with FileSafe(filename, mode='w', timout=60) as f:
        f.attrs['key'] = value
        f.create_group(...)
    .. automaticaly closed
    """
    def __init__(self, filename:str, mode='r', timeout=5, **kwargs):
        """
        :param filename:
        :param timeout: time to try acquire the lock
        """
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                super().__init__(filename, mode, **kwargs)
                return
            except BlockingIOError as e:
                time.sleep(0.01)
                continue
            break
        logging.exception(f"Unable to lock access to HDF5 file: {filename}, give up after: {timeout}s.")
        raise BlockingIOError(f"Unable to lock access to HDF5 file: {filename}, give up after: {timeout}s.")

def create_chunked_dataset(file_path, chunk_shape):
    """

    :param file_path:
    :param chunk_shape:
    :return:
    """
    # Recomended Chuk size 10kB up to 1MB
    max_shape = (None, *chunk_shape[1:])    # Allow infinite grow in the number of samples.
    init_shape = chunk_shape                # Initialize to the size of single chunk. Doesn;t metter.
    with h5py.File(file_path, 'w') as f:
        # chunks=True ... automatic chunk size
        f.create_dataset("pressure", shape=init_shape, maxshape=max_shape, chunks=True, dtype='float64')

def append_data(file_path, new_data):
    with FileSafe(file_path, mode='a', timeout=60) as f:
    # with h5py.File(file_path, 'a') as f:
        dset = f[dataset_name]
        n_existing = dset.shape[0]  # Current actual size in the N dimension

        # New size after appending the data
        new_size = n_existing + new_data.shape[0]

        # Resize the dataset to accommodate the new data
        dset.resize(new_size, axis=0)

        # Append the new data
        dset[n_existing:new_size, :, :] = new_data


def read_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        dset = f[dataset_name]
        #data = np.empty(dset.shape)
        data = dset[...]
        return data

# Example usage
# file_path = 'your_file.h5'
# dataset_name = 'your_dataset'
# initial_shape = (K, M, N + extra_space)  # Pre-allocate extra space in the N dimension
# chunks = (K, M, chunk_size)  # Define a suitable chunk size
#
# create_chunked_dataset(file_path, dataset_name, initial_shape, max_shape, chunks)
#
# # When you have new data to append
# new_data = np.random.rand(K, M, n)  # Your new data
# append_data(file_path, dataset_name, new_data)