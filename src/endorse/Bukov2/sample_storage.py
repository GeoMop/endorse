import h5py
import numpy as np
import time
import traceback

dataset_name="pressure"
failed_ids_name="failed_samples"
done_ids_name="done_samples"

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


# def create_chunked_dataset(file_path, chunk_shape):
#     """
#
#     :param file_path:
#     :param chunk_shape:
#     :return:
#     """
#     # Recomended Chuk size 10kB up to 1MB
#     max_shape = (None, *chunk_shape[1:])    # Allow infinite grow in the number of samples.
#     init_shape = chunk_shape                # Initialize to the size of single chunk. Does not matter.
#     with h5py.File(file_path, 'w') as f:
#         # chunks=True ... automatic chunk size
#         f.create_dataset(dataset_name, shape=init_shape, maxshape=max_shape, chunks=True, dtype='float64')

def create_chunked_dataset(file_path, shape, chunks):
    """

    :param file_path:
    :param shape:
    :param chunks:
    :return:
    """
    # Recomended Chuk size 10kB up to 1MB
    with h5py.File(file_path, 'w') as f:
        # chunks=True ... automatic chunk size
        # f.create_dataset(dataset_name, data=np.zeros(shape), chunks=True, dtype='float64')
        f.create_dataset(dataset_name, shape=shape, chunks=chunks, dtype='float64')
        f.create_dataset(failed_ids_name, shape=(0,), maxshape=(shape[0],), dtype='int')
        f.create_dataset(done_ids_name, shape=(0,), maxshape=(shape[0],), dtype='int')


def append_new_dataset(file_path, name, data):
    with h5py.File(file_path, 'a') as f:
        f.create_dataset(name=name, data=data, dtype='float64')


def set_sample_data(file_path, new_data, idx):
    try:
        # with FileSafe(file_path, mode='a', timeout=60) as f:
        with h5py.File(file_path, 'a') as f:
            if new_data is None:
                set_status_sample(f[failed_ids_name], idx)
            else:
                dset = f[dataset_name]
                if dset.shape[1:] == new_data.shape[1:]:
                    dset[idx, :, :] = new_data
                    set_status_sample(f[done_ids_name], idx)
                else:
                    print("Save sample data failed - wrong shape {}, idx {}.".format(new_data.shape, idx))
                    set_status_sample(f[failed_ids_name], idx)
    except:
        print("Save sample data failed. idx ", idx)
        traceback.print_exc()


def set_status_sample(dset, idx):
    n_existing = dset.shape[0]  # Current actual size in the N dimension
    dset.resize(n_existing + 1, axis=0)
    dset[n_existing] = idx


# def append_data(file_path, new_data):
#     with FileSafe(file_path, mode='a', timeout=60) as f:
#     # with h5py.File(file_path, 'a') as f:
#         dset = f[dataset_name]
#         n_existing = dset.shape[0]  # Current actual size in the N dimension
#
#         if new_data is None:
#             # New size after appending the data
#             new_size = n_existing + 1
#
#             # Resize the dataset to accommodate the new data
#             dset.resize(new_size, axis=0)
#
#             # create empty data
#             empty_data = np.empty(dset.shape[1:])
#             empty_data.fill(np.nan)
#
#
#             # Append the new data
#             dset[n_existing:new_size, :, :] = empty_data
#         else:
#             # New size after appending the data
#             new_size = n_existing + new_data.shape[0]
#
#             # Resize the dataset to accommodate the new data
#             dset.resize(new_size, axis=0)
#
#             # Append the new data
#             dset[n_existing:new_size, :, :] = new_data


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
def get_hdf5_field(hdf_file):
    with h5py.File(hdf_file, 'r') as f:
        return np.array(f[dataset_name])

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
