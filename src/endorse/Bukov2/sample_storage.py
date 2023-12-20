import h5py
import numpy as np

dataset_name="pressure"

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
    with h5py.File(file_path, 'a') as f:
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



def get_hdf5_field(hdf_file):
    with h5py.File(hdf_file, 'r') as f:
        return f[dataset_name]


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
