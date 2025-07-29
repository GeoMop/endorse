import numpy as np
from pathlib import Path

import xarray as xr
import zarr

def init_zarr_store(store_path: Path,
                    # qmc_size: int,
                    # block_size: int,
                    time_size: int,
                    x_size: int,
                    y_size: int,
                    z_size: int,
                    dtype=np.float32,
                    compressor=None) -> None:
    """
    Initialize an empty Zarr store with the specified dimensions and chunking.

    Parameters
    ----------
    store_path : Path
        Path to the Zarr store (directory or Zarr URL).
    qmc_size : int
        Length of the 'qmc' dimension.
    block_size : int
        Number of blocks: A, B, AB, (BA).
    x_size : int
        Length of the 'X' dimension.
    y_size : int
        Length of the 'Y' dimension.
    time_size : int
        Length of the 'time' dimension.
    dtype : NumPy dtype, optional
        Data type of the array (default: np.float32).
    compressor : zarr compressor, optional
        Compressor to use (default: None).
    """
    # Define dimensions: 'sample' is unlimited/appendable
    # dims = ('sample', 'qmc', 'block', 'time', 'X', 'Y', 'Z')
    # shape = (0, qmc_size, block_size, time_size, x_size, y_size, 2)
    dims = ('sample', 'time', 'X', 'Y', 'Z')
    shape = (10, time_size, x_size, y_size, z_size)

    # Create an empty DataArray with 0 length sample dimension
    data = xr.DataArray(
        np.zeros(shape, dtype=dtype),
        dims=dims,
        coords={
            # 'qmc': np.arange(qmc_size),
            # 'block': np.arange(block_size),
            'sample': np.arange(10),
            'time': np.arange(time_size),
            'X': np.arange(x_size),
            'Y': np.arange(y_size),
            'Z': np.arange(z_size),
        }
    )

    ds = xr.Dataset({'data': data})

    # Set encoding for chunking and compression
    ds.encoding['data'] = {
        # 'chunks': (1, qmc_size, block_size, time_size, x_size, y_size, 2),
        'chunks': (1, time_size, x_size, y_size, z_size),
        'compressor': compressor
    }

    # Write to Zarr, overwrite if exists
    ds.to_zarr(str(store_path), mode='w')


def write_zarr_slice(store_path: str,
                     sample_idx: int,
                     # qmc_idx: int,
                     # block_idx: int,
                     slice_array: np.ndarray) -> None:
    """
    Write a slice of data into an existing Zarr store for given sample and qmc indices.

    Parameters
    ----------
    store_path : str
        Path to the Zarr store.
    sample_idx : int
        Index along the 'sample' dimension where data will be written.
    qmc_idx : int
        Index along the 'qmc' dimension where data will be written.
    block_idx : int
        Index along the 'block' dimension where data will be written.
    slice_array : np.ndarray
        NumPy array of shape (time, X, Y, Z) matching the store dimensions.
    """
    # Open the existing Zarr store as an Xarray dataset
    ds = xr.open_zarr(store_path, consolidated=True)

    # Validate slice_array shape
    expected_shape = (ds.sizes['time'], ds.sizes['X'], ds.sizes['Y'], ds.sizes['Z'])
    if slice_array.shape != expected_shape:
        raise ValueError(f"slice_array must have shape {expected_shape}, got {slice_array.shape}")

    # Wrap slice_array into a DataArray with new sample and qmc coords
    da = xr.DataArray(
        # slice_array[np.newaxis, np.newaxis, np.newaxis, ...],  # add sample, qmc and block dims
        slice_array[np.newaxis, ...],  # add sample, qmc and block dims
        # dims=('sample', 'qmc', 'block', 'time', 'X', 'Y', 'Z'),
        dims=('sample', 'time', 'X', 'Y', 'Z'),
        coords={
            'sample': [sample_idx],
            # 'qmc': [qmc_idx],
            # 'block': [block_idx],
            'time': ds.coords['time'],
            'X': ds.coords['X'],
            'Y': ds.coords['Y'],
            'Z': ds.coords['Z'],
        }
    )

    # Write the slice by specifying the region to overwrite
    da.to_dataset(name='data').to_zarr(
    # da.to_dataset(name='data').drop_vars(['time', 'X', 'Y', 'Z']).to_zarr(
        store_path,
        mode='a',
        # region='auto'
        region={
            # 'sample': 'auto',
            'sample': slice(sample_idx, sample_idx + 1),
            'time': 'auto',
            'X': 'auto',
            'Y': 'auto',
            'Z': 'auto',
            # 'qmc': slice(qmc_idx, qmc_idx + 1),
            # 'block': slice(block_idx, block_idx + 1)
        #     # 'data': {
        #     #     'sample': slice(sample_idx, sample_idx + 1),
        #     #     'qmc': slice(qmc_idx, qmc_idx + 1),
        #     #     'block': slice(block_idx, block_idx + 1)
        #     # }
        }
    )


def main():
    workdir = Path(".").absolute() / "workdir_zarr"
    # temporary shortcut for direct zarr
    # qmc_size = 2
    # block_size = 4 # second order salteli
    time_size = 18
    grid_size = [20, 20]
    init_zarr_store(workdir / "transport_sampling",
                    # qmc_size=qmc_size,
                    time_size=18,
                    # block_size=block_size,
                    x_size=grid_size[0],
                    y_size=grid_size[1],
                    z_size=2)

    tags = [0, 1, 2]
    values = np.random.rand(time_size, *grid_size, 2)

    # block = values.reshape(time_size, *grid_size, 2)
    write_zarr_slice(store_path=str(workdir / "transport_sampling"),
                     sample_idx=tags[0],
                     # qmc_idx=tags[1],
                     # block_idx=tags[2],
                     slice_array=values)


if __name__ == '__main__':
    main()