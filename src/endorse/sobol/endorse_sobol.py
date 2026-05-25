"""
Demonstration script combining:

1. usage of Dask managing and communication PBS jobs
2. 
"""

import xarray as xr
import numpy as np
from dask.distributed import Client
from dask_jobqueue import PBSCluster

# Step 1: Set up the Dask cluster
cluster = PBSCluster(
    queue='regular',
    project='myproject',
    cores=24,
    memory='100GB',
    resource_spec='select=1:ncpus=24:mem=100GB',
    walltime='02:00:00',
    local_directory='$TMPDIR'
)

# Scale cluster to the desired number of workers
cluster.scale(jobs=10)  # Assume we want 10 workers

# Connect to the cluster
client = Client(cluster)

# Step 2: Define your data array (N x m)
data = np.random.rand(100, 10)  # Example: 100 rows, 10 columns
xarray_data = xr.DataArray(data, dims=['rows', 'cols'])

# Step 3: Define the function F to be applied to each row
def function_F(row):
    # Example function: compute the sum of squares of the row elements
    return np.sum(row**2)

# Step 4: Apply function in parallel using Dask
# Convert xarray DataArray to Dask array with appropriate chunking
dask_array = xarray_data.chunk({'rows': 1})  # Chunk by rows for parallel row-wise operation

# Map the function over rows using map_blocks. Here we ensure the client is explicitly used.
fn = lambda x: np.apply_along_axis(function_F, 1, x)
result_array = dask_array.map_blocks(fn, dtype=float).compute(scheduler=client

# Convert the result back into an xarray DataArray
result_xarray = xr.DataArray(result_array, dims=['rows'])

# Print results
print(result_xarray)



import dask.array as da
import xarray as xr
import zarr
from dask.distributed import Client
from dask_jobqueue import PBSCluster

# Step 1: Initialize the Dask cluster
n_proc = 24
cluster = PBSCluster(
    queue='charon', 
    cores=n_proc, 
    memory='100GB', 
    walltime='02:00:00',
    local_directory='$TMPDIR'
) #, resource_spec='select=1:ncpus=24:mem=100GB')
cluster.scale(jobs=n_proc)  # Adjust according to your needs
client = Client(cluster)


# Step 2: Prepare the input data
data = da.random.random((10000, 1000), chunks=(100, 1000))  # Large dataset, chunked by rows
store = zarr.DirectoryStore('input_data.zarr')
xarray_data = xr.DataArray(data)
xarray_data.to_zarr(store, mode='w')

# Define a more complex function that results in vectors, not scalars
def complex_function(subarray):
    # Example: some complex operations resulting in a vector of length K
    return da.exp(subarray).mean(axis=1)

# Step 3: Apply the function
# Load data as Dask array from Zarr
dask_array = da.from_zarr('input_data.zarr')

# Apply function in parallel and store results directly to Zarr
result = dask_array.map_blocks(complex_function, dtype=float, chunks=(100, 10))
output_store = zarr.DirectoryStore('output_data.zarr')
result.to_zarr(output_store, mode='w', compute=True)

# Properly close client and cluster after computation
client.close()
cluster.close()





# Shut down the Dask client and cluster
client.close()
cluster.close()
