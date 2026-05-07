import dask.array as da
import zarr
from dask.distributed import Client, LocalCluster

# Define the function F (applies to rows)
def function_F(row):
    # Assuming function F results in a vector of length K from each row
    return da.exp(row).sum()

# Define the function G (applies to columns)
def function_G(column):
    # Assuming function G results in a single value from each column
    return column.mean()

# Setup the Dask client
cluster = LocalCluster()
client = Client(cluster)

# Step 1: Prepare or load the initial data
input_store = zarr.DirectoryStore('input_data.zarr')
dask_array = da.from_zarr(input_store)

# Step 2: Apply function F to each row
result_F = dask_array.map_blocks(lambda block: da.apply_along_axis(function_F, 1, block),
                                 dtype=float, chunks=(1, 1))

# Store result_F in a Zarr file
intermediate_store = zarr.DirectoryStore('intermediate_data.zarr')
result_F.to_zarr(intermediate_store, compute=False)

# Step 3: Load result_F back for further processing
result_F = da.from_zarr(intermediate_store)

# Step 4: Rechunk data for column-wise operations (if necessary)
result_F = result_F.rechunk({0: result_F.shape[0], 1: 'auto'})

# Step 5: Apply function G to each column
result_G = result_F.map_blocks(lambda block: da.apply_along_axis(function_G, 0, block),
                               dtype=float, chunks=(1, 1))

# Step 6: Store the final result in a new Zarr file
output_store = zarr.DirectoryStore('output_data.zarr')
result_G.to_zarr(output_store, compute=True)

# Close the resources
client.close()
cluster.close()
