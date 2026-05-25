from dask.distributed import Client
from dask_jobqueue import PBSCluster

# Define the PBSCluster to use a Singularity container
cluster = PBSCluster(
    cores=24,
    memory="100GB",
    walltime="02:00:00",
    queue='batch',
    job_extra=[
        'source /etc/profile',  # Load necessary modules or environment variables
    ],
    # Specify the Singularity command to run the worker
    worker_command="singularity exec /path/to/your/container.sif /usr/local/bin/python -m distributed.cli.dask_worker"
)

# Start the cluster
cluster.scale(jobs=10)  # Adjust number of jobs based on your needs

# Connect the client
client = Client(cluster)

# Assume function F and data preparation from earlier examples
# Here, insert the data processing tasks that apply function F

# Close the client and cluster after finishing computations
client.close()
cluster.close()
