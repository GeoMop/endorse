from dask_kubernetes import KubeCluster
import dask.distributed

# Load the YAML spec into the KubeCluster
cluster = KubeCluster.from_yaml('worker-spec.yaml')

# Scale the cluster to the desired number of workers
cluster.scale(10)  # Scale the cluster to 10 workers

# Connect Dask to the cluster
client = dask.distributed.Client(cluster)
print(client)
