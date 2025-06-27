import pathlib

# common output driectory
__script_dir__ = pathlib.Path(__file__).parent
work_dir = __script_dir__.parent / "workdir"
work_dir.mkdir(parents=True, exist_ok=True)

# Following is public
input_dir = __script_dir__.parent / "input_data"

transport_config = input_dir / "trans_mesh_config.yaml"

sensitivity_dirname = "sensitivity"
param_dirname = "parameters"
empty_hdf_dirname = "empty_hdfs"
pbs_job_dirname = "pbs_jobs"


