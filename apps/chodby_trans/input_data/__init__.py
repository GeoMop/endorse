from pathlib import Path
import os

# common output directory
__script_dir__ = Path(__file__).parent

input_data_dirname = "input_data"
work_dirname = "workdir"

def resolve_dirs():
  path = os.environ.get("SCRATCHDIR", os.getcwd())
  if "SCRATCHDIR" in os.environ and "/job" in path:
    scratchdir = Path(os.environ.get("SCRATCHDIR", os.getcwd()))
    workdir = scratchdir / work_dirname
    inputdir = scratchdir / input_data_dirname
    outputdir = __script_dir__.parent / work_dirname
  else:
    workdir = __script_dir__.parent / work_dirname
    inputdir = __script_dir__.parent / input_data_dirname
    outputdir = workdir
  
  if not workdir.exists():
    workdir.mkdir(parents=True, exist_ok=True)
  if not outputdir.exists():
    outputdir.mkdir(parents=True, exist_ok=True)

  return workdir, inputdir, outputdir

# Following is public
work_dir, input_dir, output_dir = resolve_dirs()

transport_cfg_path = input_dir / "trans_mesh_config.yaml"

data_schema_yaml = input_dir / "data_schema.yaml"
data_schema_empty_yaml = input_dir / "data_schema_empty.yaml"

zarr_store_path = output_dir / "transport_sampling"

# transport_config = "trans_mesh_config.yaml"
# data_schema_yaml = "data_schema.yaml"
# data_schema_empty_yaml = "data_schema_empty.yaml"

sensitivity_dirname = "sensitivity"
param_dirname = "parameters"
empty_hdf_dirname = "empty_hdfs"
pbs_job_dirname = "pbs_jobs"

