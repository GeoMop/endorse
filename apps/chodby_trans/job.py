"""
Purpose of the module 'job.py':
- enable to refactor the code to run within a given workdir that could change between runs
  but contains also all inputs so it is independent on their change
- document structure of input, scratch and output directories
- avoid hardcoded paths in the code

Usage simply replace current 'input_data.some_path_var' by:

job.input.some_path_var
or
job.scratch.some_path_var
or
job.output.some_path_var

TODO:

- use anv SCRATCHDIR if available and resolve meta vs. local setting in the set_workdir function
- resolve how to deal with zar_store path on the project directory, ideally put whole output dir there.
"""
import os
import attrs
from pathlib import Path
from typing import Any

input = None
scratch = None
output = None



@attrs.define
class DotDir:
    dir_path: Path

    def __getattr__(self, name: str) -> Any:
        # Look up the *class* attribute named f"_{name}" without calling __getattr__ again
        try:
            suffix = getattr(type(self), f"_{name}")
        except AttributeError as e:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}") from e
        return self.dir_path / Path(suffix)


class Input(DotDir):
    _data_schema_yaml = "data_schema.yaml"
    _data_schema_empty_yaml = "data_schema_empty.yaml"
    _transport_cfg_path = "trans_mesh_config.yaml"

class Scratch(DotDir):
    _zarr_store_path = "transport_sampling"
    _sensitivity_dir = "sensitivity"
    _param_dir = "parameters"
    # _empty_hdf_dir = "empty_hdfs"
    # _pbs_job_dir = "pbs_jobs"

class Output(DotDir):
    _input_data_dir = "input_data"
    # _work_dir = "workdir"
    _sensitivity_dir = "sensitivity"
    _plots = "plots"
    _pbs_script = "sensitivity_sampling.pbs"
    _zarr_store_path = "transport_sampling"


def set_workdir(workdir: Path, ):
    """_summary_
    ?? zarr_store dir in project dir on meta ?
    Args:
        workdir (Path): _description_
    """
    path = os.environ.get("SCRATCHDIR", os.getcwd())
    if "SCRATCHDIR" in os.environ and "/job" in path:
        work_dirname = "workdir"
        scratchdir = Path(os.environ.get("SCRATCHDIR", os.getcwd())) / work_dirname
        if not scratchdir.exists():
            scratchdir.mkdir(parents=True, exist_ok=True)
    else:
        scratchdir = workdir

    global input, scratch, output
    input = Input(workdir / "input_data")
    scratch = Scratch(scratchdir)
    output = Output(workdir)


def to_str():
    s = '-' * 50 +"\n"
    def format_dir(d: DotDir):
        return f"{d.dir_path} [{'True' if d.dir_path.exists() else 'False'}]"
    s = s + f"input: {format_dir(input)}\n"
    s = s + f"scratch: {format_dir(scratch)}\n"
    s = s + f"output: {format_dir(output)}\n"
    s = s + '-' * 50
    return s
