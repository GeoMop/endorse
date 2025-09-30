"""
Purpose of the modele 'job.py':
- able to refactor the code to run within a given workdir that could change between runs
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
- resolve how to deal with zar_store path on the project directoy, ideally put whole output dir there.
"""
import attrs
from pathlib import Path


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
    _transport_cfg_path =

class Scratch(DotDir):
    _zarr_store_path = "transport_sampling"
    _sensitivity_dir = "sensitivity"
    _param_dir = "parameters"
    _empty_hdf_dir = "empty_hdfs"
    _pbs_job_dir = "pbs_jobs"

class Output(DotDir):
    _input_data_dir = "input_data"
    _work_dir = "workdir"
    _zarr_store_path = "transport_sampling"

def set_workdir(workdir: Path, ):
    """_summary_
    ?? zarr_store dir in project dir on meta ?
    Args:
        workdir (Path): _description_
    """
    global input, scratch, output
    input = Input(workdir / "input_data")
    scratch = Scratch(workdir)
    output = Output(workdir / "output")


