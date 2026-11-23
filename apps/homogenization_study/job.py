"""Run-directory paths for the homogenization study.

The structure mirrors the Chodby application where the concepts are the same,
while exposing only artifacts owned by this study.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import attrs


INPUT_DIR_ENV = "ENDORSE_INPUT_DIR"
OUTPUT_DIR_ENV = "ENDORSE_OUTPUT_DIR"

input = None
output = None


@attrs.define(frozen=True)
class DotDir:
    """Resolve declared artifact suffixes relative to one run directory."""

    dir_path: Path

    def __getattr__(self, name: str) -> Any:
        """Return the path declared by the corresponding private class attribute."""
        try:
            suffix = getattr(type(self), f"_{name}")
        except AttributeError as error:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}") from error
        return self.dir_path / suffix


class Input(DotDir):
    """Input files copied into a self-contained study run."""

    _study_cfg_path = "homogenization_study.yaml"
    _data_schema_yaml = "data_schema.yaml"


class Scratch(DotDir):
    """Node-local or local working directories."""

    _mlmc_output_dir = "output"


class Output(DotDir):
    """Persistent study artifacts."""

    _input_data_dir = "input_data"
    _zarr_store_path = "homogenization_sampling"
    _mlmc_hdf_path = "homogenization_mlmc.hdf"
    _homogenization_mesh_dir = "homogenization_mesh"
    _plots_dir = "plots"
    _pbs_script = "homogenization_sampling.pbs"


def in_pbs_job() -> bool:
    """Return whether this process belongs to a PBS job."""
    return bool(os.environ.get("PBS_JOBID"))


def _scratch(local_output: Output | None) -> Scratch | None:
    if in_pbs_job() or "SCRATCHDIR" in os.environ:
        allocation = Path(os.environ["SCRATCHDIR"])
        if not allocation.is_absolute() or not allocation.is_dir():
            raise ValueError(f"Expected an existing scratch directory, got {allocation}")
        return Scratch(allocation / "workdir")
    return Scratch(local_output.dir_path) if local_output is not None else None


def __getattr__(name: str) -> Any:
    """Resolve job.scratch from the environment of the current process on access."""
    if name == "scratch":
        return _scratch(output)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def set_workdir(workdir: Path, input_dir: Path | None = None) -> None:
    """Initialize input, scratch, and persistent output paths for one run."""
    output_dir = Path(workdir).expanduser().resolve()
    output_paths = Output(output_dir)
    scratch_paths = _scratch(output_paths)
    assert scratch_paths is not None
    scratch_paths.dir_path.mkdir(parents=True, exist_ok=True)

    if input_dir is None:
        input_dir = Path(os.environ.get(INPUT_DIR_ENV, output_dir / "input_data")).expanduser().resolve()
    else:
        input_dir = Path(input_dir).expanduser().resolve()
        os.environ[INPUT_DIR_ENV] = str(input_dir)
    os.environ[OUTPUT_DIR_ENV] = str(output_dir)

    global input, output
    input = Input(input_dir)
    output = output_paths


def to_str():
    s = '-' * 50 +"\n"
    def format_dir(d: DotDir):
        return f"{d.dir_path} [{'True' if d.dir_path.exists() else 'False'}]"
    s = s + f"input: {format_dir(input)}\n"
    s = s + f"scratch: {format_dir(_scratch(output))}\n"
    s = s + f"output: {format_dir(output)}\n"
    s = s + '-' * 50
    return s
