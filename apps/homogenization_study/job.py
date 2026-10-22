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
scratch = None
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


class Scratch(DotDir):
    """Node-local or local working directories."""

    _samples_dir = "samples"


class Output(DotDir):
    """Persistent study artifacts."""

    _input_data_dir = "input_data"
    _zarr_store_path = "homogenization_sampling"
    _mlmc_hdf_path = "homogenization_mlmc.hdf"
    _homogenization_mesh_dir = "homogenization_mesh"
    _plots_dir = "plots"
    _pbs_script = "homogenization_sampling.pbs"


def set_workdir(workdir: Path, input_dir: Path | None = None) -> None:
    """Initialize input, scratch, and persistent output paths for one run."""
    output_dir = Path(workdir).expanduser().resolve()
    scratch_env = os.environ.get("SCRATCHDIR")
    scratch_dir = (Path(scratch_env).expanduser().resolve() / "workdir"
                   if scratch_env and "/job" in scratch_env else output_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    if input_dir is None:
        input_dir = Path(os.environ.get(INPUT_DIR_ENV, output_dir / "input_data")).expanduser().resolve()
    else:
        input_dir = Path(input_dir).expanduser().resolve()
        os.environ[INPUT_DIR_ENV] = str(input_dir)
    os.environ[OUTPUT_DIR_ENV] = str(output_dir)

    global input, scratch, output
    input = Input(input_dir)
    scratch = Scratch(scratch_dir)
    output = Output(output_dir)


def to_str():
    s = '-' * 50 +"\n"
    def format_dir(d: DotDir):
        return f"{d.dir_path} [{'True' if d.dir_path.exists() else 'False'}]"
    s = s + f"input: {format_dir(input)}\n"
    s = s + f"scratch: {format_dir(scratch)}\n"
    s = s + f"output: {format_dir(output)}\n"
    s = s + '-' * 50
    return s
