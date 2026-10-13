"""Three-load homogenization problem for the conductivity field study."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .model import ConductivityFieldResult, StudyMeshFiles


def run_homogenization_problem(
    cfg: Mapping[str, Any],
    workdir: Path,
    meshes: StudyMeshFiles,
) -> ConductivityFieldResult:
    """Run three Flow123d loads and return the homogenized tensor field.

    Preserve the six-component ordering returned by the current
    ``macro_flow_model`` homogenization calculation.
    """
    raise NotImplementedError(
        "USER: implement the deterministic three-load homogenization problem."
    )
