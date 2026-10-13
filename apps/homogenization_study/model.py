"""Data contracts shared by the homogenization problem and sampling storage."""

from __future__ import annotations

from pathlib import Path

import attrs
import numpy as np


TENSOR_COMPONENTS = np.arange(1, 7, dtype=np.int8)
XYZ_COMPONENTS = ("x", "y", "z")


@attrs.frozen
class StudyMeshFiles:
    """Paths of the microscale problem mesh and the fixed homogenization mesh."""

    micro_mesh: Path = attrs.field(converter=Path)
    homogenization_mesh: Path = attrs.field(converter=Path)


@attrs.frozen
class ConductivityFieldResult:
    """One six-component conductivity tensor field on a fixed homogenization mesh."""

    conductivity: np.ndarray = attrs.field(converter=np.asarray)
    el_barycenter: np.ndarray = attrs.field(converter=np.asarray)
    gmsh_id: np.ndarray = attrs.field(converter=np.asarray)

    def __attrs_post_init__(self) -> None:
        """Reject arrays that cannot be stored in the agreed field layout."""
        if self.conductivity.ndim != 2 or self.conductivity.shape[1] != len(TENSOR_COMPONENTS):
            raise ValueError(
                "conductivity must have shape (n_elements, 6), "
                f"got {self.conductivity.shape}."
            )
        if self.el_barycenter.shape != (self.n_elements, len(XYZ_COMPONENTS)):
            raise ValueError(
                "el_barycenter must have shape (n_elements, 3), "
                f"got {self.el_barycenter.shape}."
            )
        if self.gmsh_id.shape != (self.n_elements,):
            raise ValueError(
                "gmsh_id must have shape (n_elements,), "
                f"got {self.gmsh_id.shape}."
            )
        if not np.issubdtype(self.gmsh_id.dtype, np.integer):
            raise ValueError(f"gmsh_id must use an integer dtype, got {self.gmsh_id.dtype}.")
        if np.unique(self.gmsh_id).size != self.n_elements:
            raise ValueError("gmsh_id values must be unique.")
        if not np.all(np.isfinite(self.conductivity)):
            raise ValueError("conductivity contains non-finite values.")
        if not np.all(np.isfinite(self.el_barycenter)):
            raise ValueError("el_barycenter contains non-finite values.")

    @property
    def n_elements(self) -> int:
        """Return the number of homogenization-mesh elements in the field."""
        return int(self.conductivity.shape[0])
