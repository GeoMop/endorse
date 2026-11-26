"""Map source regular polygons to the unit-area shapes of BGEM JB_homo."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bgem.stochastic import FractureSet, PolygonShape

from dfn_data import NetworkData


@dataclass(frozen=True)
class BgemGroup:
    fractures: FractureSet
    source: Path
    source_rows: np.ndarray
    family_name: str


def to_fracture_sets(network: NetworkData):
    groups = []
    for family in network.families:
        a = family.arrays
        if np.any(a['fracture_ratio'] != 1):
            raise ValueError('BGEM regular polygon conversion cannot represent non-circular source polygons exactly; use explicit vertices')
        nv = int(a['fracture_nverts'][0])
        shape = PolygonShape(nv)
        normal = family.vector('normal')
        u = family.vector('rvec')/a['fracture_aaxis'][:, None]
        phase = np.pi/nv
        target = np.cos(phase)*u + np.sin(phase)*np.cross(normal, u)
        # Use BGEM's own normal-rotation convention, including both Z poles.
        reference = np.broadcast_to([1., 0.], (len(family), 2)).copy()
        # The upstream routine corrects pole divisions immediately afterward.
        with np.errstate(invalid='ignore'):
            rotation = FractureSet.rotation_matrix_from_normals(normal, reference)
        local = np.einsum('nji,nj->ni', rotation, target)
        scales = np.repeat((a['fracture_aaxis']/shape.R)[:, None], 2, axis=1)
        fractures = FractureSet(base_shape_idx=nv, radius=scales, center=family.center,
                                normal=normal, shape_axis=local[:, :2],
                                family=a['fracture_set_idx'])
        groups.append(BgemGroup(fractures, family.source, family.source_rows, family.name))
    return groups
