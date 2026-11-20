"""Numeric DFN input, independent of BGEM and Gmsh."""
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class FamilyData:
    source: Path
    name: str
    arrays: Mapping[str, np.ndarray]
    source_rows: np.ndarray

    def __len__(self):
        return len(self.source_rows)

    @property
    def center(self):
        return self.vector('center')

    def vector(self, name):
        return np.column_stack([self.arrays[f'fracture_{name}_{s}'] for s in 'xyz'])

    def select(self, selection):
        return replace(self, arrays={k: v[selection] for k, v in self.arrays.items()},
                       source_rows=self.source_rows[selection])


@dataclass(frozen=True)
class NetworkData:
    bounds: np.ndarray  # (3, 2), lower/upper per axis
    families: tuple[FamilyData, ...]
    domain_idx: int = 0
    realization: int = 0
    coordinate_scale: float = 1.0

    def __len__(self):
        return sum(map(len, self.families))


def load_network(directory: Path, domain_idx=0, realization=0, coordinate_scale=1.0):
    """Load exactly one domain/realization, retaining original NPZ row identities."""
    directory = Path(directory)
    if coordinate_scale <= 0:
        raise ValueError('coordinate_scale must be positive')
    with (directory / 'DFN_gen-vars.yaml').open() as stream:
        metadata = yaml.safe_load(stream)
    if metadata['domain_types'][domain_idx] != 'cuboid':
        raise ValueError('Only cuboid domains are supported')
    bounds = np.asarray(metadata['domain_cuboid_xyzs'][domain_idx], dtype=float)*coordinate_scale
    families = []
    for name in metadata['domain_frsets_names'][domain_idx]:
        family_idx = metadata['frsets_names'].index(name)
        source = directory / f'ellipses_s{family_idx:02d}_d{domain_idx:03d}_r{realization:03d}.npz'
        with source.with_suffix('.yaml').open() as stream:
            info = yaml.safe_load(stream)
        if (info['fracture_set_idx'], info['domain_idx'], info['realization']) != (family_idx, domain_idx, realization):
            raise ValueError(f'Inconsistent source metadata: {source}')
        with np.load(source, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in info['column_names']}
        if ({v.shape for v in arrays.values()} != {(info['number_of_fractures'],)} or
                np.any(arrays['fracture_set_idx'] != family_idx)):
            raise ValueError(f'NPZ arrays disagree with metadata: {source}')
        dimensional = ['fracture_aaxis', 'fracture_req'] + [f'fracture_{v}_{s}' for v in ('center', 'rvec') for s in 'xyz']
        arrays = {k: v*coordinate_scale if k in dimensional else v for k, v in arrays.items()}
        family = FamilyData(source, name, arrays, np.arange(info['number_of_fractures']))
        # Projection to homogeneous vertex-count groups permits dense broadcasting.
        families.extend(family.select(np.flatnonzero(arrays['fracture_nverts'] == nv))
                        for nv in np.unique(arrays['fracture_nverts']))
    return NetworkData(bounds, tuple(families), domain_idx, realization, coordinate_scale)


def ellipse_polygon(center, normal, rvec, aaxis, ratio, nverts):
    """Reconstruct source polygons, including their documented angular sampling."""
    theta = np.pi/nverts + np.arange(nverts)*2*np.pi/nverts
    t = np.arctan2(ratio[:, None]*np.sin(theta), np.cos(theta))
    u = rvec/aaxis[:, None]
    v = np.cross(normal, u)
    return center[:, None, :] + aaxis[:, None, None]*(
        np.cos(t)[:, :, None]*u[:, None, :] +
        ratio[:, None, None]*np.sin(t)[:, :, None]*v[:, None, :])


def polygon_vertices(family: FamilyData):
    a = family.arrays
    return ellipse_polygon(family.center, family.vector('normal'), family.vector('rvec'),
                           a['fracture_aaxis'], a['fracture_ratio'],
                           int(a['fracture_nverts'][0]))


def polygon_area(vertices):
    # Translation before the cross product avoids cancellation in global coordinates.
    points = vertices - vertices[..., :1, :]
    return np.linalg.norm(np.cross(points, np.roll(points, -1, axis=-2)).sum(axis=-2), axis=-1)/2


def filter_by_radius(network: NetworkData, rmin=None, rmax=None):
    """Keep polygons with inclusive area-equivalent radii in scaled units.

    The source fracture_req is sqrt(polygon_area/pi), rather than the ellipse's
    semi-major axis or BGEM's shape-normalized radius. None leaves a bound open.
    """
    lower = 0.0 if rmin is None else rmin
    upper = np.inf if rmax is None else rmax
    if not 0 <= lower <= upper:
        raise ValueError('Radius bounds must satisfy 0 <= rmin <= rmax')
    selected = (family.select(np.flatnonzero(
        (family.arrays['fracture_req'] >= lower) &
        (family.arrays['fracture_req'] <= upper))) for family in network.families)
    return replace(network, families=tuple(family for family in selected if len(family)))


def crop_network(network: NetworkData, bounds):
    """Select conservatively by polygon AABB; actual clipping is done by OCC."""
    bounds = np.asarray(bounds, dtype=float)
    families = []
    for family in network.families:
        vertices = polygon_vertices(family)
        mask = np.all((vertices.max(axis=1) >= bounds[:, 0]) &
                      (vertices.min(axis=1) <= bounds[:, 1]), axis=1)
        selected = family.select(np.flatnonzero(mask))
        if len(selected):
            families.append(selected)
    return replace(network, bounds=bounds, families=tuple(families))


def compare_vtp(network: NetworkData, path: Path):
    """Compare source polygons to a VTP, independent of cell/cyclic vertex order."""
    import pyvista as pv
    from scipy.spatial import cKDTree

    mesh = pv.read(Path(path))
    centers = np.column_stack([mesh.cell_data[f'fracture_center_{s}'] for s in 'xyz'])
    family_ids = mesh.cell_data['fracture_set_idx']
    # VTK's cell accessor handles arbitrary polygon sizes as well as hexagons.
    matched = 0
    max_error = 0.0
    for family in network.families:
        idx = int(family.arrays['fracture_set_idx'][0])
        cells = np.flatnonzero(family_ids == idx)
        distance, nearest = cKDTree(centers[cells]).query(family.center/network.coordinate_scale)
        if np.any(distance > 1e-9) or len(np.unique(nearest)) != len(nearest):
            raise ValueError(f'VTP centers do not match {family.source.name}')
        for vertices, cell in zip(polygon_vertices(family), cells[nearest]):
            other = mesh.get_cell(int(cell)).points*network.coordinate_scale
            if vertices.shape != other.shape:
                raise ValueError(f'Vertex count differs at VTP cell {cell}')
            error = min(np.max(np.linalg.norm(vertices-np.roll(order, shift, axis=0), axis=1))
                        for order in (other, other[::-1]) for shift in range(len(other)))
            max_error = max(max_error, float(error))
            matched += 1
    if matched != mesh.n_cells or max_error > 1e-9*network.coordinate_scale:
        raise ValueError(f'VTP vertex mismatch: matched={matched}, max_error={max_error}')
    return {'matched': matched, 'max_vertex_error': max_error}


def clip_polygon_to_box(vertices, bounds):
    """Sutherland–Hodgman clipping of a convex 3-D planar polygon to a box."""
    points = np.asarray(vertices, dtype=float)
    for axis in range(3):
        for side, sign in ((0, 1), (1, -1)):
            if not len(points):
                return points
            distance = sign*(points[:, axis]-bounds[axis, side])
            inside = distance >= 0
            if np.all(inside):
                continue
            previous = np.roll(points, 1, axis=0)
            old_distance = np.roll(distance, 1)
            crossing = inside != np.roll(inside, 1)
            # Emit each entering/leaving intersection before its edge's endpoint.
            output = []
            for i in range(len(points)):
                if crossing[i]:
                    alpha = old_distance[i]/(old_distance[i]-distance[i])
                    intersection = previous[i] + alpha*(points[i]-previous[i])
                    intersection[axis] = bounds[axis, side]
                    output.append(intersection)
                if inside[i]:
                    output.append(points[i])
            points = np.asarray(output, dtype=float).reshape(-1, 3)
    if len(points):
        # A vertex on a clipping plane can be emitted both as intersection/endpoint.
        points = points[np.any(points != np.roll(points, 1, axis=0), axis=1)]
    return points
