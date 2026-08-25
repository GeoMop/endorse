"""
Geometry is a cube of edge L_ext = L_ext_factor * L_inner, centered at the origin (bgem
convention), carrying the boundary conditions. The inner averaging cube is NOT part of it -
averaging volumes are postprocessing selections, so the mesh does not conform to them.

The DFN comes from a bgem Population (cfg.fractures.stochastic)
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from bgem.gmsh import gmsh, gmsh_io, heal_mesh, options
from bgem.stochastic import EllipseShape, FractureSet, Population
from bgem.stochastic import fr_mesh
from endorse.common import dotdict, File, memoize
from endorse.fullscale_transport import fracture_map
from endorse.mesh import mesh_tools
from endorse.mesh.fracture_tools import RegionFracture
from endorse.mesh_class import _load_mesh, load_mesh

def _ellipse_gmsh_base_shape(self, gmsh_geom):
    # bgem workaround: gmsh_base_shape references an undefined self.scale; the unit-area disc
    # radius is self.R
    return gmsh_geom.disc(rx=self.R, ry=self.R)


_orig_fracture_set_from_list = FractureSet.from_list.__func__


def _fracture_set_from_list(cls, fr_list):
    # bgem workaround: FractureSet.from_list discards each Fracture's own .family and sets it to 0
    fr_set = _orig_fracture_set_from_list(cls, fr_list)
    families = [getattr(fr, "family", None) for fr in fr_list]
    if any(f is not None for f in families):
        fr_set.family = np.array([0 if f is None else f for f in families], dtype=np.int32)
    return fr_set


@dataclass
class MicroMesh:
    # the healed mesh
    mesh_file: File
    # per-element cross_section field in a small file; Flow123d reads it via !FieldFE
    cross_section_file: File
    fractures: FractureSet
    # inner averaging window bounds: (lo, hi) so the window is [lo, hi]^3
    inner_box: Tuple[float, float]
    # outer cube edge
    L_ext: float
    regions: Dict[str, Tuple[int, int]]
    bulk_region: str
    fracture_region_prefix: str
    fracture_regions: List[str]

    @property
    def bulk_region_id(self) -> int:
        return self.regions[self.bulk_region][0]

    @property
    def fracture_region_ids(self) -> List[int]:
        return [self.regions[name][0] for name in self.fracture_regions]

    @property
    def fracture_radii(self) -> np.ndarray:
        if len(self.fractures) == 0:
            return np.array([])
        return self.fractures.radius[:, 0] * self.fractures.base_shape.R

    @property
    def fracture_radius_by_region_id(self) -> Dict[int, float]:
        radii = self.fracture_radii
        return {
            self.regions[name][0]: float(radii[int(name.rsplit("_", 1)[1])])
            for name in self.fracture_regions
        }

    def aperture_by_region_id(self, aperture_per_r: float) -> Dict[int, float]:
        return {rid: aperture_per_r * radius
                for rid, radius in self.fracture_radius_by_region_id.items()}


@memoize
def make_fractures(cfg_fractures: dotdict, box) -> FractureSet:
    # Sample the DFN for the given box (centered at origin).
    cfg_st = cfg_fractures.stochastic
    # bgem reads the size bounds as r_min/r_max; the config uses the report's r_0/r_infty.
    # bgem ignores keys it does not know, so its names just go in alongside
    families = {name: {**dotdict.serialize(fam), "r_min": fam.r_0, "r_max": fam.r_infty}
                for name, fam in cfg_st.population.items()}
    population = Population.from_cfg(families, box, shape=EllipseShape())
    # optional: None on either bound means "use the population's own range"
    sr = cfg_st.get("sample_range", None)
    sample_range = (float(sr[0]), float(sr[1])) if sr is not None else (None, None)
    fractures = mesh_tools.generate_fractures(
        population, sample_range, cfg_st.get("n_frac_limit", None), box, int(cfg_st.seed))
    fr_set = FractureSet.from_list(fractures)
    print(f"[upscale_m dfn] sampled {len(fr_set)} fracture(s), bgem sizes in "
          f"[{fr_set.radius[:, 0].min():.4g}, {fr_set.radius[:, 0].max():.4g}]")
    return fr_set


def _corner_tetrahedron(factory, corner: np.ndarray, d: float):
    # Small support tetrahedron for SUBC's rigid-body removal
    sign = np.sign(corner)
    arms = corner + d * sign * np.eye(3)
    p_common = factory.model.addPoint(*corner)
    p_arm = [factory.model.addPoint(*a) for a in arms]

    l_common = [factory.model.addLine(p_common, p_arm[i]) for i in range(3)]
    l_ring = [factory.model.addLine(p_arm[i], p_arm[(i + 1) % 3]) for i in range(3)]

    def face(loop):
        return factory.model.addPlaneSurface([factory.model.addCurveLoop(loop)])

    f_z = face([l_common[0], l_ring[0], -l_common[1]])
    f_x = face([l_common[1], l_ring[1], -l_common[2]])
    f_y = face([l_common[2], l_ring[2], -l_common[0]])
    f_outer = face([-l_ring[0], -l_ring[2], -l_ring[1]])

    vol = factory.model.addVolume([factory.model.addSurfaceLoop([f_x, f_y, f_z, f_outer])])
    factory._need_synchronize = True
    return factory.object(3, vol)


def _support_faces(factory, tet, corner: np.ndarray, axis_names: Dict[str, str], tol: float):
    boundary = tet.get_boundary()
    picked = {}
    for dim, tag in boundary.dim_tags:
        face_obj = factory.object(dim, tag)
        center, mass = face_obj.center_of_mass()
        if mass == 0:
            continue
        for axis, name in axis_names.items():
            if abs(center["xyz".index(axis)] - corner["xyz".index(axis)]) < tol:
                picked[name] = face_obj
    missing = set(axis_names.values()) - picked.keys()
    assert not missing, f"support face(s) {missing} not found at corner {corner}"
    return picked


def _support_spec(half: float) -> List[Tuple[np.ndarray, Dict[str, str]]]:
    # The 3 support corners (3-2-1 scheme: 3+2+1 = 6 constraints = the 6 rigid-body modes), on the
    # outer cube's bottom face
    return [
        (np.array([-half, -half, -half]),
         {"x": "support_origin_X", "y": "support_origin_Y", "z": "support_origin_Z"}),
        (np.array([half, -half, -half]),
         {"y": "support_tetra_one_norm_Y", "z": "support_tetra_one_norm_Z"}),
        (np.array([-half, half, -half]),
         {"z": "support_tetra_two_norm_Z"}),
    ]


def make_geometry(factory, cfg_geometry: dotdict, cfg_mesh: dotdict, fracture_set: FractureSet,
                  subc_support: bool = False):
    # Outer cube + fractures + named boundary sides (+ SUBC support tetrahedra)
    # Fractures keep bgem's per-fracture regions -- they must not be
    # joined, because the aperture field is keyed by region id

    L_ext = float(cfg_geometry.L_ext_factor) * float(cfg_geometry.L_inner)
    box_dims = [L_ext, L_ext, L_ext]

    box, sides = mesh_tools.box_with_sides(factory, box_dims)  # centered at origin
    box.set_region(cfg_geometry.bulk_region)
    geometry_set = [box]
    if len(fracture_set) > 0:
        fr_shapes, _region_map = fr_mesh.geometry_gmsh(fracture_set, factory)
        fr_group = fr_shapes.intersect(box).mesh_step(float(cfg_mesh.fracture_mesh_step))
        geometry_set.append(fr_group)
    else:
        print("[upscale_m mesh] WARNING: empty fracture set, mesh has no fracture regions")

    n_main = len(geometry_set)

    support_spec = _support_spec(L_ext / 2.0) if subc_support else []
    d = float(cfg_geometry.support_fraction_d) * float(cfg_geometry.L_inner)
    geometry_set += [_corner_tetrahedron(factory, corner, d) for corner, _ in support_spec]

    factory.synchronize()
    fragmented = factory.fragment(*geometry_set, *sides.values())
    geometry_final = fragmented[:n_main]

    if support_spec:
        support_tets = fragmented[n_main:n_main + len(support_spec)]
        for (corner, axis_names), tet in zip(support_spec, support_tets):
            faces = _support_faces(factory, tet, corner, axis_names, tol=1e-9 * L_ext)
            geometry_final += [f.set_region('.' + name) for name, f in faces.items()]
        geometry_final.append(
            factory.group(*support_tets).set_region(cfg_geometry.support_region))

    side_start = n_main + len(support_spec)
    for side_name, side_fr in zip(sides.keys(), fragmented[side_start:]):
        geometry_final.append(side_fr.set_region('.' + side_name))

    geometry_final = factory.group(*geometry_final)
    factory.synchronize()
    factory.keep_only(geometry_final)
    factory.synchronize()
    factory.remove_duplicate_entities()
    factory.synchronize()
    return geometry_final


def meshing(factory, objects, mesh_filename: str, cfg_mesh: dotdict):
    step = float(cfg_mesh.fracture_mesh_step)
    factory.mesh_options.CharacteristicLengthMin = step / float(cfg_mesh.mesh_size_min_fraction)
    factory.mesh_options.CharacteristicLengthMax = step
    factory.mesh_options.MinimumCirclePoints = int(cfg_mesh.min_circle_points)
    factory.mesh_options.MinimumCurvePoints = int(cfg_mesh.min_curve_points)
    factory.mesh_options.ToleranceInitialDelaunay = float(cfg_mesh.tolerance_initial_delaunay)
    factory.mesh_options.Algorithm = options.Algorithm3d.Delaunay

    factory.make_mesh(objects, dim=3, eliminate=False)
    factory.write_mesh(format=gmsh.MeshFormat.msh2)
    os.rename(factory.model_name + ".msh2", mesh_filename)


def make_gmsh(cfg_geometry: dotdict, cfg_mesh: dotdict, fracture_set: FractureSet,
              work_dir: str = ".", subc_support: bool = False) -> File:

    # Geometry and meshing in a gmsh model
    # The SUBC adds the support tetrahedra, while KUBC is completely unaffected
    mesh_name = cfg_mesh.mesh_name
    final_mesh_filename = os.path.join(work_dir, mesh_name + ".msh")

    factory = gmsh.GeometryOCC(mesh_name, verbose=False)
    factory.geom_options.Tolerance = float(cfg_mesh.tolerance)
    factory.geom_options.ToleranceBoolean = float(cfg_mesh.tolerance_boolean)

    geometry_final = make_geometry(factory, cfg_geometry, cfg_mesh, fracture_set, subc_support)

    print(f"[upscale_m mesh] meshing: L_ext={cfg_geometry.L_ext_factor}*"
          f"{cfg_geometry.L_inner}, step={cfg_mesh.fracture_mesh_step}, "
          f"{len(fracture_set)} fracture(s)")
    meshing(factory, [geometry_final], final_mesh_filename, cfg_mesh)

    factory.close()
    return File(final_mesh_filename)


def heal_mesh_preparation(mesh_name: str, mesh_file: File, work_dir: str = ".",
                   gamma_tol: float = 0.01) -> Path:
    mesh_file_healed = Path(work_dir) / (mesh_name + "_healed.msh")
    if not mesh_file_healed.exists():
        print("[upscale_m mesh] healing mesh ...")
        hm = heal_mesh.HealMesh.read_mesh(mesh_file.path)
        hm.heal_mesh(gamma_tol=gamma_tol)
        hm.write(file_name=str(mesh_file_healed))
        print(f"[upscale_m mesh] healed mesh written: {mesh_file_healed}")
    return mesh_file_healed


def make_healed_mesh(cfg_geometry: dotdict, cfg_mesh: dotdict, fracture_set: FractureSet,
              work_dir: str = ".", subc_support: bool = False) -> File:
    mesh_name = cfg_mesh.mesh_name
    raw_mesh_path = Path(work_dir) / (mesh_name + ".msh")
    if not raw_mesh_path.exists():
        mesh_file = make_gmsh(cfg_geometry, cfg_mesh, fracture_set, work_dir, subc_support)
    else:
        mesh_file = File(str(raw_mesh_path))

    mesh_file_healed = heal_mesh_preparation(mesh_name, mesh_file, work_dir, gamma_tol=float(cfg_mesh.heal_gamma_tol))
    return File(str(mesh_file_healed))


def make_cross_section_field(healed_file: File, fracture_set: FractureSet, aperture_per_r: float,
                             field_path: Path, fracture_region_prefix: str) -> None:
    mesh = load_mesh(healed_file)
    cross_section = np.ones(len(mesh.elements))
    fam_names = sorted((n for n in mesh.gmsh_io.physical
                        if n.startswith(fracture_region_prefix)),
                       key=lambda n: int(n.rsplit("_", 1)[1]))
    if fam_names:
        region_fractures = [
            RegionFracture(fracture_set[int(n.rsplit("_", 1)[1])], gmsh.Region.get(n))
            for n in fam_names]
        elm_to_ifr = fracture_map(mesh, region_fractures, n_large=0, dim=3)
        # geometric radii of the fractures; bgem size r = rho * sqrt(pi)
        rho = np.array([fr.r for fr in region_fractures]) * fracture_set.base_shape.R
        el_idx = np.fromiter(elm_to_ifr.keys(), dtype=int)
        i_fr = np.fromiter(elm_to_ifr.values(), dtype=int)
        cross_section[el_idx] = aperture_per_r * rho[i_fr]

    out_mesh = _load_mesh(File(healed_file.path), None)
    out_mesh.write_fields(str(field_path), dict(cross_section=cross_section))


def make_micro_mesh(cfg_geometry: dotdict, cfg_mesh: dotdict, cfg_fractures: dotdict,
                    aperture_per_r: float, work_dir: str = ".",
                    subc_support: bool = False,
                    fracture_box: Optional[List[float]] = None) -> MicroMesh:

    EllipseShape.gmsh_base_shape = _ellipse_gmsh_base_shape
    FractureSet.from_list = classmethod(_fracture_set_from_list)

    L = float(cfg_geometry.L_inner)
    factor = float(cfg_geometry.L_ext_factor)
    assert factor >= 1.0, f"L_ext_factor must be >= 1 (got {factor})"
    L_ext = factor * L

    fracture_set = make_fractures(cfg_fractures, fracture_box or [L_ext, L_ext, L_ext])
    healed_file = make_healed_mesh(cfg_geometry, cfg_mesh, fracture_set, work_dir, subc_support)

    frac_prefix = cfg_geometry.fracture_region_prefix
    mesh_name = cfg_mesh.mesh_name
    cross_section_path = Path(work_dir) / (mesh_name + "_cross_section.msh")
    if not cross_section_path.exists():
        make_cross_section_field(healed_file, fracture_set, float(aperture_per_r),
                                 cross_section_path, frac_prefix)
        print(f"[upscale_m mesh] cross_section field written: {cross_section_path}")

    regions = dict(gmsh_io.GmshIO(healed_file.path).physical)

    fracture_regions = [n for n in regions if n.startswith(frac_prefix)]

    return MicroMesh(
        mesh_file=healed_file,
        cross_section_file=File(str(cross_section_path)),
        fractures=fracture_set,
        inner_box=(-L / 2.0, L / 2.0),
        L_ext=L_ext,
        regions=regions,
        fracture_regions=fracture_regions,
        bulk_region=cfg_geometry.bulk_region,
        fracture_region_prefix=frac_prefix,
    )
