"""
Micro mesh generation for mechanical upscaling (apps/upscale_m).

Geometry: two concentric cubes. The outer cube [0, L_ext]^3 carries the boundary conditions, the
inner cube (edge L, centered) is the homogenization/averaging volume. Default L_ext = 2 L gives a
buffer shell of thickness L/2 on every face: BC artifacts decay across the buffer (Saint-Venant)
before reaching the averaging volume.

The domain is fragmented by the inner cube, so the final mesh CONFORMS to the averaging-volume
boundary: every bulk and fracture element lies entirely inside or outside it, making the inner-cube
selection exact (by region for bulk, by barycenter for fracture elements).

Physical regions in the healed mesh (msh2, Flow123d ready):
    rock_inner  (3D, tag 1)  rock inside the averaging cube
    rock_outer  (3D, tag 2)  buffer rock
    fractures   (2D, tag 3)  fracture elements (whole domain, clipped to the outer cube)
    .surface_[xyz]_[minus|plus]  (2D)  outer boundary faces for BC prescription
      (names match R. Siddall's existing Flow123d templates)

Fracture input: explicit list of penny-shaped fractures (r, center, normal). Stochastic population
sampling is a separate step (PLAN.md task 4; bgem@JB_homo API differs from the original code).

Meshing options are ported from R. Siddall's proven mesher (src_internship_cvut). Entity
classification after boolean fragmentation uses the occ.fragment output map instead of geometric
heuristics, so axis-aligned fractures cannot be confused with the inner-cube interface.
"""
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import gmsh as raw_gmsh

from bgem.gmsh import heal_mesh
from endorse.common import dotdict, File

# geometric tolerance for bounding-box classification (fraction of unit cube)
EPS_GEO = 1e-4


@dataclass
class MicroFracture:
    """One penny-shaped fracture: radius, center and unit normal (in outer-domain coordinates)."""
    r: float
    center: np.ndarray
    normal: np.ndarray


@dataclass
class MicroMesh:
    """Result of make_micro_mesh."""
    mesh_file: File
    # healed msh2 mesh with the regions documented in the module docstring
    fractures: List[MicroFracture]
    # inner averaging cube bounds: (lo, hi) so the cube is [lo, hi]^3
    inner_box: Tuple[float, float]
    # outer cube edge
    L_ext: float
    # True when a buffer shell exists (region 'rock_outer' present in the mesh)
    has_buffer: bool = True
    # subdomain grid inside the averaging cube (mirrors macro_conductivity's per-subdomain
    # tensors): (nx, ny, nz) and one (lo_xyz, hi_xyz) box per cell, z-fastest order. Cells are
    # imprinted into the mesh (conforming), so per-cell averaging stays exact — no weights needed.
    subdivision: Tuple[int, int, int] = (1, 1, 1)
    subdomain_boxes: List[Tuple[np.ndarray, np.ndarray]] = None

    def subdomain_index(self, i: int) -> Tuple[int, int, int]:
        """Flat subdomain index -> (ix, iy, iz)."""
        nx, ny, nz = self.subdivision
        return i // (ny * nz), (i // nz) % ny, i % nz


def fractures_from_config(cfg_fractures: dotdict, L_domain: float) -> List[MicroFracture]:
    """Fracture set for the outer domain [0, L_domain]^3, per cfg (explicit list or stochastic)."""
    mode = str(cfg_fractures.mode).strip().lower()
    if mode == "explicit":
        fractures = []
        for item in cfg_fractures.explicit_list:
            normal = np.asarray(item["normal"], dtype=float)
            norm = np.linalg.norm(normal)
            assert norm > 0.0, f"Zero fracture normal: {item}"
            fractures.append(MicroFracture(
                r=float(item["r"]),
                center=np.asarray(item["center"], dtype=float),
                normal=normal / norm))
        return fractures
    elif mode == "stochastic":
        from stochastic_dfn import stochastic_fracture_set
        return stochastic_fracture_set(cfg_fractures.stochastic, L_domain)
    else:
        raise ValueError(f"Unknown fractures.mode: {cfg_fractures.mode!r}")


def _add_rotated_disk(occ, fr: MicroFracture) -> int:
    """Add a disk of radius fr.r at fr.center, rotated from the base normal [0,0,1] to fr.normal."""
    tag = occ.addDisk(fr.center[0], fr.center[1], fr.center[2], fr.r, fr.r)
    base = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(base, fr.normal), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        pass  # already oriented
    elif dot < -1.0 + 1e-9:
        occ.rotate([(2, tag)], *fr.center, 1.0, 0.0, 0.0, np.pi)
    else:
        axis = np.cross(base, fr.normal)
        axis = axis / np.linalg.norm(axis)
        occ.rotate([(2, tag)], *fr.center, axis[0], axis[1], axis[2], float(np.arccos(dot)))
    return tag


def _bbox_outside(dim_tag, L_ext: float, eps: float) -> bool:
    xmin, ymin, zmin, xmax, ymax, zmax = raw_gmsh.model.getBoundingBox(*dim_tag)
    return (xmin < -eps or xmax > L_ext + eps or
            ymin < -eps or ymax > L_ext + eps or
            zmin < -eps or zmax > L_ext + eps)


def make_micro_mesh(cfg_geometry: dotdict, cfg_mesh: dotdict, cfg_fractures: dotdict,
                    work_dir: str = ".") -> MicroMesh:
    """
    Build, mesh and heal the buffered fractured cube. Returns MicroMesh with the healed msh file.
    All output files are written into `work_dir`.
    """
    L = float(cfg_geometry.L_inner)
    factor = float(cfg_geometry.get("L_ext_factor", 2.0))
    assert factor >= 1.0, f"L_ext_factor must be >= 1 (got {factor})"
    L_ext = factor * L
    lo = (L_ext - L) / 2.0
    hi = lo + L
    eps = EPS_GEO * L_ext
    embed_inner = factor > 1.0 + 1e-12  # factor 1.0 => no buffer, no inner box to embed

    # subdomain grid inside the averaging cube: one effective tensor per cell (default: one cell)
    sub = cfg_geometry.get("subdomains", [1, 1, 1])
    nx, ny, nz = (int(v) for v in sub)
    assert nx >= 1 and ny >= 1 and nz >= 1, f"subdomains must be positive ints, got {sub}"
    n_cells = nx * ny * nz
    cell = np.array([L / nx, L / ny, L / nz])
    subdomain_boxes = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                lo3 = np.array([lo, lo, lo]) + np.array([ix, iy, iz]) * cell
                subdomain_boxes.append((lo3, lo3 + cell))

    fractures = fractures_from_config(cfg_fractures, L_ext)

    mesh_name = cfg_mesh.get("mesh_name", "micro_mesh")
    mesh_path = os.path.join(work_dir, mesh_name + ".msh")

    if raw_gmsh.isInitialized():
        raw_gmsh.clear()
    else:
        raw_gmsh.initialize()
    occ = raw_gmsh.model.occ

    raw_gmsh.option.setNumber("Geometry.Tolerance", float(cfg_mesh.get("tolerance", 1e-6)))
    raw_gmsh.option.setNumber("Geometry.ToleranceBoolean", float(cfg_mesh.get("tolerance_boolean", 1e-5)))

    # --- geometry --------------------------------------------------------------------------
    outer_tag = occ.addBox(0, 0, 0, L_ext, L_ext, L_ext)
    tools = []
    n_cell_tools = 0
    if n_cells > 1:
        # imprint every subdomain cell (their union = the inner cube, so no separate inner box)
        for lo3, hi3 in subdomain_boxes:
            size3 = hi3 - lo3
            tools.append((3, occ.addBox(lo3[0], lo3[1], lo3[2], size3[0], size3[1], size3[2])))
        n_cell_tools = n_cells
    elif embed_inner:
        tools.append((3, occ.addBox(lo, lo, lo, L, L, L)))
        n_cell_tools = 1
    disk_tags = [_add_rotated_disk(occ, fr) for fr in fractures]
    tools += [(2, t) for t in disk_tags]

    # Fragment: makes everything conforming and returns the input -> output entity map.
    # Map entries follow the input order: [outer_box, (cell boxes...), disk_0, disk_1, ...]
    _, out_map = occ.fragment([(3, outer_tag)], tools)
    occ.synchronize()

    all_vols = [tag for dim, tag in out_map[0] if dim == 3]
    if n_cell_tools:
        inner_vols = sorted({tag for k in range(1, 1 + n_cell_tools)
                             for dim, tag in out_map[k] if dim == 3})
        i_disk_map = 1 + n_cell_tools
    else:
        inner_vols = all_vols
        i_disk_map = 1
    inner_set = set(inner_vols)
    outer_vols = [t for t in all_vols if t not in inner_set]
    assert inner_vols, "No inner (averaging) volumes after fragmentation."

    # Fracture pieces: keep those inside the outer cube, delete protruding remnants.
    kept_fracture_surfs = []
    to_remove = []
    for i in range(len(disk_tags)):
        for dim, tag in out_map[i_disk_map + i]:
            if dim != 2:
                continue
            if _bbox_outside((dim, tag), L_ext, eps):
                to_remove.append((dim, tag))
            else:
                kept_fracture_surfs.append(tag)
    if to_remove:
        occ.remove(to_remove, recursive=True)
        occ.synchronize()
        print(f"[upscale_m mesh] clipped {len(to_remove)} fracture fragment(s) outside the domain")
    assert kept_fracture_surfs or not fractures, "All fracture surfaces were clipped away."

    # --- boundary faces of the outer cube --------------------------------------------------
    def faces_in_slab(x0, y0, z0, x1, y1, z1):
        ents = raw_gmsh.model.getEntitiesInBoundingBox(
            x0 - eps, y0 - eps, z0 - eps, x1 + eps, y1 + eps, z1 + eps, dim=2)
        return [tag for _, tag in ents]

    E = L_ext
    boundary_faces = {
        ".surface_x_minus": faces_in_slab(0, 0, 0, 0, E, E),
        ".surface_x_plus":  faces_in_slab(E, 0, 0, E, E, E),
        ".surface_y_minus": faces_in_slab(0, 0, 0, E, 0, E),
        ".surface_y_plus":  faces_in_slab(0, E, 0, E, E, E),
        ".surface_z_minus": faces_in_slab(0, 0, 0, E, E, 0),
        ".surface_z_plus":  faces_in_slab(0, 0, E, E, E, E),
    }
    for name, tags in boundary_faces.items():
        assert tags, f"No boundary faces found for {name}"

    # --- physical regions ------------------------------------------------------------------
    raw_gmsh.model.addPhysicalGroup(3, inner_vols, tag=1, name="rock_inner")
    if outer_vols:
        raw_gmsh.model.addPhysicalGroup(3, outer_vols, tag=2, name="rock_outer")
    if kept_fracture_surfs:
        raw_gmsh.model.addPhysicalGroup(2, kept_fracture_surfs, tag=3, name="fractures")
    for name, tags in boundary_faces.items():
        raw_gmsh.model.addPhysicalGroup(2, tags, name=name)

    # --- meshing --------------------------------------------------------------------------
    step = float(cfg_mesh.fracture_mesh_step)
    min_fraction = float(cfg_mesh.get("mesh_size_min_fraction", 10.0))
    raw_gmsh.option.setNumber("Mesh.MeshSizeMin", step / min_fraction)
    raw_gmsh.option.setNumber("Mesh.MeshSizeMax", step)
    raw_gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    raw_gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    raw_gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 2)
    raw_gmsh.option.setNumber("Mesh.MinimumCirclePoints", 6)
    raw_gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay",
                              float(cfg_mesh.get("tolerance_initial_delaunay", 0.01)))
    raw_gmsh.option.setNumber("Mesh.Algorithm", 6)
    # Flow123d / bgem GmshIO consume the legacy v2.2 format.
    raw_gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    print(f"[upscale_m mesh] meshing: L={L}, L_ext={L_ext}, step={step}, "
          f"{len(fractures)} fracture(s), subdomains {nx}x{ny}x{nz}")
    raw_gmsh.model.mesh.generate(3)
    raw_gmsh.write(mesh_path)

    if str(cfg_mesh.get("show_gui", "no")).lower() in ("yes", "y", "true"):
        try:
            raw_gmsh.fltk.run()
        except Exception as e:
            print(f"[upscale_m mesh] gmsh GUI unavailable: {e}")

    # --- healing (required for stable Flow123d runs) ----------------------------------------
    print("[upscale_m mesh] healing mesh ...")
    hm = heal_mesh.HealMesh.read_mesh(mesh_path)
    hm.heal_mesh(gamma_tol=float(cfg_mesh.get("heal_gamma_tol", 0.01)))
    healed_path = os.path.join(work_dir, mesh_name + "_healed.msh")
    hm.write(healed_path)
    print(f"[upscale_m mesh] healed mesh written: {healed_path}")

    return MicroMesh(
        mesh_file=File(healed_path),
        fractures=fractures,
        inner_box=(lo, hi),
        L_ext=L_ext,
        has_buffer=bool(outer_vols),
        subdivision=(nx, ny, nz),
        subdomain_boxes=subdomain_boxes,
    )
