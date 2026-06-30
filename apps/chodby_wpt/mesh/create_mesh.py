"""Generate a local Flow123d-compatible mesh around one borehole interval.

The script reads ``input_data/mesh.yaml`` and creates ``.brep`` and ``.msh2``
files in ``workdir``. The geometry follows the same global coordinate
convention as ``chodby_inv``:

* X points in the lateral-tunnel direction, positive towards L6.
* Y follows the L5 main shaft stationing after applying ``y_offset``.
* Z is vertical and positive upwards.

The modeled interval is bounded by two packers around the configured measured
borehole section. Bulk physical regions are:

* ``rock`` - the surrounding cylinder after subtracting water and packers.
* ``water`` - the tested open borehole section.
* ``packer_near`` and ``packer_far`` - packer volumes.
* ``fracture_<index>`` - selected fracture surfaces, conformingly embedded in
  the rock mesh.

Boundary physical regions start with a dot. Only true external boundaries are
marked: ``.rock``, ``.packer_near``, ``.packer_far`` and ``.fracture_<index>``.
Internal rock-water, rock-packer and fracture-rock interfaces are left without
final boundary labels.
"""

import math
import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import input_data
from bgem.gmsh import field, gmsh, options
from endorse import common


output_dir = input_data.work_dir


def borehole_by_name(boreholes, name):
    """Return the borehole configuration with the requested name."""
    return next(bh for bh in boreholes if bh.name == name)


def borehole_start(bh, cfg):
    """Return the borehole collar point in the L5 coordinate system."""
    orientation = 1 if bh.id[0] == "P" else -1
    x = orientation * cfg.boreholes.geometry.l5_width / 2
    y = bh.stationing - cfg.boreholes.geometry.y_offset
    return np.array([x, y, bh.starting_height], dtype=float)


def borehole_direction(bh, cfg):
    """Return the unit vector of a borehole axis in the L5 coordinate system."""
    l5_azimuth = cfg.boreholes.geometry.l5_azimuth
    x = math.cos((l5_azimuth + 90 - bh.azimuth) * math.pi / 180)
    y = math.sin((l5_azimuth + 90 - bh.azimuth) * math.pi / 180)
    z = math.sin(bh.inclination * math.pi / 180)
    direction = np.array([x, y, z], dtype=float)
    return direction / np.linalg.norm(direction)


def fracture_normal(fracture, cfg):
    """Return a unit normal of a fracture plane in the L5 coordinate system."""
    l5_azimuth = cfg.boreholes.geometry.l5_azimuth
    x = math.cos((l5_azimuth + 90 - fracture.azimuth) * math.pi / 180)
    y = math.sin((l5_azimuth + 90 - fracture.azimuth) * math.pi / 180)
    z = math.sin(fracture.inclination * math.pi / 180)
    normal = np.array([x, y, z], dtype=float)
    return normal / np.linalg.norm(normal)


def section_distances(bh, section_index):
    """Return measured-distance limits of the open water interval."""
    start = bh.packer_centers[section_index] + bh.packer_width / 2
    end = bh.packer_centers[section_index + 1] - bh.packer_width / 2
    return start, end


def packer_distances(bh, packer_index):
    """Return measured-distance limits of a packer body."""
    start = bh.packer_centers[packer_index] - bh.packer_width / 2
    end = bh.packer_centers[packer_index] + bh.packer_width / 2
    return start, end


def domain_distances(bh, section_index):
    """Return measured-distance limits of the local cylindrical domain."""
    start, _ = packer_distances(bh, section_index)
    _, end = packer_distances(bh, section_index + 1)
    return start, end


def borehole_section(cfg):
    """Return borehole data, section index, collar point and axis direction."""
    section_cfg = cfg.borehole_section
    bh = borehole_by_name(cfg.boreholes.boreholes, section_cfg.borehole)
    direction = borehole_direction(bh, cfg)
    start = borehole_start(bh, cfg)
    return bh, section_cfg.section, start, direction


def points_at_distances(start, direction, distances):
    """Map measured distances on a borehole axis to 3D points."""
    return [start + distance * direction for distance in distances]


def section_points(cfg):
    """Return start and end points of the open water interval."""
    bh, section_index, bh_start, direction = borehole_section(cfg)
    start_distance, end_distance = section_distances(bh, section_index)
    return points_at_distances(bh_start, direction, [start_distance, end_distance])


def geometry_points(cfg):
    """Return domain and water-section endpoints in 3D."""
    bh, section_index, bh_start, direction = borehole_section(cfg)
    domain_start_distance, domain_end_distance = domain_distances(bh, section_index)
    section_start_distance, section_end_distance = section_distances(bh, section_index)
    domain_start, domain_end, section_start, section_end = points_at_distances(
        bh_start,
        direction,
        [domain_start_distance, domain_end_distance, section_start_distance, section_end_distance],
    )
    return domain_start, domain_end, section_start, section_end


def interval_points(cfg, distances):
    """Return 3D endpoints for an interval given by measured distances."""
    _, _, bh_start, direction = borehole_section(cfg)
    return points_at_distances(bh_start, direction, distances)


def fracture_disc_radius(center, domain_start, domain_end, cfg):
    """Return a conservative disk radius for cutting the domain cylinder."""
    endpoint_distance = max(
        np.linalg.norm(domain_start - center),
        np.linalg.norm(domain_end - center),
    )
    return endpoint_distance + cfg.geometry.domain_radius


def plane_intersects_cylinder(position, normal, direction, start_distance, end_distance, radius, tol=1e-9):
    """Return whether a fracture plane intersects the finite domain cylinder.

    ``position`` is the measured borehole distance where the plane intersects
    the borehole axis. The test intentionally accepts fractures whose borehole
    intersection is outside the modeled interval when the plane still cuts the
    surrounding domain cylinder.
    """
    normal_axis_projection = np.dot(normal, direction)
    signed_distances = [
        (start_distance - position) * normal_axis_projection,
        (end_distance - position) * normal_axis_projection,
    ]
    centerline_distance = (
        0
        if signed_distances[0] * signed_distances[1] <= 0
        else min(abs(distance) for distance in signed_distances)
    )
    radial_distance = radius * math.sqrt(max(0, 1 - normal_axis_projection ** 2))
    return centerline_distance <= radial_distance + tol


def fracture_name(fracture_index):
    """Return the physical-region name used for a borehole fracture index."""
    return f"fracture_{fracture_index}"


def selected_fractures(cfg):
    """Return selected fracture names or ``None`` for all configured fractures.

    The ``mesh.yaml`` value can be ``all``, a list of integer indices, or a list
    of explicit region names such as ``fracture_2``.
    """
    selection = cfg.get("fractures", "all")
    if selection == "all":
        return None
    return {
        fracture_name(item) if isinstance(item, int) else str(item)
        for item in selection
    }


def borehole_fractures(cfg):
    """Return fracture planes selected for the local model.

    The result contains ``(region_name, center, normal)`` triples. Selection is
    applied first, then planes are filtered by finite-cylinder intersection.
    """
    bh, section_index, bh_start, direction = borehole_section(cfg)
    domain_start_distance, domain_end_distance = domain_distances(bh, section_index)
    selected = selected_fractures(cfg)
    fractures = [
        (
            name,
            bh_start + fracture.position * direction,
            normal,
        )
        for idx, fracture in enumerate(bh.fractures)
        for name in [fracture_name(idx)]
        for normal in [fracture_normal(fracture, cfg)]
        if selected is None or name in selected
        if plane_intersects_cylinder(
            fracture.position,
            normal,
            direction,
            domain_start_distance,
            domain_end_distance,
            cfg.geometry.domain_radius,
        )
    ]
    return fractures


def borehole_cylinder(factory, cfg, start, end):
    """Create a cylinder with borehole radius between two axis points."""
    return factory.cylinder(r=cfg.geometry.borehole_radius, center=start, axis=end - start)


def select_boundary_at_distance(boundary, start, direction, distance, tol=1e-6):
    """Select boundary faces whose center lies at a given borehole distance."""
    dimtags = []
    regions = []
    direction = direction / np.linalg.norm(direction)
    for dimtag, region in boundary.dimtagreg():
        center = np.array(boundary.factory.model.getCenterOfMass(*dimtag))
        center_distance = np.dot(center - start, direction)
        if abs(center_distance - distance) < tol:
            dimtags.append(dimtag)
            regions.append(region)
    return gmsh.ObjectSet(boundary.factory, dimtags, regions)


def select_boundary_on_fractures(boundary, fractures, tol=1e-6):
    """Select boundary faces lying on any selected fracture plane."""
    dimtags = []
    regions = []
    for dimtag, region in boundary.dimtagreg():
        center = np.array(boundary.factory.model.getCenterOfMass(*dimtag))
        if any(abs(np.dot(center - fracture_center, normal)) < tol for _, fracture_center, normal in fractures):
            dimtags.append(dimtag)
            regions.append(region)
    return gmsh.ObjectSet(boundary.factory, dimtags, regions)


def curve_length(curve):
    """Return curve length, normalizing the list form returned by some objects."""
    mass = curve.get_mass()
    if isinstance(mass, list):
        return mass[1]
    return mass


def line_distance_edz(factory, line, cfg_mesh):
    """Create a mesh-size field around the tested borehole section."""
    line_length = curve_length(line)
    n_sampling = max(2, int(line_length / cfg_mesh.r_inner))
    dist = field.distance(line, sampling=n_sampling)
    inner = field.geometric(dist, a=(cfg_mesh.r_inner, cfg_mesh.h_inner), b=(cfg_mesh.r_outer, cfg_mesh.h_outer))
    outer = field.polynomial(dist, a=(cfg_mesh.r_outer, cfg_mesh.h_outer), b=(cfg_mesh.r_inf, cfg_mesh.h_inf), q=cfg_mesh.q_outer)
    return field.maximum(inner, outer)


def make_geometry(factory, cfg):
    """Build the OCC geometry and assign physical regions.

    The domain is fragmented together with the selected fracture surfaces. This
    makes fracture triangles share nodes and faces with adjacent tetrahedra, so
    Flow123d can detect fracture-bulk coupling from mesh topology.
    """
    bh, section_index, _, _ = borehole_section(cfg)
    domain_start, domain_end, section_start, section_end = geometry_points(cfg)
    domain_axis = domain_end - domain_start
    packer_near_start, packer_near_end = interval_points(cfg, packer_distances(bh, section_index))
    packer_far_start, packer_far_end = interval_points(cfg, packer_distances(bh, section_index + 1))
    fractures = borehole_fractures(cfg)

    domain = factory.cylinder(r=cfg.geometry.domain_radius, center=domain_start, axis=domain_axis)
    water = borehole_cylinder(factory, cfg, section_start, section_end)
    packer_near = borehole_cylinder(factory, cfg, packer_near_start, packer_near_end)
    packer_far = borehole_cylinder(factory, cfg, packer_far_start, packer_far_end)
    borehole_group = factory.group(water, packer_near, packer_far)
    fracture_surfaces = {
        name: factory.disc_discrete(
            fracture_disc_radius(center, domain_start, domain_end, cfg),
            center,
            axis=normal,
            n_points=12,
        )
        for name, center, normal in fractures
    }
    section_line = factory.line(factory.point(section_start), factory.point(section_end))
    factory.synchronize()

    # Fractures split only the rock. Borehole water and packer materials stay uncut.
    fracture_surfaces = {
        name: fracture.intersect(domain.copy()).cut(borehole_group.copy())
        for name, fracture in fracture_surfaces.items()
    }
    domain_fr, water_fr, packer_near_fr, packer_far_fr, *fractures_fr = factory.fragment(
        domain,
        water,
        packer_near,
        packer_far,
        *fracture_surfaces.values(),
    )
    rock = domain_fr.dt_drop(water_fr, packer_near_fr, packer_far_fr)
    rock.set_region("rock")
    water_fr.set_region("water")
    packer_near_fr.set_region("packer_near")
    packer_far_fr.set_region("packer_far")

    fractures_group = factory.group(*fractures_fr) if fractures_fr else None
    for name, fracture in zip(fracture_surfaces, fractures_fr):
        fracture.set_region(name)

    rock_boundary = rock.get_boundary().split_by_dimension()[2]
    borehole_boundary = water_fr.get_boundary().split_by_dimension()[2]
    packer_near_boundary = packer_near_fr.get_boundary().split_by_dimension()[2]
    packer_far_boundary = packer_far_fr.get_boundary().split_by_dimension()[2]
    _, _, bh_start, direction = borehole_section(cfg)
    packer_near_start_distance, _ = packer_distances(bh, section_index)
    _, packer_far_end_distance = packer_distances(bh, section_index + 1)

    water_rock_wall = rock_boundary.dt_intersection(borehole_boundary)
    water_rock_wall.set_region("__internal_water_rock").mesh_step(cfg.mesh.borehole_mesh_step)

    packer_near_rock_wall = rock_boundary.dt_intersection(packer_near_boundary)
    packer_near_external = select_boundary_at_distance(
        packer_near_boundary,
        bh_start,
        direction,
        packer_near_start_distance,
    )
    packer_near_external.set_region(".packer_near").mesh_step(cfg.mesh.borehole_mesh_step)

    packer_far_rock_wall = rock_boundary.dt_intersection(packer_far_boundary)
    packer_far_external = select_boundary_at_distance(
        packer_far_boundary,
        bh_start,
        direction,
        packer_far_end_distance,
    )
    packer_far_external.set_region(".packer_far").mesh_step(cfg.mesh.borehole_mesh_step)

    domain_boundary = rock_boundary.dt_drop(
        water_rock_wall,
        packer_near_rock_wall,
        packer_far_rock_wall,
    )
    if fractures_group is not None:
        domain_boundary.dt_drop(fractures_group)
        domain_boundary.dt_drop(select_boundary_on_fractures(domain_boundary, fractures))
    domain_boundary.set_region(".rock").mesh_step(cfg.mesh.boundary_mesh_step)

    domain_edges = domain_boundary.get_boundary().split_by_dimension()[1]
    fracture_external_boundaries = []
    for name, fracture in zip(fracture_surfaces, fractures_fr):
        fracture_boundary = fracture.get_boundary().split_by_dimension()[1]
        fracture_external_boundary = fracture_boundary.dt_intersection(domain_edges)
        if fracture_external_boundary.dim_tags:
            fracture_external_boundary.set_region(f".{name}").mesh_step(cfg.mesh.boundary_mesh_step)
            fracture_external_boundaries.append(fracture_external_boundary)

    refinement = line_distance_edz(factory, section_line, cfg.mesh.refinement)
    factory.set_mesh_step_field(refinement)

    geometry = factory.group(
        rock,
        water_fr,
        packer_near_fr,
        packer_far_fr,
        water_rock_wall,
        packer_near_external,
        packer_far_external,
        domain_boundary,
        *fractures_fr,
        *fracture_external_boundaries,
    )
    factory.synchronize()
    factory.keep_only(geometry, section_line)
    factory.synchronize()
    factory.remove_duplicate_entities()
    factory.synchronize()
    return geometry


def strip_physical_regions(mesh_file, names):
    """Remove temporary physical regions and their elements from an MSH2 file."""
    lines = mesh_file.read_text(encoding="utf-8").splitlines()
    physical_start = lines.index("$PhysicalNames")
    physical_end = lines.index("$EndPhysicalNames")

    stripped_ids = set()
    kept_names = []
    for line in lines[physical_start + 2:physical_end]:
        dim, physical_id, name = line.split(maxsplit=2)
        if name.strip('"') in names:
            stripped_ids.add(int(physical_id))
        else:
            kept_names.append(line)
    lines[physical_start + 1] = str(len(kept_names))
    lines[physical_start + 2:physical_end] = kept_names

    elements_start = lines.index("$Elements")
    elements_end = lines.index("$EndElements")
    kept_elements = []
    for line in lines[elements_start + 2:elements_end]:
        parts = line.split()
        n_tags = int(parts[2])
        physical_id = int(parts[3]) if n_tags else None
        if physical_id not in stripped_ids:
            kept_elements.append(line)
    lines[elements_start + 1] = str(len(kept_elements))
    lines[elements_start + 2:elements_end] = kept_elements
    mesh_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mesh_geometry(factory, geometry, cfg):
    """Mesh the geometry and write output files into ``workdir``."""
    factory.mesh_options.MinimumCirclePoints = 12
    factory.mesh_options.MinimumCurvePoints = 3
    factory.mesh_options.Algorithm = options.Algorithm3d.Delaunay

    brep_stem = output_dir / cfg.mesh_name
    mesh_file = output_dir / f"{cfg.mesh_name}.msh2"

    factory.write_brep(str(brep_stem))
    factory.make_mesh([geometry], dim=3)
    factory.write_mesh(filename=str(mesh_file), format=gmsh.MeshFormat.msh2)
    strip_physical_regions(mesh_file, {"__internal_water_rock"})
    return mesh_file


def make_mesh(cfg):
    """Create geometry, generate the mesh and return the output mesh file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    factory = gmsh.GeometryOCC(cfg.mesh_name, verbose=True)
    factory.get_logger().start()
    geometry = make_geometry(factory, cfg)
    mesh_file = mesh_geometry(factory, geometry, cfg)
    del factory
    return common.File(mesh_file)




if __name__ == "__main__":
    cfg = common.config.load_config(input_data.mesh_cfg_yaml)
    make_mesh(cfg)
