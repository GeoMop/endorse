"""OCC CAD healing, source-face recovery, and geometric feature-size reports."""
import gmsh
import numpy as np


def _range(values):
    """Represent empty entity collections without non-JSON infinities."""
    return dict(count=len(values), min=min(values, default=None), max=max(values, default=None))


def geometry_metrics():
    """Measure all CAD faces/volumes, retaining boundary-shell multiplicity.

    Both area/perimeter and volume/surface_area have units of length. All loops
    (including holes) contribute to a face's perimeter. Shared faces contribute
    to the surface area of each incident volume.
    """
    entities = {dim: gmsh.model.getEntities(dim) for dim in range(4)}
    mass = {dt: gmsh.model.occ.getMass(*dt)
            for dim in (1, 2, 3) for dt in entities[dim]}

    def ratios(dim):
        return [float(mass[dt]/sum(mass[boundary] for boundary in gmsh.model.getBoundary(
            [dt], combined=False, oriented=False))) for dt in entities[dim]]

    return dict(entities={str(dim): len(items) for dim, items in entities.items()},
                area_perimeter=_range(ratios(2)), volume_surface_area=_range(ratios(3)))


def _polygon_frame(polygon):
    """Return a plane normal and inward edge normals for a convex source face."""
    edges = np.roll(polygon, -1, axis=0)-polygon
    normal = np.cross(polygon-polygon[0], np.roll(polygon, -1, axis=0)-polygon[0]).sum(axis=0)
    normal /= np.linalg.norm(normal)
    inward = np.cross(normal, edges)
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    return normal, inward


def heal_cad(polygons, tolerance):
    """Heal the whole shared CAD model and recover source IDs on new face tags.

    Sewing/rebuilding solids is deliberately disabled: those operations can
    discard internal fracture sheets or destroy the bulk's partitioned solids.
    The bulk domain is a cuboid and source polygons are convex and planar.
    Geometry tests, rather than old OCC entity tags, determine ownership after
    the kernel renumbers the model.
    """
    original_vertices = np.array([gmsh.model.getValue(dim, tag, [])
                                  for dim, tag in gmsh.model.getEntities(0)])
    bounds = np.array([original_vertices.min(axis=0), original_vertices.max(axis=0)])
    gmsh.model.occ.healShapes(tolerance=tolerance, fixDegenerated=True,
                             fixSmallEdges=True, fixSmallFaces=True,
                             sewFaces=False, makeSolids=False)
    gmsh.model.occ.synchronize()
    if not gmsh.model.getEntities(3):
        raise ValueError('OCC healing removed all bulk solids')
    if not polygons:
        return {}

    names = list(polygons)
    points = [polygons[name] for name in names]
    frames = [_polygon_frame(polygon) for polygon in points]
    normals = np.array([normal for normal, inward in frames])
    origins = np.array([polygon[0] for polygon in points])
    lower = np.array([polygon.min(axis=0) for polygon in points])-tolerance
    upper = np.array([polygon.max(axis=0) for polygon in points])+tolerance
    vertices = {tag: gmsh.model.getValue(dim, tag, []) for dim, tag in gmsh.model.getEntities(0)}
    precision = 128*np.finfo(float).eps*np.max(np.abs(np.concatenate(points)))
    ownership = {}
    for dimtag in gmsh.model.getEntities(2):
        dim, tag = dimtag
        center = np.array(gmsh.model.occ.getCenterOfMass(dim, tag))
        uv = gmsh.model.getParametrization(dim, tag, center)
        normal = gmsh.model.getNormal(tag, uv)
        candidates = np.flatnonzero(
            np.all((center >= lower) & (center <= upper), axis=1) &
            (np.abs(np.einsum('ij,ij->i', center-origins, normals)) <= tolerance) &
            (np.abs(np.einsum('ij,j->i', normals, normal)) >= 1-1e-8))
        if not len(candidates):
            continue
        boundary = gmsh.model.getBoundary([dimtag], combined=False, oriented=False, recursive=True)
        face_points = np.array([vertices[node] for _, node in boundary])
        owners = []
        for i in candidates:
            source_normal, inward = frames[i]
            distances = np.einsum('nej,ej->ne', face_points[:, None, :]-points[i], inward)
            residual = max(float(np.max(np.abs((face_points-origins[i]) @ source_normal))),
                           max(0.0, -float(distances.min())))
            if residual <= tolerance:
                owners.append((names[i], residual))
        if not owners:
            continue
        # Prefer the preserved plane/polygon over another source merely within
        # the healing radius. Nearby parallel sheets must keep distinct owners.
        best = min(residual for source, residual in owners)
        # The cuboid's exterior is also an ownership candidate. An unchanged
        # box face must not become a nearby internal fracture merely because
        # their separation is smaller than the healing tolerance. Exact source
        # matches win ties so fractures genuinely on the exterior stay labelled.
        boundary_distance = np.max(np.abs(face_points[:, None, :]-bounds), axis=0).min()
        if boundary_distance+precision < best:
            continue
        closest = [source for source, residual in owners if residual <= best+precision]
        if len(closest) > 1:
            raise ValueError(f'Ambiguous source ownership after OCC healing: {dimtag}: {closest}')
        ownership[dimtag] = closest[0]
    return ownership
