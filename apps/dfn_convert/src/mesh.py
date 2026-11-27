"""Conforming polygon/bulk meshing with BGEM's OpenCASCADE wrapper."""
from dataclasses import dataclass, fields
import importlib.metadata
import json
import logging
from pathlib import Path
import resource
import time

import gmsh
import numpy as np
from bgem.gmsh.gmsh import GeometryOCC

from dfn_data import NetworkData, clip_polygon_to_box, polygon_area, polygon_vertices
from mesh_verify import face_keys as _face_keys, tetra_face_index
from mesh_output import write_fracture_radius
from healing import heal_mesh_file, validate_gamma
from geometry_healing import geometry_metrics, heal_cad


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeshConfig:
    fracture_size: float = .05
    alpha: float = 0.0
    bulk_size: float = .25
    tolerance: float = 1e-8
    geometry_tol: float = 0.0
    verbose: bool = False
    threads: int = 1
    occ_parallel: bool = False
    refinement_distance: float = .1
    gamma: float | None = None

    def __post_init__(self):
        # PyYAML can leave bare exponents such as 1e-3 as strings. Canonicalize
        # floating settings before validation and before any expensive CAD work.
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type in (float, float | None) and value is not None:
                object.__setattr__(self, field.name, float(value))
        if self.gamma is not None:
            validate_gamma(self.gamma)
        if min(self.fracture_size, self.bulk_size, self.tolerance) <= 0 or self.alpha < 0 or self.threads < 1 or self.refinement_distance < 0:
            raise ValueError('Sizes/tolerance/threads must be positive; alpha/refinement_distance must be nonnegative')


@dataclass
class GeometryResult:
    factory: GeometryOCC
    objects: list
    fractures: list[dict]
    timings: dict
    boundary_membership: dict
    geometry: dict


def _surface_regions(objects):
    return {obj.regions[0].name: obj for obj in objects.split_by_region() if obj.size}


def _plane_side(dimtag, bounds, tolerance):
    # Exact boundary vertices avoid the fixed absolute padding of OCC AABBs.
    vertices = gmsh.model.getBoundary([dimtag], combined=False, oriented=False, recursive=True)
    points = np.array([gmsh.model.getValue(0, tag, []) for dim, tag in vertices])
    for axis in range(3):
        for side in range(2):
            if len(points) and np.all(np.abs(points[:, axis]-bounds[axis, side]) < tolerance):
                return f'side_{"xyz"[axis]}{side}'
    return None


def build_geometry(network: NetworkData, mesh_config: MeshConfig, factory: GeometryOCC, records=None, geometry_report=None):
    """Build source polygons, fragment with the bulk, and optionally heal CAD.

    The caller supplies and owns the factory, and must close it in a finally block.
    generate_mesh manages this lifetime for the normal conversion workflow.
    """
    started = time.perf_counter()
    bounds = network.bounds
    bulk = factory.box(bounds[:, 1]-bounds[:, 0], center=bounds.mean(axis=1)).set_region('bulk')
    shapes = []
    manifest = records if records is not None else []
    clip_seconds = 0.0
    polygons = {}
    metrics = {} if geometry_report is None else geometry_report
    for family in network.families:
        vertices = polygon_vertices(family)
        areas = polygon_area(vertices)
        for index, (row, family_idx, points, area) in enumerate(
                zip(family.source_rows, family.arrays['fracture_set_idx'], vertices, areas)):
            region = f'fr_s{int(family_idx):02d}_d{network.domain_idx:03d}_r{network.realization:03d}_{int(row):07d}'
            manifest.append(dict(source_id=region, region=region, source=str(family.source), source_row=int(row),
                                 family_id=int(family_idx), family_name=family.name,
                                 source_area=float(area), fracture_radius=float(family.arrays['fracture_req'][index]),
                                 clipped_area=0., status='outside'))
            clip_start = time.perf_counter()
            points = clip_polygon_to_box(points, bounds)
            clip_seconds += time.perf_counter()-clip_start
            if len(points) >= 3 and polygon_area(points) > 0:
                polygons[region] = points
                manifest[-1]['expected_clipped_area'] = float(polygon_area(points))
                shapes.append(factory.make_polygon(points).set_region(region))
    log.info('Created %d clipped CAD fracture faces from %d sources', len(shapes), len(manifest))
    created = time.perf_counter()
    fractures = factory.group(*shapes)
    clipped = time.perf_counter()
    if fractures.size:
        bulk, fractures = factory.fragment(bulk, fractures)
        bulk = bulk.split_by_dimension()[3]
        fractures = fractures.split_by_dimension()[2]
    log.info('Joint fragmentation complete')
    fragmented = time.perf_counter()
    ownership = {}
    for dimtag, region in fractures.dimtagreg():
        if dimtag in ownership and ownership[dimtag] != region.name:
            raise ValueError(f'Source fractures overlap on CAD face {dimtag}: {ownership[dimtag]}, {region.name}')
        ownership[dimtag] = region.name
    regions = _surface_regions(fractures)
    for record in manifest:
        if record['region'] in regions:
            record.update(status='retained', clipped_area=float(regions[record['region']].mass()),
                          surface_tags=regions[record['region']].tags)
        if 'expected_clipped_area' in record and (record['status'] != 'retained' or not np.isclose(
                record['clipped_area'], record['expected_clipped_area'], rtol=1e-7, atol=mesh_config.tolerance**2)):
            raise ValueError(f'Fragmentation lost or changed source polygon: {record}')
    factory.synchronize()
    metrics.update(geometry_tol=mesh_config.geometry_tol,
                   healing_applied=mesh_config.geometry_tol > 0,
                   before_healing=geometry_metrics())
    heal_started = time.perf_counter()
    if mesh_config.geometry_tol > 0:
        log.info('OCC healing with geometry_tol=%g', mesh_config.geometry_tol)
        ownership = heal_cad(polygons, mesh_config.geometry_tol)
        bulk = factory.group(*(factory.object(*dt) for dt in gmsh.model.getEntities(3))).set_region('bulk')
        fractures = factory.group(*(factory.object(*dt).set_region(source)
                                    for dt, source in ownership.items()))
        regions = _surface_regions(fractures)
        for record in manifest:
            if record['status'] != 'retained':
                continue
            source = record['source_id']
            before = record['clipped_area']
            area = float(regions[source].mass()) if source in regions else 0.0
            polygon = polygons[source]
            perimeter = np.linalg.norm(polygon-np.roll(polygon, 1, axis=0), axis=1).sum()
            area_budget = mesh_config.geometry_tol*perimeter + np.pi*mesh_config.geometry_tol**2
            record.update(pre_heal_clipped_area=before, clipped_area=area,
                          geometry_area_change=area-before,
                          surface_tags=regions[source].tags if source in regions else [],
                          status='retained' if source in regions else 'healed_removed')
            if abs(area-before) > area_budget + 1e-7*before:
                raise ValueError(f'OCC healing changed source area beyond its tolerance budget: {source}')
    healed = time.perf_counter()
    metrics['after_healing'] = geometry_metrics() if mesh_config.geometry_tol > 0 else metrics['before_healing']
    log.info('CAD feature ratios: %s', metrics)
    # Source regions serve only to preserve provenance through Boolean splitting.
    # Physical regions exposed to consumers are the input fracture-family names.
    family_regions = {r['source_id']: r['family_name'] for r in manifest}
    fractures.regions = [factory.get_region_name(family_regions[r.name]).complete(2)
                         for r in fractures.regions]
    ownership = {dimtag: family_regions[source] for dimtag, source in ownership.items()}
    for record in manifest:
        record['region'] = record['family_name']
    bulk.mesh_step(mesh_config.bulk_size)
    # Descending sizes ensure shared points receive the finest incident target.
    for record in sorted(manifest, key=lambda r: r['fracture_radius'], reverse=True):
        record['mesh_size'] = min(mesh_config.bulk_size, max(
            mesh_config.fracture_size, mesh_config.alpha * record['fracture_radius']))
        for tag in record.get('surface_tags', []):
            factory.object(2, tag).mesh_step(record['mesh_size'])
    factory.synchronize()
    plane_tol = 128*np.finfo(float).eps*max(np.abs(bounds).max(), np.ptp(bounds, axis=1).max())
    boundary = []
    membership = {f'side_{axis}{side}': set() for axis in 'xyz' for side in range(2)}
    exterior_faces = {}
    for dimtag in bulk.get_boundary(combined=True).dim_tags:
        side = _plane_side(dimtag, bounds, plane_tol)
        if side:
            exterior_faces[dimtag] = side
            region = ownership.get(dimtag, '.'+side)
            membership[side].add(region)
            if dimtag not in ownership:
                boundary.append(factory.object(*dimtag).set_region(region).mesh_step(mesh_config.bulk_size))
    for record in manifest:
        record['boundary_sides'] = sorted({exterior_faces[(2, tag)] for tag in record.get('surface_tags', [])
                                           if (2, tag) in exterior_faces})
    # Only fracture traces lying on the domain's exterior get boundary labels.
    if fractures.size:
        edges = set(fractures.get_boundary().dim_tags)
        for dimtag in edges:
            side = _plane_side(dimtag, bounds, plane_tol)
            if side:
                boundary.append(factory.object(*dimtag).set_region('.fr_outer_'+side))
    return GeometryResult(factory, [bulk, fractures, *boundary], manifest,
                          dict(create=created-started-clip_seconds, clip=clip_seconds,
                               fragment=fragmented-clipped, geometry_healing=healed-heal_started,
                               boundaries=time.perf_counter()-healed),
                          {side: sorted(names) for side, names in membership.items()}, metrics)


def _elements(dim, tag=-1):
    types, tags, nodes = gmsh.model.mesh.getElements(dim, tag)
    expected_type, width = {2: (2, 3), 3: (4, 4)}[dim]
    if any(int(t) != expected_type for t in types):
        raise ValueError(f'Expected first-order simplex elements in dimension {dim}: {types}')
    if not len(types):
        return np.empty(0, dtype=np.uint64), np.empty((0, width), dtype=np.uint64)
    return tags[0], nodes[0].reshape(-1, width)


def verify_current_mesh(bounds, records, directory=Path('.')):
    """Verify serialized geometry with bounded working memory and disk-backed faces."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(node_tags)
    node_tags, coords = node_tags[order], np.asarray(coords).reshape(-1, 3)[order]
    tags, tetra = _elements(3)
    if not len(tetra):
        raise ValueError('Bulk contains no tetrahedra')
    volume, min_volume = 0.0, np.inf
    for start in range(0, len(tetra), 200_000):
        points = coords[np.searchsorted(node_tags, tetra[start:start+200_000])]
        volumes = np.linalg.det(points[:, 1:]-points[:, :1])/6
        if np.min(volumes) <= 0:
            raise ValueError('Bulk contains nonpositive tetrahedra')
        volume += float(volumes.sum())
        min_volume = min(min_volume, float(volumes.min()))
    expected_volume = float(np.prod(bounds[:, 1]-bounds[:, 0]))
    if not np.isclose(volume, expected_volume, rtol=1e-7, atol=0):
        raise ValueError(f'Bulk volume {volume} differs from {expected_volume}')
    with tetra_face_index(tetra, directory=directory) as index:
        result = _verify_regions(bounds, records, node_tags, coords, index)
    qualities = gmsh.model.mesh.getElementQualities(tags, 'minSICN')
    return dict(result, nodes=len(node_tags), tetrahedra=len(tetra), bulk_volume=volume,
                min_tetra_volume=min_volume, min_quality=float(np.min(qualities)))


def _verify_regions(bounds, records, node_tags, coords, index):
    expected = {r.get('source_id', r['region']): r for r in records if r['status'] == 'retained'}
    families = {r['region'] for r in expected.values()}
    owners = {tag: source for source, r in expected.items() for tag in r['surface_tags']}
    source_areas = dict.fromkeys(expected, 0.0)
    seen_sources, found = set(), set()
    boundary_areas = {f'side_{axis}{side}': 0.0 for axis in 'xyz' for side in range(2)}
    boundary_keys = []
    nonconforming, triangles = 0, 0
    length_tol = 1e-9*np.min(bounds[:, 1]-bounds[:, 0])
    for dim, physical in gmsh.model.getPhysicalGroups(2):
        name = gmsh.model.getPhysicalName(dim, physical)
        is_fracture = name in families
        if not is_fracture and not name.startswith('.side_'):
            continue
        if is_fracture:
            found.add(name)
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, physical):
            entity = int(entity)
            source = owners.get(entity)
            if is_fracture and (source is None or expected[source]['region'] != name):
                raise ValueError(f'Incorrect fracture-family ownership on surface {entity}: {name}')
            if not is_fracture and source is not None:
                raise ValueError(f'Duplicate exterior/fracture physical ownership on surface {entity}')
            tri = _elements(2, entity)[1]
            query = _face_keys(tri)
            incidence = index.incidence(tri)
            tri_points = coords[np.searchsorted(node_tags, tri)]
            areas = np.linalg.norm(np.cross(tri_points[:, 1]-tri_points[:, 0], tri_points[:, 2]-tri_points[:, 0]), axis=1)/2
            exterior = bool(expected[source].get('boundary_sides')) if is_fracture else True
            nonconforming += int(np.count_nonzero(incidence != (1 if exterior else 2)))
            boundary_keys.append(query[incidence == 1])
            if is_fracture:
                if not len(tri):
                    raise ValueError(f'No triangles for source surface {entity}')
                triangles += len(tri)
                source_areas[source] += float(areas.sum())
                seen_sources.add(source)
            for axis in range(3):
                for side in range(2):
                    on_side = (incidence == 1) & np.all(np.isclose(
                        tri_points[:, :, axis], bounds[axis, side], rtol=0, atol=length_tol), axis=1)
                    boundary_areas[f'side_{"xyz"[axis]}{side}'] += float(areas[on_side].sum())
    for source, record in expected.items():
        if not np.isclose(source_areas[source], record['clipped_area'], rtol=1e-6, atol=0):
            raise ValueError(f'Mesh area differs from clipped CAD area: {source}')
    for axis in range(3):
        side_area = np.prod(np.delete(bounds[:, 1]-bounds[:, 0], axis))
        for side in range(2):
            if not np.isclose(boundary_areas[f'side_{"xyz"[axis]}{side}'], side_area, rtol=1e-7, atol=0):
                raise ValueError(f'Incorrect exterior boundary area on axis {axis}, side {side}')
    covered = np.unique(np.concatenate(boundary_keys)) if boundary_keys else np.empty(0, dtype=index.keys.dtype)
    if not np.array_equal(covered, index.boundary):
        raise ValueError('Exterior tetrahedron faces are missing from named boundary regions')
    if found != families or seen_sources != set(expected) or nonconforming:
        raise ValueError(f'Mesh verification failed: missing={set(expected)-seen_sources}, nonconforming={nonconforming}')
    return dict(fracture_triangles=triangles, fracture_regions=len(found), fracture_sources=len(seen_sources),
                boundary_regions=len(boundary_areas), boundary_areas=boundary_areas,
                nonconforming_triangles=nonconforming)


def _environment():
    dist = importlib.metadata.distribution('bgem')
    return dict(gmsh=gmsh.__version__, bgem=dist.version,
                bgem_source=json.loads(dist.read_text('direct_url.json') or '{}'),
                bgem_module=str(Path(dist.locate_file('bgem')).resolve()))


def generate_mesh(network: NetworkData, mesh_config: MeshConfig, output: Path):
    """Generate/verify the raw mesh, then optionally run HealMesh in its own session."""
    report = _generate_raw_mesh(network, mesh_config, output)
    if mesh_config.gamma is None:
        return report
    try:
        report['healing'] = heal_mesh_file(output, mesh_config.gamma)
        return report
    except Exception as error:
        report.update(status='failed', stage='healing', error=str(error))
        raise
    finally:
        Path(output).with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n')


def _generate_raw_mesh(network: NetworkData, mesh_config: MeshConfig, output: Path):
    """Write a mixed-dimensional MSH 2.2 and a verified, source-indexed JSON report."""
    if gmsh.isInitialized():
        raise RuntimeError('Close the active Gmsh session or use build_geometry with its factory')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    factory = GeometryOCC(output.stem, verbose=mesh_config.verbose)
    report = dict(bounds=network.bounds.tolist(),
                  coordinate_scale=network.coordinate_scale, config=vars(mesh_config),
                  source_fractures=len(network), physical_grouping='family', status='building')
    gmsh.logger.start()
    stage = 'geometry'
    started = time.perf_counter()
    try:
        report['environment'] = _environment()
        output.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n')
        output.with_suffix('.verification.json').write_text(json.dumps(
            dict(status='building', mesh=str(output)), indent=2)+'\n')
        for key, value in {'Geometry.Tolerance': mesh_config.tolerance,
                           'Geometry.ToleranceBoolean': mesh_config.tolerance,
                           'Geometry.OCCParallel': int(mesh_config.occ_parallel),
                           'General.NumThreads': mesh_config.threads,
                           'Mesh.MshFileVersion': 2.2, 'Mesh.Binary': 0,
                           'Mesh.ElementOrder': 1, 'Mesh.MeshSizeMax': mesh_config.bulk_size}.items():
            gmsh.option.setNumber(key, value)
        report['fractures'] = []
        report['geometry'] = {}
        geometry = build_geometry(network, mesh_config, factory, records=report['fractures'],
                                  geometry_report=report['geometry'])
        report.update(fractures=geometry.fractures, timings=geometry.timings,
                      boundary_membership=geometry.boundary_membership)
        report['entities'] = {str(d): len(gmsh.model.getEntities(d)) for d in range(4)}
        output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        output.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n')
        # Keep the exact CAD checkpoint even if subsequent meshing is interrupted.
        factory.write_brep(str(output.with_suffix('')))
        stage = 'mesh'
        if mesh_config.refinement_distance and geometry.objects[1].size:
            _set_refinement_field(geometry.objects[1], mesh_config, geometry.fractures)
        before_mesh = time.perf_counter()
        log.info('Meshing %s', report['entities'])
        # All construction inputs were consumed by fragmentation. Preserve the
        # complete final topology, including embedded curves/surfaces. BGEM's
        # keep_only otherwise attempts expensive removal of its descendants.
        factory.make_mesh(geometry.objects, dim=3, eliminate=False)
        if gmsh.model.mesh.getLastEntityError():
            raise ValueError(f'Gmsh reported bad entities: {gmsh.model.mesh.getLastEntityError()}')
        report['timings']['mesh'] = time.perf_counter()-before_mesh
        physical_ids = {gmsh.model.getPhysicalName(d, t): int(t) for d, t in gmsh.model.getPhysicalGroups(2)}
        for record in report['fractures']:
            if record['status'] == 'retained':
                record['physical_id'] = physical_ids[record['region']]
        factory.write_mesh(str(output))
        _assert_msh22(output)
        stage = 'verification'
        log.info('Reopening mesh to verify volume, areas and triangle/tetrahedron conformity')
        gmsh.clear()
        gmsh.open(str(output))
        before_verify = time.perf_counter()
        report['verification'] = verify_current_mesh(network.bounds, report['fractures'], directory=output.parent)
        report['timings']['verification'] = time.perf_counter()-before_verify
        radii = {tag: record['fracture_radius'] for record in report['fractures']
                 if record['status'] == 'retained' for tag in record['surface_tags']}
        groups = [(ids, radii.get(tag, 0.0) if dim == 2 else 0.0)
                  for dim, tag in gmsh.model.getEntities()
                  for ids in gmsh.model.mesh.getElements(dim, tag)[1]]
        with output.open('a') as stream:
            write_fracture_radius(stream, sum(len(ids) for ids, _ in groups), groups)
        report['status'] = 'verified'
        return report
    except Exception as error:
        report.update(status='failed', stage=stage, error=str(error))
        if stage != 'verification':
            factory.synchronize()
            gmsh.write(str(output.with_suffix('.failed.brep')))
        raise
    finally:
        try:
            report['elapsed_seconds'] = time.perf_counter()-started
            report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            try:
                output.with_suffix('.gmsh.log').write_text('\n'.join(gmsh.logger.get())+'\n')
            finally:
                gmsh.logger.stop()
            output.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n')
            summary = {key: report[key] for key in ('status', 'verification', 'error', 'stage') if key in report}
            summary['mesh'] = str(output)
            output.with_suffix('.verification.json').write_text(json.dumps(summary, indent=2)+'\n')
        finally:
            factory.close()


def _assert_msh22(output):
    with Path(output).open() as stream:
        if (stream.readline().strip(), stream.readline().strip()) != ('$MeshFormat', '2.2 0 8'):
            raise ValueError('Output is not ASCII MSH 2.2')


def verify_mesh_file(output: Path):
    """Recheck an exported mesh against its saved CAD manifest, without remeshing."""
    output = Path(output).resolve()
    report_path = output.with_suffix('.verification.json')
    started = time.perf_counter()
    verification = dict(status='checking', mesh=str(output))
    report_path.write_text(json.dumps(verification, indent=2)+'\n')
    initialized = False
    try:
        _assert_msh22(output)
        manifest = json.loads(output.with_suffix('.manifest.json').read_text())
        if gmsh.isInitialized():
            raise RuntimeError('Close the active Gmsh session before verifying a saved mesh')
        gmsh.initialize()
        initialized = True
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.open(str(output))
        result = verify_current_mesh(np.array(manifest['bounds']), manifest['fractures'], directory=output.parent)
        verification.update(status='verified', verification=result, environment=_environment())
        return verification
    except Exception as error:
        verification.update(status='failed', error=str(error))
        raise
    finally:
        try:
            verification.update(elapsed_seconds=time.perf_counter()-started,
                                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            report_path.write_text(json.dumps(verification, indent=2)+'\n')
        finally:
            if initialized:
                gmsh.finalize()


def _size_groups(records):
    """Bin targets into at most five geometric bands, using each band's minimum."""
    retained = [r for r in records if r['status'] == 'retained']
    if not retained:
        return []
    sizes = np.array([r['mesh_size'] for r in retained])
    boundaries = np.geomspace(sizes.min(), sizes.max(), 6)[1:-1]
    bands = np.searchsorted(boundaries, sizes, side='right')
    return [(sorted({tag for r, band in zip(retained, bands) if band == group
                     for tag in r['surface_tags']}), float(sizes[bands == group].min()))
            for group in np.unique(bands)]


def _set_refinement_field(fractures, config, records=None):
    """Combine per-source fields, taking the finest size at intersections."""
    groups = ([(fractures.tags, config.fracture_size)] if records is None
              else _size_groups(records))
    fields = [_fracture_refinement_field(tags, size, config) for tags, size in groups]
    minimum = gmsh.model.mesh.field.add('Min')
    gmsh.model.mesh.field.setNumbers(minimum, 'FieldsList', fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
    for option in ('MeshSizeExtendFromBoundary', 'MeshSizeFromPoints', 'MeshSizeFromCurvature'):
        gmsh.option.setNumber('Mesh.'+option, 0)


def _fracture_refinement_field(tags, size, config):
    """Grade bulk sizes without boundary-point interpolation overriding the field."""
    distance = gmsh.model.mesh.field.add('Distance')
    gmsh.model.mesh.field.setNumbers(distance, 'SurfacesList', tags)
    gmsh.model.mesh.field.setNumber(distance, 'Sampling', 10)
    threshold = gmsh.model.mesh.field.add('Threshold')
    for key, value in dict(InField=distance, SizeMin=size,
                           SizeMax=config.bulk_size, DistMin=0,
                           DistMax=config.refinement_distance).items():
        gmsh.model.mesh.field.setNumber(threshold, key, value)
    fine = gmsh.model.mesh.field.add('MathEval')
    gmsh.model.mesh.field.setString(fine, 'F', str(size))
    restrict = gmsh.model.mesh.field.add('Restrict')
    gmsh.model.mesh.field.setNumber(restrict, 'InField', fine)
    entities = {(2, tag) for tag in tags}
    for dimension in (2, 1):
        roots = [dt for dt in entities if dt[0] == dimension]
        for dimtag in roots:
            entities.update(gmsh.model.mesh.getEmbedded(*dimtag))
        roots = [dt for dt in entities if dt[0] == dimension]
        entities.update(gmsh.model.getBoundary(roots, combined=False, oriented=False))
    for dimension, key in enumerate(('PointsList', 'CurvesList', 'SurfacesList')):
        gmsh.model.mesh.field.setNumbers(restrict, key, [tag for dim, tag in entities if dim == dimension])
    minimum = gmsh.model.mesh.field.add('Min')
    gmsh.model.mesh.field.setNumbers(minimum, 'FieldsList', [threshold, restrict])
    return minimum
