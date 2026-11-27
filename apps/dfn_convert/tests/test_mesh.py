from pathlib import Path

import numpy as np
import pytest

from dfn_data import FamilyData, NetworkData
from mesh import MeshConfig, generate_mesh


def synthetic_network(centers, normals, radii, bounds=None):
    centers = np.array(centers, dtype=float)
    normals = np.array(normals, dtype=float)
    radii = np.array(radii, dtype=float)
    n = len(centers)
    u = np.cross(normals, [0, 1, 0])
    u /= np.linalg.norm(u, axis=1)[:, None]
    arrays = {'fracture_set_idx': np.zeros(n, dtype=np.int32),
              'fracture_nverts': np.full(n, 6, dtype=np.int32),
              'fracture_aaxis': radii, 'fracture_ratio': np.ones(n),
              'fracture_req': radii*np.sqrt(3*np.sqrt(3)/(2*np.pi))}
    for kind, vector in [('center', centers), ('normal', normals), ('rvec', u*radii[:, None])]:
        arrays.update({f'fracture_{kind}_{axis}': vector[:, i] for i, axis in enumerate('xyz')})
    family = FamilyData(Path('synthetic.npz'), 'test', arrays, np.arange(n))
    return NetworkData(np.array(bounds if bounds is not None else [[-.5, .5]]*3), (family,))


@pytest.mark.parametrize('occ_parallel', [False, True])
def test_crossing_clipped_and_isolated_fractures_form_conforming_mesh(tmp_path, occ_parallel):
    # Losing source IDs in booleans, failing to embed isolated faces, or independently
    # meshing fracture triangles makes verification fail.
    network = synthetic_network([[1, 2, 3], [1, 2, 3], [1.2, 2.2, 3.3], [5, 5, 5]],
                                [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]],
                                [2, 2, .1, .1], [[.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
    output = tmp_path / 'combined.msh2'
    report = generate_mesh(network, MeshConfig(fracture_size=.1, bulk_size=.25, occ_parallel=occ_parallel, threads=2), output)
    assert report['verification']['bulk_volume'] == pytest.approx(1)
    assert report['verification']['nonconforming_triangles'] == 0
    assert report['verification']['min_tetra_volume'] > 0
    assert report['verification']['fracture_regions'] == 1
    assert report['verification']['fracture_sources'] == 3
    assert report['verification']['boundary_regions'] == 6
    assert report['fractures'][0]['clipped_area'] == pytest.approx(1)
    assert report['fractures'][1]['clipped_area'] == pytest.approx(1)
    assert report['fractures'][2]['clipped_area'] == pytest.approx(.015*np.sqrt(3))
    assert report['fractures'][3]['status'] == 'outside'
    assert output.read_text().splitlines()[:2] == ['$MeshFormat', '2.2 0 8']
    assert output.with_suffix('.report.json').is_file()


def test_bulk_only_mesh(tmp_path):
    network = NetworkData(np.array([[-1, 1]]*3), ())
    report = generate_mesh(network, MeshConfig(fracture_size=.2, bulk_size=.5), tmp_path/'bulk.msh2')
    assert report['verification']['bulk_volume'] == pytest.approx(8)
    assert report['verification']['fracture_regions'] == 0


def test_invalid_mesh_size_fails_early():
    with pytest.raises(ValueError):
        MeshConfig(fracture_size=0)


def test_overlapping_source_fractures_fail_instead_of_duplicating_mesh_elements(tmp_path):
    network = synthetic_network([[0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 1]], [.3, .3])
    output = tmp_path/'overlap.msh2'
    with pytest.raises(ValueError, match='overlap'):
        generate_mesh(network, MeshConfig(fracture_size=.1, bulk_size=.25), output)
    assert not output.exists()
    assert output.with_suffix('.report.json').exists()


@pytest.mark.parametrize('radius', [.2, 2.])
def test_fracture_on_exterior_has_unique_physical_ownership(tmp_path, radius):
    network = synthetic_network([[0, 0, .5]], [[0, 0, 1]], [radius])
    report = generate_mesh(network, MeshConfig(fracture_size=.1, bulk_size=.25), tmp_path/'exterior.msh2')
    check = report['verification']
    assert check['nonconforming_triangles'] == 0
    assert check['boundary_regions'] == 6
    assert report['fractures'][0]['region'] in report['boundary_membership']['side_z1']
    assert check['boundary_areas']['side_z1'] == pytest.approx(1)


def test_scaled_mesh_verification_rejects_wrong_bulk_volume(tmp_path):
    import gmsh
    from mesh import verify_current_mesh
    network = NetworkData(np.array([[-.001, .001]]*3), ())
    output = tmp_path/'small.msh2'
    generate_mesh(network, MeshConfig(fracture_size=.0002, bulk_size=.0005), output)
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        gmsh.open(str(output))
        with pytest.raises(ValueError, match='Bulk volume'):
            verify_current_mesh(np.array([[-.0005, .0005]]*3), [])
    finally:
        gmsh.finalize()


def test_report_write_failure_still_closes_gmsh(tmp_path, monkeypatch):
    import gmsh
    original = Path.write_text

    def fail_report(path, *args, **kwargs):
        if path.name.endswith('.report.json'):
            raise OSError('report write failed')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', fail_report)
    network = NetworkData(np.array([[-.5, .5]]*3), ())
    with pytest.raises(OSError, match='report write failed') as caught:
        generate_mesh(network, MeshConfig(fracture_size=.2, bulk_size=.5), tmp_path/'cleanup.msh2')
    assert caught.value.__traceback__ is not None
    assert not gmsh.isInitialized()


def test_geometry_contains_only_final_entities(tmp_path):
    from pathlib import Path
    from bgem.gmsh.gmsh import GeometryOCC
    from dfn_data import load_network, crop_network
    from mesh import build_geometry
    import gmsh
    network = crop_network(load_network(Path(__file__).resolve().parents[1]/'DFN_gen-out_v01_d004'),
                           np.array([[-.25, .25]]*3))
    factory = GeometryOCC('closure')
    try:
        geometry = build_geometry(network, MeshConfig(), factory)
        closure = {dt for obj in geometry.objects for dt in obj.dim_tags}
        for dimension in (3, 2, 1):
            roots = [dt for dt in closure if dt[0] == dimension]
            for dimtag in roots:
                closure.update(gmsh.model.mesh.getEmbedded(*dimtag))
            roots = [dt for dt in closure if dt[0] == dimension]
            closure.update(gmsh.model.getBoundary(roots, combined=False, oriented=False))
        assert closure == set(gmsh.model.getEntities())
    finally:
        factory.close()


def test_background_refinement_allows_coarser_bulk_between_fractures(tmp_path):
    import gmsh
    network = synthetic_network([[0, 0, -.3], [0, 0, 0], [0, 0, .3]], [[0, 0, 1]]*3, [2]*3)
    output = tmp_path/'graded.msh2'
    generate_mesh(network, MeshConfig(fracture_size=.04, bulk_size=.4, refinement_distance=.1), output)
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
    try:
        gmsh.open(str(output))
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tetra = gmsh.model.mesh.getElementsByType(4)[1].reshape(-1, 4)
        points = np.asarray(xyz).reshape(-1, 3)[order][np.searchsorted(tags[order], tetra)]
        center = points.mean(axis=1)
        far = (np.abs(center[:, 0]) < .3) & (np.abs(center[:, 1]) < .3) & (center[:, 2] > .06) & (center[:, 2] < .24)
        edges = np.stack([np.linalg.norm(points[far, i]-points[far, j], axis=1)
                          for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
        assert edges.mean() > .08
        fracture_tags = [int(entity) for dim, physical in gmsh.model.getPhysicalGroups(2)
                         if gmsh.model.getPhysicalName(dim, physical) == 'test'
                         for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, physical)]
        triangles = np.concatenate([gmsh.model.mesh.getElementsByType(2, tag)[1].reshape(-1, 3)
                                    for tag in fracture_tags])
        surface = np.asarray(xyz).reshape(-1, 3)[order][np.searchsorted(tags[order], triangles)]
        fracture_edges = np.linalg.norm(surface-np.roll(surface, 1, axis=1), axis=2)
        assert fracture_edges.mean() < .06
    finally:
        gmsh.finalize()


def test_failed_recheck_replaces_previous_success_report(tmp_path):
    import json
    import gmsh
    from mesh import verify_mesh_file
    output = tmp_path/'recheck.msh2'
    generate_mesh(NetworkData(np.array([[-.5, .5]]*3), ()), MeshConfig(), output)
    verify_mesh_file(output)
    manifest = json.loads(output.with_suffix('.manifest.json').read_text())
    manifest['bounds'] = [[-.25, .25]]*3
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Bulk volume'):
        verify_mesh_file(output)
    report = json.loads(output.with_suffix('.verification.json').read_text())
    assert report['status'] == 'failed'
    assert 'Bulk volume' in report['error']
    assert not gmsh.isInitialized()


def test_mesh_generation_preserves_existing_gmsh_session(tmp_path):
    import gmsh
    gmsh.initialize()
    gmsh.model.add('existing')
    try:
        with pytest.raises(RuntimeError, match='active Gmsh'):
            generate_mesh(NetworkData(np.array([[-.5, .5]]*3), ()), MeshConfig(), tmp_path/'nested.msh2')
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == 'existing'
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def test_fracture_physical_groups_use_input_family_names(tmp_path):
    from dataclasses import replace
    import gmsh
    original = synthetic_network([[0, 0, -.25], [0, 0, 0], [0, 0, .25]], [[0, 0, 1]]*3, [.2]*3)
    ew = replace(original.families[0].select([0, 1]), name='EW')
    ns = replace(original.families[0].select([2]), name='NS_FOL')
    ns = replace(ns, arrays={**ns.arrays, 'fracture_set_idx': np.array([1])})
    output = tmp_path/'families.msh2'
    report = generate_mesh(replace(original, families=(ew, ns)), MeshConfig(), output)
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        gmsh.open(str(output))
        names = {gmsh.model.getPhysicalName(d, p) for d, p in gmsh.model.getPhysicalGroups(2)}
        assert names == {'EW', 'NS_FOL', '.side_x0', '.side_x1', '.side_y0', '.side_y1', '.side_z0', '.side_z1'}
        assert [r['region'] for r in report['fractures']] == ['EW', 'EW', 'NS_FOL']
        assert len({r['source_id'] for r in report['fractures']}) == 3
        assert report['verification']['fracture_regions'] == 2
        assert report['verification']['fracture_sources'] == 3
    finally:
        gmsh.finalize()


def test_family_can_include_both_exterior_and_internal_fractures(tmp_path):
    network = synthetic_network([[0, 0, .5], [0, 0, 0]], [[0, 0, 1]]*2, [.2, .2])
    report = generate_mesh(network, MeshConfig(), tmp_path/'mixed.msh2')
    assert report['verification']['fracture_regions'] == 1
    assert report['verification']['fracture_sources'] == 2
    assert report['verification']['nonconforming_triangles'] == 0
    assert report['fractures'][0]['boundary_sides'] == ['side_z1']
    assert report['fractures'][1]['boundary_sides'] == []


def test_gamma_runs_healmesh_after_generation(tmp_path):
    network = synthetic_network([[0, 0, 0]], [[0, 0, 1]], [.2])
    report = generate_mesh(network, MeshConfig(gamma=.01), tmp_path/'gamma.msh2')
    assert report['healing']['gamma'] == .01
    assert report['healing']['status'] == 'completed'
    assert 'test' in report['healing']['physical_names']
    healed = Path(report['healing']['output'])
    assert healed.is_file()
    assert '"test"' in healed.read_text()


@pytest.mark.parametrize('geometry_tol', [0., -1., 1e-4])
def test_geometry_healing_reports_ratios_and_preserves_fracture_families(tmp_path, geometry_tol):
    from dataclasses import replace
    import json
    original = synthetic_network([[0, 0, 0], [0, 0, .25], [0, 0, .5]],
                                 [[0, 0, 1]]*3, [2., .1, .2])
    main = replace(original.families[0].select([0]), name='EW')
    other = replace(original.families[0].select([1, 2]), name='NS_FOL')
    other = replace(other, arrays={**other.arrays, 'fracture_set_idx': np.array([1, 1])})
    output = tmp_path/'geometry.msh2'
    report = generate_mesh(replace(original, families=(main, other)),
                           MeshConfig(geometry_tol=geometry_tol), output)
    assert report['verification']['fracture_sources'] == 3
    assert report['verification']['fracture_regions'] == 2
    assert report['verification']['bulk_volume'] == pytest.approx(1)
    assert report['verification']['nonconforming_triangles'] == 0
    assert [r['region'] for r in report['fractures']] == ['EW', 'NS_FOL', 'NS_FOL']
    metrics = report['geometry']
    assert metrics['healing_applied'] == (geometry_tol > 0)
    for stage in ('before_healing', 'after_healing'):
        for ratio in ('area_perimeter', 'volume_surface_area'):
            assert 0 < metrics[stage][ratio]['min'] <= metrics[stage][ratio]['max']
    assert json.loads(output.with_suffix('.manifest.json').read_text())['geometry'] == metrics
    assert json.loads(output.with_suffix('.report.json').read_text())['geometry'] == metrics
    if geometry_tol > 0:
        assert all('pre_heal_clipped_area' in r for r in report['fractures'])


def test_geometry_metrics_are_persisted_before_native_meshing(tmp_path, monkeypatch):
    import json
    from bgem.gmsh.gmsh import GeometryOCC
    output = tmp_path/'checkpoint.msh2'

    def stop_before_meshing(self, *args, **kwargs):
        # A real SIGSEGV skips finally blocks, so this must already be on disk.
        saved = json.loads(output.with_suffix('.report.json').read_text())
        assert saved['geometry']['after_healing']['area_perimeter']['min'] == pytest.approx(.25)
        assert saved['geometry']['after_healing']['volume_surface_area']['min'] == pytest.approx(1/6)
        raise RuntimeError('stop before native meshing')

    monkeypatch.setattr(GeometryOCC, 'make_mesh', stop_before_meshing)
    with pytest.raises(RuntimeError, match='stop before native meshing'):
        generate_mesh(NetworkData(np.array([[-.5, .5]]*3), ()), MeshConfig(), output)


@pytest.mark.parametrize('name', ['fracture_size', 'bulk_size', 'tolerance',
                                  'geometry_tol', 'refinement_distance', 'gamma'])
def test_scientific_notation_strings_are_normalized_before_validation(name):
    # PyYAML safe_load('value: 1e-3') returns a string, unlike CLI type=float.
    config = MeshConfig(**{name: '1e-3'})
    assert getattr(config, name) == pytest.approx(.001)
    assert isinstance(getattr(config, name), float)


def test_fracture_radius_survives_healing(tmp_path):
    import gmsh

    network = synthetic_network([[0, 0, -.25], [0, 0, .25]],
                                [[0, 0, 1]]*2, [.1, .2])
    output = tmp_path/'radii.msh2'
    report = generate_mesh(network, MeshConfig(gamma=1e-8), output)
    radii = {tag: record['fracture_radius'] for record in report['fractures']
             for tag in record['surface_tags']}
    assert sorted(radii.values()) == pytest.approx(
        network.families[0].arrays['fracture_req'])
    for path in (output, report['healing']['output']):
        gmsh.initialize()
        try:
            gmsh.open(str(path))
            kind, ids, values, _, components = gmsh.view.getModelData(
                gmsh.view.getTags()[0], 0)
            assert (kind, components) == ('ElementData', 1)
            actual = dict(zip(ids, np.ravel(values)))
            expected = {eid: radii.get(tag, 0) if dim == 2 else 0
                        for dim, tag in gmsh.model.getEntities()
                        for group in gmsh.model.mesh.getElements(dim, tag)[1]
                        for eid in group}
            assert actual == expected
        finally:
            gmsh.finalize()


def test_size_groups_bound_field_count_and_preserve_resolution():
    from mesh import _size_groups
    sizes = np.geomspace(.05, 5, 845)
    records = [dict(status='retained', mesh_size=float(size), surface_tags=[i+1])
               for i, size in enumerate(sizes)]
    records.append(dict(status='outside', mesh_size=.001, surface_tags=[]))
    groups = _size_groups(records)
    assert len(groups) == 5
    assert sorted(tag for tags, _ in groups for tag in tags) == list(range(1, 846))
    for tags, size in groups:
        assert all(size <= sizes[tag-1] for tag in tags)
    for record in records[:-1]:
        record['mesh_size'] = .05
    assert _size_groups(records) == [(list(range(1, 846)), .05)]
