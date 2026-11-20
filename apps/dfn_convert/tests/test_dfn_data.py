from pathlib import Path

import numpy as np
import pytest

import dfn_data as data


DATA = Path(__file__).resolve().parents[1] / 'DFN_gen-out_v01_d004'


def test_rotated_hexagon_preserves_radius_phase_and_normal():
    # Swapping the two in-plane axes or treating aaxis as diameter breaks this.
    vertices = data.ellipse_polygon(
        np.array([[1., 2., 3.]]), np.array([[0., 1., 0.]]),
        np.array([[2., 0., 0.]]), np.array([2.]), np.array([1.]), 6,
    )[0]
    np.testing.assert_allclose(vertices[0], [1 + np.sqrt(3), 2, 2])
    np.testing.assert_allclose(vertices[1], [1, 2, 1], atol=1e-14)
    assert data.polygon_area(vertices) == pytest.approx(6 * np.sqrt(3))


def test_non_circular_polygon_uses_source_angular_sampling():
    vertices = data.ellipse_polygon(
        np.zeros((1, 3)), np.array([[0., 0., 1.]]),
        np.array([[2., 0., 0.]]), np.array([2.]), np.array([0.5]), 6,
    )[0]
    np.testing.assert_allclose(vertices[0], [4*np.sqrt(3/13), 1/np.sqrt(13), 0])
    np.testing.assert_allclose(vertices[1], [0, 1, 0], atol=1e-14)


def test_load_supplied_network_and_scale():
    network = data.load_network(DATA)
    assert [len(f) for f in network.families] == [5327, 2250, 832]
    assert [f.name for f in network.families] == ['EW', 'NS_FOL', 'NS_SH']
    np.testing.assert_array_equal(network.bounds, [[-2, 2]]*3)
    crossing = 0
    for family in network.families:
        vertices = data.polygon_vertices(family)
        np.testing.assert_allclose(data.polygon_area(vertices), np.pi*family.arrays['fracture_req']**2, atol=5e-15)
        crossing += np.any((vertices < -2) | (vertices > 2), axis=(1, 2)).sum()
    assert crossing == 684
    scaled = data.load_network(DATA, coordinate_scale=10)
    np.testing.assert_allclose(data.polygon_vertices(scaled.families[0]), 10*data.polygon_vertices(network.families[0]))
    np.testing.assert_array_equal(scaled.families[0].source_rows, np.arange(5327))


def test_missing_realization_fails_instead_of_loading_another():
    with pytest.raises(FileNotFoundError):
        data.load_network(DATA, realization=1)


def test_crop_keeps_polygons_intersecting_box_even_with_centers_outside():
    network = data.load_network(DATA)
    bounds = np.array([[-0.1, 0.1]]*3)
    cropped = data.crop_network(network, bounds)
    kept = {f.source.name: set(f.source_rows) for f in cropped.families}
    for family in network.families:
        vertices = data.polygon_vertices(family)
        overlap = np.all((vertices.max(axis=1) >= bounds[:, 0]) & (vertices.min(axis=1) <= bounds[:, 1]), axis=1)
        assert kept[family.source.name] == set(np.flatnonzero(overlap))
    assert any(np.any(np.abs(f.center) > 0.1) for f in cropped.families)


def test_npz_polygons_match_vtp_vertices():
    pytest.importorskip('pyvista')
    result = data.compare_vtp(data.load_network(DATA), DATA / 'fractures_r000.vtp')
    assert result['matched'] == 8409
    assert result['max_vertex_error'] < 1e-12


def test_box_clipping_preserves_plane_and_clipped_area():
    polygon = np.array([[-2., -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]])
    clipped = data.clip_polygon_to_box(polygon, np.array([[-.5, .5]]*3))
    assert data.polygon_area(clipped) == pytest.approx(1)
    assert len(clipped) == 4
    assert np.all(clipped[:, 2] == 0)
    assert len(data.clip_polygon_to_box(polygon + [0, 0, 3], np.array([[-.5, .5]]*3))) == 0


@pytest.mark.parametrize('corruption', ['count', 'family_id'])
def test_inconsistent_npz_metadata_fails_early(tmp_path, corruption):
    import yaml
    network = data.load_network(DATA)
    family = network.families[0].select([0])
    source = tmp_path/family.source.name
    metadata = yaml.safe_load((DATA/'DFN_gen-vars.yaml').read_text())
    metadata['domain_frsets_names'] = [['EW']]
    (tmp_path/'DFN_gen-vars.yaml').write_text(yaml.safe_dump(metadata))
    info = yaml.safe_load(family.source.with_suffix('.yaml').read_text())
    info['number_of_fractures'] = 2 if corruption == 'count' else 1
    source.with_suffix('.yaml').write_text(yaml.safe_dump(info))
    arrays = dict(family.arrays)
    if corruption == 'family_id':
        arrays['fracture_set_idx'] = np.array([2])
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match='metadata'):
        data.load_network(tmp_path)


@pytest.mark.parametrize('rmin,rmax,expected', [
    (2., 3., {'small': [8, 12]}),
    (None, 2., {'small': [4, 8]}),
    (3., None, {'small': [12], 'large': [3, 10]}),
    (2., 2., {'small': [8]}),
    (6., None, {}),
    (None, None, {'small': [4, 8, 12], 'large': [3, 10]}),
])
def test_radius_filter_is_inclusive_and_preserves_source_rows(rmin, rmax, expected):
    # Radii are already scaled; different aaxis values catch use of the wrong field.
    families = (
        data.FamilyData(Path('small.npz'), 'small',
                        {'fracture_req': np.array([1., 2., 3.]),
                         'fracture_aaxis': np.array([10., 20., 30.])}, np.array([4, 8, 12])),
        data.FamilyData(Path('large.npz'), 'large',
                        {'fracture_req': np.array([4., 5.]),
                         'fracture_aaxis': np.array([40., 50.])}, np.array([3, 10])),
    )
    network = data.NetworkData(np.array([[-20., 20.]]*3), families, coordinate_scale=10.)
    filtered = data.filter_by_radius(network, rmin, rmax)
    assert {f.name: f.source_rows.tolist() for f in filtered.families} == expected
    np.testing.assert_array_equal(filtered.bounds, network.bounds)
    assert filtered.coordinate_scale == 10.
    for family in filtered.families:
        original = next(f for f in families if f.name == family.name)
        positions = np.searchsorted(original.source_rows, family.source_rows)
        for name, array in original.arrays.items():
            np.testing.assert_array_equal(family.arrays[name], array[positions])
    assert [len(f) for f in network.families] == [3, 2]


@pytest.mark.parametrize('rmin,rmax', [(-1., None), (None, -1.), (3., 2.), (np.nan, 2.)])
def test_invalid_radius_interval_fails_early(rmin, rmax):
    network = data.NetworkData(np.array([[-1., 1.]]*3), ())
    with pytest.raises(ValueError, match='rmin.*rmax'):
        data.filter_by_radius(network, rmin, rmax)
