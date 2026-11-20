from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dfn_data import load_network, polygon_vertices
from bgem_adapter import to_fracture_sets


DATA = Path(__file__).resolve().parents[1] / 'DFN_gen-out_v01_d004'


def test_bgem_round_trip_matches_all_source_vertices_and_ids():
    network = load_network(DATA)
    groups = to_fracture_sets(network)
    for family, group in zip(network.families, groups):
        fr_set = group.fractures
        actual = np.einsum('nij,kj->nki', fr_set.transform_mat, fr_set.base_shape.vertices()) + fr_set.center[:, None, :]
        np.testing.assert_allclose(actual, polygon_vertices(family), atol=2e-13)
        np.testing.assert_array_equal(fr_set.family, family.arrays['fracture_set_idx'])
        np.testing.assert_array_equal(group.source_rows, family.source_rows)


@pytest.mark.parametrize('normal', [[0., 0., 1.], [0., 0., -1.]])
def test_bgem_round_trip_at_normal_poles(normal):
    network = load_network(DATA)
    family = network.families[0].select([0])
    arrays = dict(family.arrays)
    for axis, n, r in zip('xyz', normal, [arrays['fracture_aaxis'][0], 0, 0]):
        arrays[f'fracture_normal_{axis}'] = np.array([n])
        arrays[f'fracture_rvec_{axis}'] = np.array([r])
    family = replace(family, arrays=arrays)
    group = to_fracture_sets(replace(network, families=(family,)))[0]
    fs = group.fractures
    with np.errstate(invalid='ignore'):  # BGEM corrects its transient normal-pole division.
        actual = np.einsum('nij,kj->nki', fs.transform_mat, fs.base_shape.vertices()) + fs.center[:, None, :]
    np.testing.assert_allclose(actual, polygon_vertices(family), atol=1e-14)


def test_non_circular_bgem_conversion_fails_without_changing_geometry():
    network = load_network(DATA)
    family = network.families[0].select([0])
    family = replace(family, arrays={**family.arrays, 'fracture_ratio': np.array([0.5])})
    with pytest.raises(ValueError, match='non-circular'):
        to_fracture_sets(replace(network, families=(family,)))
