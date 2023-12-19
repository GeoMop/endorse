import numpy as np
import pytest
import timeit
import os
from endorse import common
from endorse.Bukov2 import boreholes, plot_boreholes
from  pathlib import Path
import pickle

script_dir = Path(__file__).absolute().parent

def test_length_of_transversals():
    z0 = -2
    z1 = 9
    z2 = 15
    # Input data
    line1 = np.array([-10, 0, z0, 1, 0, 0])
    lines = np.array([
        [7, 8, z1, 0, 1, 0],
        [13, 14, z2, 1, 1, 0],
    ])

    # Expected output
    expected_lengths = np.array([
        z1 - z0,
        z2 - z0
    ])

    # Run the function
    calculated_lengths = boreholes.BoreholeSet.transversal_lengths(line1, lines)

    # Assert (test) if the calculated lengths are close to the expected lengths
    np.testing.assert_allclose(calculated_lengths, expected_lengths, rtol=1e-6)

def test_transversal_params():
    z0 = -2
    z1 = 9
    z2 = 15
    # Input data
    line1 = np.array([-10, 0, z0, 1, 0, 0])

    line2 = np.array([7, 8, z1, 0, -1, 0])
    l, t1, t2, p1, p2, yz_tangent = boreholes.BoreholeSet.transversal_params(line1, line2)
    assert z1 - z0 == l
    np.testing.assert_allclose(p1, [7, 0, z0], rtol=1e-6)
    np.testing.assert_allclose(p2, [7, 0, z1], rtol=1e-6)
    np.testing.assert_allclose(yz_tangent, [0, 1, 0], rtol=1e-6)

    line2 = np.array([0, -14, z2, 1, 1, 0])
    l, t1, t2, p1, p2, yz_tangent = boreholes.BoreholeSet.transversal_params(line1, line2)
    assert z2 - z0 == l
    np.testing.assert_allclose(p1, [14, 0, z0], rtol=1e-6)
    np.testing.assert_allclose(p2, [14, 0, z2], rtol=1e-6)
    np.testing.assert_allclose(yz_tangent, [0, -1, 0], rtol=1e-6)


def test_read_fields():
    pattern = script_dir / 'flow_reduced' / 'flow_*.vtu'
    pressure_array, mesh = boreholes.get_time_field(str(pattern), 'pressure_p0')
    assert pressure_array.shape == (10, 82609)

    # test identification of IDs and value cumsum for two lines
    lines = np.array(
        [[[-10, 0, 1.5], [1, 1, 0.5]],
         [[10, 0, 1.5], [-1, -1, 0.5]]]
    )

    points = boreholes.line_points(lines, 100)
    def call():
        boreholes.interpolation_slow(mesh, points)
    print("time: ", timeit.timeit(call, number=1))

    id_matrix = boreholes.interpolation_slow(mesh, points)
    cumlines = boreholes.cum_borehole_array(pressure_array[None, :, :], id_matrix)
    assert cumlines.shape == (2, 100, 10, 1)  # n_lines, n_samples, n_times, n_points

    #point_values = boreholes.get_values_on_lines(mesh, pressure_array, lines, n_points=10)
    #assert point_values.shape == (2, pressure_array.shape[0], 3)


#@pytest.mark.skip
def test_borhole_set():
    cfg = common.config.load_config(script_dir / "Bukov2_mesh.yaml")
    plotter = plot_boreholes.create_scene(cfg.geometry)

    bh_set = boreholes.BoreholeSet.from_cfg(cfg.boreholes.zk_30)
    print("N boreholes:", bh_set.n_boreholes)
    plot_boreholes.plot_bh_set(plotter, bh_set)
    plotter.camera.parallel_projection = True
    plotter.show()

    # Construct test interpolation ID matrix and cumsum on boreholes.
    pattern = script_dir / 'flow_reduced' / 'flow_*.vtu'
    pressure_array, mesh = boreholes.get_time_field(str(pattern), 'pressure_p0')
    ref_field_on_lines = bh_set.project_field(mesh, pressure_array[None, :, :], cached=True)

    # Test serialization
    serialized = pickle.dumps(bh_set)
    new_bh_set = pickle.loads(serialized)
    new_field_on_lines = new_bh_set.project_field(mesh, pressure_array[None, :, :], cached=True)
    np.testing.assert_allclose(ref_field_on_lines, new_field_on_lines, rtol=1e-6)