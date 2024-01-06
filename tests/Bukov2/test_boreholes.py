import itertools

import h5py
import numpy as np
import pytest
import timeit
import os
import pyvista as pv
import endorse.Bukov2.bukov_common as bcommon
from endorse import common
from endorse.Bukov2 import boreholes, plot_boreholes, mock
import shutil
from  pathlib import Path
import pickle

script_dir = Path(__file__).absolute().parent


@pytest.mark.skip
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
    l, t1, t2, p1, p2, yz_tangent = boreholes.Lateral.transversal_params(line1, line2)
    assert z1 - z0 == l
    np.testing.assert_allclose(p1, [7, 0, z0], rtol=1e-6)
    np.testing.assert_allclose(p2, [7, 0, z1], rtol=1e-6)
    np.testing.assert_allclose(yz_tangent, [0, 1, 0], rtol=1e-6)

    line2 = np.array([0, -14, z2, 1, 1, 0])
    l, t1, t2, p1, p2, yz_tangent = boreholes.Lateral.transversal_params(line1, line2)
    assert z2 - z0 == l
    np.testing.assert_allclose(p1, [14, 0, z0], rtol=1e-6)
    np.testing.assert_allclose(p2, [14, 0, z2], rtol=1e-6)
    np.testing.assert_allclose(yz_tangent, [0, -1, 0], rtol=1e-6)

    z0 = 0
    z2 = -2
    line1 = np.array([0, 0, z0, 1, 0, 0])
    line2 = np.array([0, 1, z2, 4, 0, 1])
    l, t1, t2, p1, p2, yz_tangent = boreholes.Lateral.transversal_params(line1, line2)
    assert l == 1.0
    np.testing.assert_allclose(p1, [8, 0, 0], rtol=1e-6)
    np.testing.assert_allclose(p2, [8, 1, 0], rtol=1e-6)
    np.testing.assert_allclose(yz_tangent, [0, 0, -1], rtol=1e-6)

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
    assert id_matrix.shape == (2, 100)  # n_lines,  n_points

    #point_values = boreholes.get_values_on_lines(mesh, pressure_array, lines, n_points=10)
    #assert point_values.shape == (2, pressure_array.shape[0], 3)



@pytest.mark.skip
def test_borhole_set():
    workdir, cfg = bcommon.load_cfg(script_dir / "3d_model_mock/Bukov2_mesh.yaml")
    lateral = boreholes.Lateral.from_cfg(cfg.boreholes.zk_30)
    bh_set = lateral.set_from_cfg(cfg.boreholes.zk_30)
    print("N boreholes:", bh_set.n_boreholes)

    bh_set.boreholes_print_sorted()

    plotter = pv.Plotter()
    plotter = plot_boreholes.create_scene(plotter, cfg.geometry)
    plot_boreholes.add_cylinders(plotter, lateral)
    plot_boreholes.plot_bh_set(plotter, bh_set)
    plotter.camera.parallel_projection = True
    plotter.show()

@pytest.mark.skip
def test_from_end_points():
    workdir, cfg = bcommon.load_cfg(script_dir / "3d_model_mock/Bukov2_mesh.yaml")

    lateral = boreholes.Lateral.from_cfg(cfg.boreholes.zk_30)
    y_pos = -9.5

    lines = [
        ([0.4, -11, -0.3], [5, 0, -3.4], 1),
        ([0.4, -9, -0.3], [11, 0, 3.4], 1),
    ]

    bh_set = lateral.set_from_points(lines)
    print("N boreholes:", bh_set.n_boreholes)

    bh_set.boreholes_print_sorted()

    # for i_bh, bh in enumerate(bh_set.boreholes):
    #     print(i_bh)
    #     plotter = plot_boreholes.plot_borehole_position(cfg, bh)
    #     plot_boreholes.save_projections(plotter, workdir, bh.id + ".svg")

    # plotter = pv.Plotter()
    # plotter = plot_boreholes.create_scene(plotter, cfg.geometry)
    # plot_boreholes.add_cylinders(plotter, lateral)
    # plot_boreholes.plot_bh_set(plotter, bh_set)
    # plotter.camera.parallel_projection = True
    # plotter.show()

    plot_boreholes.export_vtk_bh_set(workdir, bh_set)


def real_bh_set():
    workdir, cfg = bcommon.load_cfg(script_dir / "3d_model_mock/Bukov2_mesh.yaml")

    lateral = boreholes.Lateral.from_cfg(cfg.boreholes.zk_30)
    def promote_to_list(l):
        if not isinstance(l, list):
            l = [l]
        return l

    def lines(start_x, start_y, start_z, end_x, end_y, end_z, group):
        ptl = promote_to_list
        parm_lists = map(ptl, [start_x,start_y, start_z, end_x, end_y, end_z])
        return [([sx, sy, sz], [ex,ey,ez], 30, group) for sx,sy,sz,ex,ey,ez in itertools.product(
            *parm_lists)
        ]

    def union(*lists):
        return [x for l in lists for x in l]

    l_set = union(
        # left long variants
        lines(0.4, 20 , -0.3,
              [10, 12, 14], 0, [3.2, 3.7, 4.4, 5.2, 6], "left_up"),
        lines(0.4, 20 , -0.3,
              [10, 12, 14], 0, [-6, -5.2, -4.5, -3.8, -3.3], "left_dn"),
         lines(0.4, 20, -0.3,
              [7, 8, 9], 0, [-6, -5.2, -4.5, -3.8, -3.3], "left_mid_dn"),

        # middle variants
        lines(0.4, 5 , -0.3,
        13, [5, 4.2, 3.5, 3], [-1.5, 0, 1.5], "middle"),
        lines(0.4, [-7, -9], -0.3,
              [10, 12, 14],  0, [3.2, 3.7, 4.4, 5.2, 6], "right_far_up"),

        # Vetšina v kolizi s foliací
        ##lines(0.4, [-11, -13], -0.3,
        ##      [10, 12, 14],  0, [-6, -5.2, -4.5, -3.8, -3.3], "right_far_dn"),

        lines(0.4, [-7, -9], -0.3,
              [6, 7, 8, 9],  0, [3.2, 3.7, 4.4, 5.2, 6], "right_mid_up"),
        lines(0.4, -19, -0.3,
              [6, 7, 8, 9],  0, [-6, -5.2, -4.5, -3.8, -3.3], "left_mid_dn"),
        lines(0.4, -9, -0.3,
              [3, 4, 5],  0, [3.3, 3.7, 4.4, 5.2, 6], "right_near_up"),
        lines(0.4, -13, -0.3,
              [3, 4, 5],  0, [-6, -5.2, -4.5, -3.8, -3.3], "right_near_dn"),
        lines(0.4, [-5, -6], -0.3,
              13, [-3, -3.5, -4.2 , -5, -6], [-1.5, 0, 1.5], "right_middle"),
    )

    bh_set = lateral.set_from_points(l_set)
    return bh_set

@pytest.mark.skip
def test_from_end_points_real():
    bh_set = real_bh_set()
    print("N boreholes:", bh_set.n_boreholes)

    bh_set.boreholes_print_sorted()

    plotter = pv.Plotter()
    plotter = plot_boreholes.create_scene(plotter, cfg.geometry)
    plot_boreholes.add_cylinders(plotter, lateral)
    # fol_start = starts[0]
    # fol_dir = np.transpose(lateral.transform_matrix) @ (boreholes.Borehole._direction(lateral.foliation_longitude-lateral.l5_azimuth, lateral.foliation_latitude))
    # fol_end = starts[0] + fol_dir
    # fol_bh = lateral._make_bh([fol_start, fol_dir, fol_end, 40, [2, 38]])
    # plot_boreholes.add_bh(plotter, fol_bh,'green', 'F')
    plot_boreholes.plot_bh_set(plotter, bh_set)
    plotter.camera.parallel_projection = True
    plotter.show()




#@pytest.mark.skip
def test_field_projection():
    """
    Test projection of the full pressure field to the borehole points.
    Tests:
    - shape of output field in the file
    :return:
    """

    workdir, cfg = bcommon.load_cfg(script_dir / "3d_model/Bukov2_mesh.yaml")
    shutil.rmtree((workdir / "borehole_data"))
    #mock.mock_hdf5(cfg_file)
    #sim_cfg = load_config(workdir / cfg.simulation.cfg)
    #problem = sa_problem.sa_dict(sim_cfg)
    bh_set = real_bh_set()
    #input_hdf, field_shape = mock.mock_hdf5(cfg_file)
    #cfg.simulation.hdf = input_hdf

    bh_range = (10, 30)
    updated_files = bh_set.project_field(workdir, cfg, bh_range)
    print("Updated: ", updated_files)

    for f in updated_files:
        with h5py.File(f, mode='r') as f:
            dset = f['pressure']
            n_points = bh_set.n_points
            n_times = 5
            n_samples = 96
            assert dset.shape == (n_points, n_times, n_samples)
    #ref_field_on_lines = bh_set.project_field(mesh, pressure_array[None, :, :], cached=True)

    # Test serialization
    # serialized = pickle.dumps(bh_set)
    # new_bh_set = pickle.loads(serialized)
    # new_field_on_lines = new_bh_set.project_field(mesh, pressure_array[None, :, :], cached=True)
    # np.testing.assert_allclose(ref_field_on_lines, new_field_on_lines, rtol=1e-6)
