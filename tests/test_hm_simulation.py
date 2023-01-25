import pytest
import os
import numpy as np

from endorse import hm_simulation
from endorse import common, mesh_class
from endorse.common.memoize import File
script_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.skip
def test_run_random_samples():
    # Test execution of few random samples from the Byes inversion
    # implementation steps:
    # DONE - run single sample HM simulation equivelent to TSX model
    # - modify template to get conductivity field and porosity (possibly read results in GMSH and compute these two fields)
    # - merge config_txt and config_homogenisation
    # - run 1 sample, HM simulation but with modified geometry (2.2m diameter borehole)
    # - run 1 sample, HM simulation but with modified geometry, interpolation to other 3D tranport mesh
    # - run 1 sample, HM simulation but with modified geometry, interpolation to other 3D tranport mesh, homogenized transport execution (under 10mins localy)
    #common.EndorseCache.instance().expire_all()

    conf_file = os.path.join(script_dir, "test_data/config.yaml")
    np.random.seed(seed)
    cfg = common.load_config(conf_file)

    files_to_copy = ["test_data/accepted_parameters.csv"]
    with common.workdir(f"sandbox/hm_model_{seed}", inputs=files_to_copy, clean=False):
        hm_simulation.run_random_samples(cfg, 1)


@pytest.mark.skip
def test_tunnel_interpolation():
    conf_file = os.path.join(script_dir, "test_data/config.yaml")
    cfg = common.load_config(conf_file)
    flow_msh = mesh_class.Mesh.load_mesh(File(os.path.join(script_dir, "test_data/flow_fields.msh")))
    mech_msh = mesh_class.Mesh.load_mesh(File(os.path.join(script_dir, "test_data/mechanics_fields.msh")))

    mesh_interp = hm_simulation.TunnelInterpolator(cfg.geometry, flow_msh=flow_msh, mech_msh=mech_msh)
    points = np.array([[10, 20], [4.375 / 2 + 0.01, 0], [1.1, 3.5 / 2 + 0.01 + 1.1]])
    selected_time = 365 * 3600 * 24  # end_time of hm simulation

    field_name = "conductivity"
    print(field_name, points, selected_time)
    vals = mesh_interp.interpolate_field(field_name, points, time=selected_time)
    print(vals)

    mesh_interp.test_conductivity()

    vals = mesh_interp.compute_porosity(cfg.tsx_hm_model.hm_params, points, time=selected_time)
    print(vals)

    mesh_interp.test_porosity(cfg.tsx_hm_model.hm_params)


# @pytest.mark.skip
def test_run_single_sample():
    seed = 102
    np.random.seed(seed)
    conf_file = os.path.join(script_dir, "test_data/config_homo_tsx.yaml")
    cfg = common.load_config(conf_file)

    files_to_copy = ["test_data/accepted_parameters.csv"]
    with common.workdir(f"sandbox/hm_model_{seed}", inputs=files_to_copy, clean=False):
        fo = hm_simulation.run_single_sample(cfg)
        mesh_interp = hm_simulation.TunnelInterpolator(cfg.geometry, flow123d_output=fo)

        points = np.array([[10, 20], [4.375 / 2 + 0.01, 0], [1.1, 3.5 / 2 + 0.01 + 1.1]])
        selected_time = 365 * 3600 * 24  # end_time of hm simulation

        field_name = "conductivity"
        vals = mesh_interp.interpolate_field(field_name, points, time=selected_time)
        init_porosity, porosity = mesh_interp.compute_porosity(cfg.tsx_hm_model.hm_params, points, time=selected_time)

