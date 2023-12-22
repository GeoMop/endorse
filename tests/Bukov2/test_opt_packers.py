import time

import pytest
from endorse import common
from endorse.Bukov2 import boreholes, sa_problem, mock
from multiprocessing import Pool

from endorse.sa import sample, analyze
import numpy as np
import endorse.Bukov2.optimize as optimize
import endorse.Bukov2.optimize_packers as opt_pack

from pathlib import Path
script_dir = Path(__file__).absolute().parent


def test_borehole_set():
    workdir = script_dir
    cfg_file = workdir / "Bukov2_mesh.yaml"
    cfg = common.config.load_config(cfg_file)
    mock.mock_hdf5(cfg_file)
    bh_set = opt_pack.borehole_set(*opt_pack.load(cfg_file))


def compare_opt_results(a, b):
    for ref_bh, new_bh in zip(a, b):
        for ref, new in zip(ref_bh, new_bh):
            assert np.allclose(ref.packers, new.packers)
            assert np.allclose(ref.sobol_indices, new.sobol_indices)


# def test_fast_sobol():
#     workdir = script_dir
#     cfg_file = workdir / "Bukov2_mesh.yaml"
#     cfg = common.config.load_config(cfg_file)
#     mock.mock_hdf5(cfg_file)
#     bh_set = opt_pack.borehole_set(*opt_pack.load(cfg_file))
#     field = bh_set.project_field(None, None, cached=True)
#     values = field[20, :, :, :]
#     sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
#     problem = sa_problem.sa_dict(sim_cfg)
#     analyze.sobol_vec(values, problem)

def opt_fast_sobol(workdir, cfg, bh_set, i_bh):
    # Test fast sobol
    np.random.seed(123)
    start = time.process_time_ns()
    borhole_opt_config_fast = opt_pack.optimize_borehole(workdir, cfg, bh_set, i_bh, sobol_fn=opt_pack.vec_sobol_total_only)
    sec = (time.process_time_ns() - start) / 1e9
    print("Fast sobol time: ", sec)
    return  borhole_opt_config_fast

def opt_full_sobol(cfg_file, i_bh):
    # Tast clasical Sobol
    np.random.seed(123)
    start = time.process_time_ns()
    borhole_opt_config = opt_pack.optimize_borehole_wrapper((cfg_file, i_bh))
    sec = (time.process_time_ns() - start) / 1e9
    print("Full sobol time: ", sec)
    assert type(borhole_opt_config) is list
    assert type(borhole_opt_config[0]) is opt_pack.PackerConfig
    return  borhole_opt_config

#@pytest.mark.skip
def test_optimize_packer():
    workdir = script_dir
    cfg_file = workdir / "Bukov2_mesh.yaml"
    cfg = common.config.load_config(cfg_file)
    mock.mock_hdf5(cfg_file)
    bh_set = opt_pack.borehole_set(*opt_pack.load(cfg_file))
    i_bh = 20

    borhole_opt_config = opt_fast_sobol(workdir,cfg,bh_set,i_bh)
    print("N configurations per borehole: ", len(borhole_opt_config))
    for i_param in range(borhole_opt_config[0].n_param):
        i_max = np.argmax([conf.param_sensitivity[i_param] for conf in borhole_opt_config])
        conf_max = borhole_opt_config[i_max]
        print(f"param {i_param}, packers {conf_max.packers}, sens {conf_max.chamber_sensitivity}, ")

    print('\n'.join([str(item) for item in borhole_opt_config]))

    # write mock result file
    bh_mock_results = [borhole_opt_config for _ in range(bh_set.n_boreholes)]
    opt_pack.write_optimization_results(workdir, bh_mock_results)
    bh_mock_results_1 = opt_pack.read_optimization_results(workdir)
    compare_opt_results(bh_mock_results, bh_mock_results_1)

    # Compare against reference (about 500 times slower)
    #borhole_opt_config_full  = opt_full_sobol(cfg_file, i_bh)
    #for ref, new in zip(borhole_opt_config, borhole_opt_config_full):
    #    if np.all(ref.packers ==  new.packers):
    #        assert np.allclose(ref.chamber_sensitivity, new.chamber_sensitivity)


@pytest.mark.skip
def test_optimize_bh_set():
    workdir = script_dir
    cfg_file = workdir / "Bukov2_mesh.yaml"
    bh_set = opt_pack.borehole_set(*opt_pack.load(cfg_file))
    bhs_opt_config = opt_pack.optimize_bh_set(cfg_file, map)
    assert type(bhs_opt_config) is list
    assert len(bhs_opt_config) == bh_set.n_boreholes
    assert type(bhs_opt_config[0]) is list
    assert type(bhs_opt_config[0][0]) is opt_pack.PackerConfig


@pytest.mark.skip
def test_optimize_scoop():
    workdir = script_dir
    cfg_file = workdir / "Bukov2_mesh.yaml"
    cfg = common.config.load_config()
    make_hdf5(workdir, cfg)

    opt_pack.main_scoop(cfg_file)
