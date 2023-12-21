import pytest
from endorse import common
from endorse.Bukov2 import boreholes, sa_problem, mock
from multiprocessing import Pool
from pathlib import Path
from endorse.sa import sample
import numpy as np
import endorse.Bukov2.optimize as optimize
import endorse.Bukov2.optimize_packers as opt_pack
script_dir = Path(__file__).absolute().parent


def test_borehole_set():
    workdir = script_dir
    cfg_file = workdir / "Bukov2_mesh.yaml"
    cfg = common.config.load_config(cfg_file)
    mock.mock_hdf5(cfg_file)
    bh_set = opt_pack.borehole_set(*opt_pack.load(cfg_file))

#@pytest.mark.skip
def test_optimize_packer():
    workdir = script_dir
    cfg_file = workdir / "Bukov2_mesh.yaml"
    cfg = common.config.load_config(cfg_file)
    mock.mock_hdf5(cfg_file)
    bh_set = opt_pack.borehole_set(*opt_pack.load(cfg_file))
    i_bh = 20
    borhole_opt_config = opt_pack.optimize_borehole_wrapper((cfg_file, i_bh))
    assert type(borhole_opt_config) is list
    assert type(borhole_opt_config[0]) is opt_pack.PackerConfig

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

    for ref_bh, new_bh in zip(bh_mock_results, bh_mock_results_1):
        for ref, new in zip(ref_bh, new_bh):
            assert np.allclose(ref.packers, new.packers)
            #inf_mask1 = np.isinf(ref.sobol_indices)
            #inf_mask2 = np.isinf(new.sobol_indices)
            #mask = ~inf_mask1 & ~inf_mask2
            #assert np.allclose(ref.sobol_indices[mask], new.sobol_indices[mask], equal_nan=True)
            assert np.allclose(ref.sobol_indices, new.sobol_indices)
            # Have to fix Nans and Infs yet.

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
