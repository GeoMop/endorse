from endorse import common
from endorse.Bukov2 import bh_chambers, mock, sa_problem, optimize_packers as opt_pack, sobol_fast
from endorse.Bukov2 import plot_boreholes
from pathlib import Path
script_dir = Path(__file__).absolute().parent


def test_bh_chamebres():
    workdir = script_dir
    cfg_file = workdir / "Bukov2_mesh.yaml"
    cfg = common.config.load_config(cfg_file)
    mock.mock_hdf5(cfg_file)
    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    bh_set = opt_pack.borehole_set(*opt_pack.load(cfg_file))
    i_bh = 20
    sobol_fn = sobol_fast.vec_sobol_total_only
    chambers = bh_chambers.Chambers.from_bh_set(cfg, problem, i_bh, sobol_fn, bh_set)
    all_chambers = chambers.all_chambers

    plot_boreholes.plot_chamber_data(chambers, list(problem['names']))