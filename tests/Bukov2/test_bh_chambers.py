import pytest
import endorse.Bukov2.bukov_common as bcommon
from endorse import common
from endorse.Bukov2 import bh_chambers, mock, sa_problem, optimize_packers as opt_pack, sobol_fast
from endorse.Bukov2 import plot_boreholes, boreholes
from pathlib import Path
script_dir = Path(__file__).absolute().parent

#@pytest.mark.skip
def test_bh_chamebres():

    workdir, cfg = bcommon.load_cfg(script_dir / "3d_model/Bukov2_mesh.yaml")
    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    i_bh = 8
    sobol_fn = sobol_fast.vec_sobol_total_only
    chambers = bh_chambers.Chambers.from_bh_set(workdir, cfg, bh_set, i_bh, problem, sobol_fn)
    all_chambers = chambers.all_chambers

    param_names = list(problem['names'])
    bh_workdir = workdir / f"plot_bh_{i_bh:03d}"
    bh_workdir.mkdir(parents=True, exist_ok=True)
    plots = plot_boreholes.PlotCfg(
        bh_workdir, cfg, bh_set, chambers,
        i_bh, param_names, show=False)
    plots.all()

@pytest.mark.skip
def test_optimize_packers():

    workdir, cfg = bcommon.load_cfg(script_dir / "3d_model/Bukov2_mesh.yaml")
    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    i_bh = 9
    sobol_fn = sobol_fast.vec_sobol_total_only
    chambers = bh_chambers.Chambers.from_bh_set(workdir, cfg, bh_set, i_bh, problem, sobol_fn)

    # Optimize
    best_packer_configs = bh_chambers.optimize_packers(cfg, chambers)