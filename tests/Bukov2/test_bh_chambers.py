import pytest
import endorse.Bukov2.bukov_common as bcommon
from endorse import common
from endorse.Bukov2 import bh_chambers, mock, sa_problem, sobol_fast, process_bh
from endorse.Bukov2 import plot_boreholes, boreholes
from pathlib import Path
script_dir = Path(__file__).absolute().parent

#@pytest.mark.skip
def test_bh_chamebres():
    # Full borehole processing: optimization + plotting
    workdir, cfg = bcommon.load_cfg(script_dir / "3d_model/Bukov2_mesh.yaml")
    i_bh = 13
    process_bh.process_borehole(workdir, i_bh)
    # sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    # problem = sa_problem.sa_dict(sim_cfg)
    # bh_set = boreholes.make_borehole_set(workdir, cfg)
    #
    # sobol_fn = sobol_fast.vec_sobol_total_only
    # chambers = bh_chambers.Chambers.from_bh_set(workdir, cfg, bh_set, i_bh, problem, sobol_fn)
    #
    # # Optimize
    # best_packer_configs = bh_chambers.optimize_packers(cfg, chambers)
    #
    # param_names = list(problem['names'])
    # bh_workdir = workdir / f"plot_bh_{i_bh:03d}"
    # bh_workdir.mkdir(parents=True, exist_ok=True)
    # plots = plot_boreholes.PlotCfg(
    #     bh_workdir, cfg, bh_set, chambers,
    #     i_bh, best_packer_configs, param_names, show=False)
    # plots.all()

@pytest.mark.skip
def test_optimize_packers():
    # Optimization of packers only ( about half that time)
    # reduced number of points
    workdir, cfg = bcommon.load_cfg(script_dir / "3d_model/Bukov2_mesh.yaml")
    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    bh_field = boreholes.project_field(workdir, cfg, bh_set, from_sample=0, force=cfg.boreholes.force)
    # bh_field = boreholes.project_field(workdir, cfg, bh_set, force=cfg.boreholes.force)

    i_bh = 19
    sobol_fn = sobol_fast.vec_sobol_total_only
    chambers = bh_chambers.Chambers.from_bh_set(workdir, cfg, bh_field, i_bh, problem, sobol_fn)
    #plot_boreholes.plot_sensitivity_histograms(chambers.all_chambers[1], problem['names'])
    # Optimize
    best_packer_configs = bh_chambers.optimize_packers(cfg, chambers)