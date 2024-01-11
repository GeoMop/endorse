from typing import *
import attrs
import itertools
from endorse.Bukov2 import boreholes
from endorse.sa import analyze
from endorse.Bukov2 import sobol_fast
from endorse import common
import numpy as np
from endorse.sa.analyze import sobol_vec
from functools import cached_property
from endorse.Bukov2 import sobol_fast, bukov_common as bcommon, sa_problem, plot_boreholes, bh_chambers

def borehole_dir(workdir, i_bh):
    bh_dir = workdir / "processed_bh" / f"bh_{i_bh:03d}"
    bh_dir.mkdir(parents=True, exist_ok=True)
    return bh_dir

def process_borehole(workdir, i_bh):
    bh_workdir = borehole_dir(workdir, i_bh)
    return _process_borehole(bh_workdir, workdir, i_bh)

@bcommon.memoize
def _optimize_borehole(workdir, cfg, chambers):
    return bh_chambers.optimize_packers(cfg, chambers)


@bcommon.memoize
def _process_borehole(bh_workdir, workdir, i_bh):
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    bh_field = boreholes.project_field(workdir,cfg, bh_set, from_sample=0, force=cfg.boreholes.force)
    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    sobol_fn = sobol_fast.vec_sobol_total_only
    chambers = bh_chambers.Chambers.from_bh_set(workdir, cfg, bh_field, i_bh, problem, sobol_fn)

    best_packer_configs = _optimize_borehole(bh_workdir, cfg, chambers)
    # shuld not be necessary as the whole funcion result is memoized
    #bcommon.pkl_write(bh_workdir, best_packer_configs, "best_packer_configs.pkl")
    print("Plotting")
    param_names = list(problem['names'])
    plots = plot_boreholes.PlotCfg(
        bh_workdir, cfg, bh_set, chambers,
        i_bh, best_packer_configs, param_names, show=False)
    plots.all()

    return best_packer_configs
