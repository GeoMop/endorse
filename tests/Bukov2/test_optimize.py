from endorse import common
from endorse.Bukov2 import boreholes, sa_problem
from multiprocessing import Pool
from pathlib import Path
from endorse.sa import sample
import numpy as np
import endorse.Bukov2.optimize as optimize
script_dir = Path(__file__).absolute().parent


def make_bh_set(cfg, force=False):
    cfg_sim = common.config.load_config(script_dir / "2d_model" /  "config_sim_C02hm.yaml")
    cfg.optimize.problem = cfg_sim
    bh_set_file = script_dir / "bh_set.pickle"
    bh_set =  optimize.get_bh_set(bh_set_file)
    if force or bh_set is None:
        bh_set = boreholes.BoreholeSet.from_cfg(cfg.boreholes.zk_30)
        pattern = script_dir / 'flow_reduced' / 'flow_*.vtu'
        pressure_array, mesh = boreholes.get_time_field(str(pattern), 'pressure_p0')

        # Fictious model
        sa_dict = sa_problem.sa_dict(cfg_sim)
        param_samples = sample.saltelli(sa_dict, 16, sa_dict['second_order'])


        sigma_x = param_samples[:, 1]/ 1e6  # about 48
        sigma_y = param_samples[:, 2]/ 1e6  # about 19
        sum_other = np.sum(param_samples, axis=1)

        field_samples = sigma_x[:, None, None] * pressure_array[None, :, :] + \
                        (pressure_array[None, :, :] ** 2 ) / sigma_y[:, None, None] + sum_other[:, None, None]

        bh_set.project_field(mesh, field_samples, cached=True)
        optimize.save_bh_data(script_dir / "bh_set.pickle", bh_set)
    return bh_set

def test_optimize():
    cfg = common.config.load_config(script_dir / "Bukov2_mesh.yaml")
    bh_set = make_bh_set(cfg) #, force=True)
    with Pool() as pool:
        optimize.optimize(cfg.optimize, bh_set, pool.map)