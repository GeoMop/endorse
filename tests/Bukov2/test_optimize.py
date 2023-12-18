from endorse import common
from endorse.Bukov2 import optimize, boreholes
from multiprocessing import Pool
from pathlib import Path
script_dir = Path(__file__).absolute().parent


def test_optimize():
    cfg = common.config.load_config(script_dir / "Bukov2_mesh.yaml")
    bh_set = boreholes.BHS_zk_30()
    with Pool() as pool:
        optimize.optimize(cfg.optimize, bh_set, pool.map)