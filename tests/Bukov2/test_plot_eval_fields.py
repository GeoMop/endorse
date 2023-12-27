import endorse.Bukov2.bukov_common as bcommon
from endorse import common
from endorse.Bukov2 import mock, optimize_packers as opt_pack, plot_boreholes
import numpy as np

from pathlib import Path
script_dir = Path(__file__).absolute().parent



def test_plot_eval_fields():
    workdir = script_dir
    cfg_file = workdir / "Bukov2_mesh.yaml"
    cfg = common.config.load_config(cfg_file)
    mock.mock_hdf5(cfg_file)
    bh_set = opt_pack.borehole_set(*bcommon.load_cfg(cfg_file))
    plot_boreholes.PVD_data_on_bhset(workdir, bh_set)
