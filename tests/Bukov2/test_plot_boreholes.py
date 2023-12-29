import os
from endorse import common
from endorse.Bukov2 import plot_boreholes as pb
script_dir = os.path.dirname(os.path.abspath(__file__))



def test_borehole_scene():
    workdir = script_dir
    conf_file = os.path.join(workdir, "./Bukov2_mesh.yaml")
    cfg = common.config.load_config(conf_file)
    mesh_file = pb.create_scene(cfg.geometry)
