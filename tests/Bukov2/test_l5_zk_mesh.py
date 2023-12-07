import os
from endorse.Bukov2.l5_mesh import l5_zk_mesh
from endorse import common
script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))

def test_l5_zk_mesh():
    conf_file = os.path.join(script_dir, "./Bukov2_mesh.yaml")
    cfg = common.config.load_config(conf_file)
    mesh = l5_zk_mesh(cfg.geometry, cfg.mesh)