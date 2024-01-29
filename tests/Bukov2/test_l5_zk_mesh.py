import os
from endorse.Bukov2.l5_mesh import make_mesh
from endorse import common
script_dir = os.path.dirname(os.path.realpath(__file__))

def test_l5_zk_mesh():
    make_mesh(script_dir)