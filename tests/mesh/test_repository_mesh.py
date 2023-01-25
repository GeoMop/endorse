# Test src/repository_mesh.py
# fine EDZ: about 4M elements, 4 minutes

import os
from endorse import common
from endorse.mesh import repository_mesh, mesh_tools
#from bgem.stochastic.fracture import Fracture


script_dir = os.path.dirname(os.path.realpath(__file__))


def test_make_mesh():
    common.EndorseCache.instance().expire_all()
    # about 280 k elements
    # conf_file = os.path.join(script_dir, "./config_full_coarse.yaml")
    conf_file = os.path.join(script_dir, "../test_data/config.yaml")
    cfg = common.load_config(conf_file)
    with common.workdir("sandbox"):
        #fractures = [
        #    Fracture(4, np.array([]), np.array(), )
        #]
        mesh, fractures, n_large = repository_mesh.fullscale_transport_mesh(cfg.transport_fine, 10)
        assert mesh.path.split('/')[-2:] == ["sandbox", "one_borehole.msh2"]
        assert len(fractures) == 27
        assert n_large == 8
