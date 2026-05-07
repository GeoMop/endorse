# Test src/repository_mesh.py
# fine EDZ: about 4M elements, 4 minutes

import os
from endorse import common
from endorse.mesh import repository_mesh, mesh_tools, fracture_tools
#from bgem.stochastic.fracture import Fracture


script_dir = os.path.dirname(os.path.realpath(__file__))


def test_make_mesh():
    common.CallCache.instance(expire_all=True)
    # about 280 k elements
    # conf_file = os.path.join(script_dir, "./config_full_coarse.yaml")
    conf_file = os.path.join(script_dir, "../test_data/config.yaml")
    cfg = common.load_config(conf_file)
    with common.workdir("sandbox"):
        #fractures = [
        #    Fracture(4, np.array([]), np.array(), )
        #]
        fr_pop = fracture_tools.population_from_cfg(cfg.transport_fine.fractures.population, cfg.transport_fine.geometry.box_dimensions)
        mesh, fractures, n_large = repository_mesh.fullscale_transport_mesh_3d(cfg.transport_fine, fr_pop, 10)
        assert mesh.path.split('/')[-2:] == ["sandbox", "one_borehole.msh2"]
        assert len(fractures) == 39
        assert n_large == 1
