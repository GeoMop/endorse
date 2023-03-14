# Test src/repository_mesh.py
# fine EDZ: about 4M elements, 4 minutes

import os
from endorse import common
from endorse.mesh import repository_mesh, mesh_tools
from bgem.stochastic.fracture import Fracture, Population
#from bgem.stochastic.fracture import Fracture
from fixtures import sandbox_fname

script_dir = os.path.dirname(os.path.realpath(__file__))

# TODO test: regiony, počet puklin, jméno sítě, počet element podle dim ??

def test_make_mesh():
    common.EndorseCache.instance().expire_all()
    seed = 10
    conf_file = os.path.join(script_dir, "../test_data/config.yaml")
    cfg = common.load_config(conf_file)
    with common.workdir(sandbox_fname(script_dir)):
        #fractures = [
        #    Fracture(4, np.array([]), np.array(), )
        #]
        box = cfg.geometry.box_dimensions
        #box = [box[0], box[2], 0]
        fr_pop = Population.initialize_3d(cfg.transport_fullscale.fractures.population, box)
        mesh, fractures, n_large = repository_mesh.fullscale_transport_mesh_3d(cfg.transport_fine, fr_pop, 10)
        #mesh, fractures, n_large = repository_mesh.fullscale_transport_mesh_2d(cfg.transport_fine, 10)
        assert mesh.path.split('/')[-2:] == ["sandbox", "one_borehole.msh2"]
        assert len(fractures) == 42
        assert n_large == 2
