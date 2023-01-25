# Test src/container_position_mesh.py
# TODO: rename as the container position need not to be related to the container spacing
# fine EDZ (0.3): about  300k elements, 10 s

import pytest
import os

from endorse import common
from endorse.mesh import container_position_mesh

#from bgem.stochastic.fracture import Fracture


script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))

@pytest.mark.skip
def test_fine_micro_mesh():
    # about 280 k elements
    # conf_file = os.path.join(script_dir, "./config_full_coarse.yaml")
    with common.workdir("sandbox"):
        conf_file = os.path.join(script_dir, "./config_full_edz_fine.yaml")
        cfg = common.config.load_config(conf_file)
        #fractures = [
        #    Fracture(4, np.array([]), np.array(), )
        #]
        fractures = []
        container_position_mesh.fine_micro_mesh(cfg.geometry, fractures, 0, "test_container_position_mesh.msh")


@pytest.mark.skip
def test_coarse_micro_mesh():
    # about 280 k elements
    # conf_file = os.path.join(script_dir, "./config_full_coarse.yaml")
    conf_file = os.path.join(script_dir, "./config_full_edz_fine.yaml")
    cfg = common.config.load_config(conf_file)
    #fractures = [
    #    Fracture(4, np.array([]), np.array(), )
    #]
    fractures = []
    macro_step = 2
    container_position_mesh.coarse_micro_mesh(cfg.geometry, macro_step, fractures, 0, "test_coarse_container.msh")
