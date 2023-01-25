import pytest
import os

from endorse import macro_flow_model
from endorse import common
from endorse import homogenisation


script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))

@pytest.mark.skip
def test_homogenisation():
    with common.workdir():
        conf_file = os.path.join(script_dir, "test_data/config_homo_tsx.yaml")
        cfg = common.load_config(conf_file)
        r = 1
        sub_params = [([0,0,0], r),
                      ([2,0,0], r)]
        subdomains = homogenisation.make_subdomains_old(cfg, sub_params)
        homogenisation.subdomains_mesh(subdomains)

#def test_fine_conductivity_field():


#@pytest.mark.skip
def test_macro_transport():
   # with common.workdir("sandbox"):
    #common.EndorseCache.instance().expire_all()
    conf_file = os.path.join(script_dir, "test_data/config.yaml")
    cfg = common.load_config(conf_file)
    macro_flow_model.macro_transport(cfg)
    macro_flow_model.fine_macro_transport(cfg)
