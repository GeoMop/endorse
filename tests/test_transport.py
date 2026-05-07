import pytest
import os
from pathlib import Path
import yaml

#import endorse.macro_flow_model
from endorse import common, flow123d_inputs_path
from endorse.fullscale_transport import transport_run, transport_2d


script_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = Path(script_dir) / "test_data"
large_model = test_data_dir / "large_model_local.msh2"


def test_flow123d_templates():
    template = flow123d_inputs_path.joinpath("transport_fullscale_tmpl.yaml")
    assert os.path.isfile(template)

#@pytest.mark.skip
@pytest.mark.skipif(not large_model.exists(), reason="requires large_model_local.msh2 fixture")
def test_macro_transport():
   # with common.workdir("sandbox"):
    common.CallCache.instance(expire_all=True)
    conf_file = os.path.join(script_dir, "test_data/config.yaml")
    #cfg = common.load_config(conf_file)
    #files = input_files(cfg.transport_fullscale)
    seed = 0
    with common.workdir(f"sandbox/full_transport_{seed}", clean=False):
        # params for single container source
        cfg = common.load_config(conf_file)
        #cfg['transport_fullscale']['end_time'] = 120
        ind_time_max = transport_run(cfg, seed)
        print("Result:", ind_time_max)

@pytest.mark.skip
def test_find_large_fractures():
   # with common.workdir("sandbox"):
    common.CallCache.instance(expire_all=True)
    conf_file = os.path.join(script_dir, "test_data/config.yaml")
    for large_seed in range(0, 1000):
        #cfg = common.load_config(conf_file)
        #files = input_files(cfg.transport_fullscale)
        seed = 19
        with common.workdir(f"sandbox/full_transport_{large_seed}", clean=False):
            # params for single container source
            cfg = common.load_config(conf_file)
            cfg['transport_fullscale']['fractures']['fixed_seed'] = large_seed
            cfg['transport_fullscale']['end_time'] = 120
            try:
                ind_time_max = transport_run(cfg, seed)
            except ValueError:
                continue
            return
            print("Result:", ind_time_max)


@pytest.mark.skip
def test_transport_2d():
   # with common.workdir("sandbox"):
    #common.EndorseCache.instance().expire_all()
    conf_file = os.path.join(script_dir, "test_data/config.yaml")
    seed = 19
    with common.workdir(f"sandbox/transport_2d_{seed}", clean=False):
        # params for single container source
        cfg = common.load_config(conf_file)
        cfg['transport_fullscale']['end_time'] = 150000
        ind_time_max = transport_2d(cfg, seed)
        print("Result:", ind_time_max)

def test_fracture_conductivity():
    common.CallCache.instance(expire_all=True)
    conf_file = test_data_dir / "_fr_Forsmark_repo.yaml"
    with conf_file.open("r", encoding="utf-8") as f:
        cfg_fr = common.dotdict.create(yaml.safe_load(f))

    rho = 998.
    g = 9.81
    visc = 0.001
    r = 10.
    K = []
    for frd in cfg_fr:
        tra = float(frd.tr_a)
        trb = float(frd.tr_b)

        T = tra * r**trb
        delta = (12*T*visc / (rho*g))**(1./3.)

        K.append(T/delta)

    print("K:",K)
    import statistics
    print("mean(K):",statistics.mean(K))

    year = 365.2425*24*3600
    # fig.5 A
    # https://onlinelibrary.wiley.com/doi/epdf/10.1111/gfl.12089
    Kf_tsx = 1 * 0.001 / year
    print("Kf_tsx", Kf_tsx)



if __name__ == "__main__":
    os.chdir(os.path.join(script_dir))
    test_macro_transport()
