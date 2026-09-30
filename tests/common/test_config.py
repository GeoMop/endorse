from endorse import common
import os
from pathlib import Path
import numpy as np
from scipy import stats
import pytest

script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))




def test_dotdict():
    pass

def test_apply_variant():
    cfg = dict(
        a = [0, 1],
        b = dict(
            a='a',
            b='b'
        )
    )
    cfg = common.dotdict(cfg)
    variant = {
        'a/1' : 2,
        'b/b' : 'c'
    }
    new_cfg = common.apply_variant(cfg, variant)
    assert new_cfg['a'][0] == 0
    assert new_cfg['a'][1] == 2
    assert new_cfg['b']['a'] == 'a'
    assert new_cfg['b']['b'] == 'c'

    # test errors

    # missing key in dict
    with pytest.raises(KeyError):
        common.apply_variant(cfg, {'c':0})
    # integer key for dict
    with pytest.raises(KeyError):
        common.apply_variant(cfg, {'0':0})
    # empty path
    with pytest.raises(KeyError):
        common.apply_variant(cfg, {'':0})
    # missing index of list
    with pytest.raises(IndexError):
        common.apply_variant(cfg, {'a/2':0})
    # indexing list by key
    with pytest.raises(IndexError):
        common.apply_variant(cfg, {'a/b':0})

def test_load_config():
    conf_file = os.path.join(script_dir, "cfg_main.yaml")
    cfg = common.load_config(conf_file, collect_files=True)

    assert cfg.a.other.content == "inner empty"
    assert cfg.c[0] == 'other_file1.any'
    assert cfg.c[1].c == 'other_file2.any'
    print(cfg._file_refs)
    assert len(cfg._file_refs) == 5 #other_file1, other_file2, _cfg_a, cfg_b, cfg_main
    for f in ['other_file1.any', 'other_file2.any', '_cfg_a.yaml', '_cfg_b.yaml']:
        assert Path(script_dir) / f in cfg._file_refs


def test_dump_config_reloads_machine_config_with_resolution(tmp_path):
    """
    Verify that dump/load keeps the host map and exposes the resolved host config.
    """
    cfg = common.dotdict.create({
        "machine_config": {
            "__default__": {
                "flow_executable": ["default-flow123d"],
            },
            "cluster": {
                "flow_executable": ["cluster-flow123d"],
                "pbs": {
                    "queue": "debug",
                },
            },
        },
        "mlmc": {
            "sim_class": "RandomTransportSimulation",
            "levels": [
                {"params": {"mesh_step": 10.0}},
            ],
        },
    })
    cfg_path = tmp_path / "dumped_config.yaml"

    common.dump_config(cfg, cfg_path)
    dumped_text = cfg_path.read_text(encoding="utf-8")
    loaded_cfg = common.load_config(cfg_path, hostname="cluster")
    reloaded_path = tmp_path / "reloaded_config.yaml"

    common.dump_config(loaded_cfg, reloaded_path)
    reloaded_text = reloaded_path.read_text(encoding="utf-8")
    reloaded_cfg = common.load_config(reloaded_path, hostname="cluster")

    assert "python/object" not in dumped_text
    assert "python/object" not in reloaded_text
    assert loaded_cfg.machine_config.cluster.flow_executable == ["cluster-flow123d"]
    assert loaded_cfg.machine_config.__resolved__.flow_executable == ["cluster-flow123d"]
    assert reloaded_cfg.machine_config.__default__.flow_executable == ["default-flow123d"]
    assert reloaded_cfg.machine_config.cluster.pbs.queue == "debug"
    assert reloaded_cfg.machine_config.__resolved__.flow_executable == ["cluster-flow123d"]
    assert loaded_cfg.mlmc.sim_class == "RandomTransportSimulation"
    assert loaded_cfg.mlmc.levels[0].params.mesh_step == 10.0
