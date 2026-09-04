from endorse import common
import os
import numpy as np
import sys
from pathlib import Path
from scipy import stats
import pytest

script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))


def test_workdir():
    pass



def test_substitute_placeholders():
    pass


def test_check_conv_reasons():
    pass


def test_call_flow_accepts_direct_and_resolved_machine_config(tmp_path):
    """
    Check that `call_flow` accepts both direct flow config and host-map config.
    """
    flow_mock = Path(__file__).with_name("flow_mock.py")
    flow_executable = [sys.executable, str(flow_mock)]
    template_path = tmp_path / "template_tmpl.yaml"
    template_path.write_text("template\n", encoding="utf-8")
    template = str(template_path)

    direct_cfg = common.dotdict.create({"flow_executable": flow_executable})
    resolved_cfg = common.dotdict.create({
        "__default__": {"flow_executable": ["unused-default"]},
        "__resolved__": {"flow_executable": flow_executable},
    })

    direct_workdir = tmp_path / "call_flow_direct"
    resolved_workdir = tmp_path / "call_flow_resolved"

    with common.workdir(direct_workdir, clean=False):
        direct_result = common.call_flow(direct_cfg, template, {})

    with common.workdir(resolved_workdir, clean=False):
        resolved_result = common.call_flow(resolved_cfg, template, {})

    assert direct_result.process.returncode == 0
    assert resolved_result.process.returncode == 0
    assert direct_result.check_conv_reasons()
    assert resolved_result.check_conv_reasons()
    assert direct_result.stdout != resolved_result.stdout
    assert Path(direct_result.stdout).read_text(encoding="utf-8").startswith(
        f"flow_mock input={direct_workdir / 'template.yaml'}"
    )
    assert Path(resolved_result.stdout).read_text(encoding="utf-8").startswith(
        f"flow_mock input={resolved_workdir / 'template.yaml'}"
    )
    assert (direct_workdir / "output" / "flow123.0.log").read_text(encoding="utf-8") == "convergence reason 0,\n"
    assert (resolved_workdir / "output" / "flow123.0.log").read_text(encoding="utf-8") == "convergence reason 0,\n"


def test_sample_from_population():
    population = np.array([(i, i*i) for i in [1,2,3,4]])
    frequencies = [10, 3, 20, 4]
    i_samples = common.sample_from_population(10000, frequencies)
    samples = population[i_samples, ...]
    sampled_freq = 4 * [0]
    for i, ii in samples:
        sampled_freq[i-1] += 1
    chisq, pval = stats.chisquare(sampled_freq, np.array(frequencies) / np.sum(frequencies) * len(samples))
    print("\nChi square test pval: ", pval)
    assert pval > 0.05
