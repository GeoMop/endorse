import pytest
import logging
from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path


# Configure the root logger to print INFO+ messages to stderr
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(name)s: %(message)s')


from endorse import macro_flow_model
from endorse import common
from endorse import homogenisation
import pathlib

common.CallCache.instance(verbose=10)

script_dir = pathlib.Path(__file__).parent
test_data_dir = Path(script_dir) / "test_data"
large_model = test_data_dir / "large_model_local.msh2"
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
#@pytest.mark.skipif(not large_model.exists(), reason="requires large_model_local.msh2 fixture")
def test_macro_transport():
   # with common.workdir("sandbox"):
    #common.EndorseCache.instance().expire_all()
    conf_file = script_dir / "input/config.yaml"
    cfg = common.load_config(conf_file)
    macro_flow_model.fine_macro_transport(cfg)
    macro_flow_model.macro_transport(cfg)


"""Regression tests for tetrahedral homogenisation interaction weights."""



@dataclass(frozen=True)
class _ElementAtPoint:
    """Minimal element stand-in exposing the geometry used by ``MacroTetra``."""

    point: np.ndarray

    def barycenter(self) -> np.ndarray:
        """Return the prescribed micro-element barycentre."""
        return self.point


@dataclass(frozen=True)
class _MacroElement:
    """Minimal macro element stand-in exposing tetrahedron vertices."""

    nodes: np.ndarray

    def vertices(self) -> np.ndarray:
        """Return tetrahedron vertices in the same order as a mesh element."""
        return self.nodes


def test_macro_tetra() -> None:
    """Check outside, adaptive-core, and taper-region tetrahedron weights."""
    macro = _MacroElement(np.vstack([np.zeros(3), np.eye(3)]))
    center = np.mean(macro.vertices(), axis=0)
    scaled_vertices = center + 0.75 * (macro.vertices() - center)
    barycentric = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.3, 0.3, 0.3],
    ])
    points = barycentric @ scaled_vertices
    shape = homogenisation.MacroTetra(rel_radius=0.75)

    assert shape.interact(macro, _ElementAtPoint(np.zeros(3))) == 0.0
    interior_weight = shape.interact(macro, _ElementAtPoint(points[2]))
    assert interior_weight == 1.0

    weights = shape.interaction_weights(macro, points)
    np.testing.assert_allclose(weights, [0.0, 1.0, 1.0])

    expanded_shape = homogenisation.MacroTetra(rel_radius=1.25)
    expanded_vertices = center + 1.25 * (macro.vertices() - center)
    taper_point = np.array([0.01, 0.33, 0.33, 0.33]) @ expanded_vertices
    taper_weight = expanded_shape.interact(macro, _ElementAtPoint(taper_point))

    assert 0.0 < taper_weight < 1.0
    np.testing.assert_allclose(taper_weight, 0.2)
