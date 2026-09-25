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
    tags: tuple[int, int] = (1, 3)

    def vertices(self) -> np.ndarray:
        """Return tetrahedron vertices in the same order as a mesh element."""
        return self.nodes

    def barycenter(self) -> np.ndarray:
        """Return the tetrahedron centroid."""
        return np.mean(self.nodes, axis=0)

    def volume(self) -> float:
        """Return the signed tetrahedron volume."""
        return float(np.linalg.det((self.nodes[1:] - self.nodes[0]).T) / 6.0)


@dataclass
class _Mesh:
    """Small mesh stand-in supporting subdomain selection and diagnostics."""

    elements: list[_MacroElement]
    el_ids: list[int]

    def candidate_indices(self, aabb: np.ndarray) -> list[int]:
        """Return synthetic elements whose barycentres lie inside the AABB."""
        barycenters = self.el_barycenters()
        inside = np.all((aabb[0] <= barycenters) & (barycenters <= aabb[1]), axis=1)
        return np.flatnonzero(inside).tolist()

    def el_dim_slice(self, dim: int) -> slice:
        """Expose all synthetic tetrahedra as bulk elements."""
        assert dim == 3
        return slice(0, len(self.elements))

    def el_barycenters(self) -> np.ndarray:
        """Return synthetic element barycentres."""
        return np.asarray([element.barycenter() for element in self.elements])

    @property
    def el_volumes(self) -> np.ndarray:
        """Return synthetic element volumes."""
        return np.asarray([element.volume() for element in self.elements])


@dataclass(frozen=True)
class _Subproblem:
    """Subproblem stand-in whose input submesh is already available."""

    macro_mesh: _Mesh
    macro_el_shape: homogenisation.MacroTetra
    macro_elements: np.ndarray
    submesh: _Mesh


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


def test_macro_tetra_aabb_uses_scaled_vertices() -> None:
    """Scale the candidate-search AABB consistently with the interaction tetrahedron."""
    macro = _MacroElement(np.vstack([np.zeros(3), np.eye(3)]))
    center = macro.barycenter()
    shape = homogenisation.MacroTetra(rel_radius=1.5)
    scaled_vertices = center + shape.rel_radius * (macro.vertices() - center)

    np.testing.assert_allclose(
        shape.aabb(macro),
        [np.min(scaled_vertices, axis=0), np.max(scaled_vertices, axis=0)],
    )


def test_subdomain_selection_uses_scaled_macro_tetra_aabb() -> None:
    """Include a micro barycentre outside the original AABB but inside the scaled tetrahedron."""
    macro = _MacroElement(np.vstack([np.zeros(3), np.eye(3)]))
    macro_mesh = _Mesh([macro], [10])
    micro_center = np.asarray([-0.05, 0.25, 0.25])
    offsets = 0.02 * np.vstack([np.zeros(3), np.eye(3)])
    micro_element = _MacroElement(micro_center + offsets - np.mean(offsets, axis=0))
    micro_mesh = _Mesh([micro_element], [20])

    selection = homogenisation.Subdomain.select(
        homogenisation.MacroTetra(rel_radius=1.5), micro_mesh, macro_mesh, 0
    )

    np.testing.assert_array_equal(selection.candidate_indices, [0])
    np.testing.assert_array_equal(selection.element_indices, [0])


def _coverage_case(micro_nodes: np.ndarray, rel_radius: float = 1.0) -> homogenisation.Subproblems:
    """Construct one macro element and one candidate micro element."""
    macro_mesh = _Mesh([_MacroElement(np.vstack([np.zeros(3), np.eye(3)]))], [10])
    micro_mesh = _Mesh([_MacroElement(micro_nodes)], [20])
    shape = homogenisation.MacroTetra(rel_radius=rel_radius)
    subproblem = _Subproblem(macro_mesh, shape, np.asarray([0]), micro_mesh)
    return homogenisation.Subproblems(macro_mesh, np.asarray([0]), [subproblem])


def test_validate_subdomain_coverage_accepts_selected_micro_element(caplog) -> None:
    """Accept a macro element containing a candidate micro-element barycentre."""
    center = np.full(3, 0.25)
    offsets = 0.02 * np.vstack([np.zeros(3), np.eye(3)])
    subproblems = _coverage_case(center + offsets - np.mean(offsets, axis=0))

    with caplog.at_level(logging.INFO):
        homogenisation.validate_subdomain_coverage(subproblems)

    assert "macro_elements=1 empty=0" in caplog.text


def test_validate_subdomain_coverage_reports_geometric_overlap(caplog) -> None:
    """Report a candidate containing the macro centroid when its own barycentre lies outside."""
    macro_center = np.full(3, 0.25)
    micro_nodes = np.vstack([macro_center, 2.0 * np.eye(3)])
    subproblems = _coverage_case(micro_nodes, rel_radius=1.5)

    with caplog.at_level(logging.ERROR), pytest.raises(
            homogenisation.SubdomainCoverageError,
            match=r"1 empty macro elements: \[0\]",
    ):
        homogenisation.validate_subdomain_coverage(subproblems)

    assert "bulk_candidates=1" in caplog.text
    assert "gmsh_id=20" in caplog.text
    assert "macro_element_scale=1.5" in caplog.text
    assert "required_macro_element_scale=3.75" in caplog.text
    assert "macro_center_containers=[0]" in caplog.text


def test_macro_conductivity_runs_enabled_coverage_preflight(monkeypatch) -> None:
    """Run the coverage preflight before dispatching a microscale load."""
    sentinel = object()
    monkeypatch.setenv("ENDORSE_DISABLE_MEMOIZE", "1")

    def fail_preflight(subproblems) -> None:
        assert subproblems is sentinel
        raise homogenisation.SubdomainCoverageError("preflight called")

    def create_subproblems(_macro_mesh, _homogenized_els, _micro_mesh, macro_shape, _subdivision):
        assert macro_shape.rel_radius == 2.25
        return sentinel

    monkeypatch.setattr(macro_flow_model.Subproblems, "create", create_subproblems)
    monkeypatch.setattr(macro_flow_model, "validate_subdomain_coverage", fail_preflight)
    cfg = common.dotdict.create({
        "homogenization": {
            "coverage_preflight": True,
            "macro_element_scale": 2.25,
        },
    })

    with pytest.raises(homogenisation.SubdomainCoverageError, match="preflight called"):
        macro_flow_model.macro_conductivity(cfg, None, None, [], {})
