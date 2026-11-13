import pytest
import logging
from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path
from types import SimpleNamespace


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

    with caplog.at_level(logging.WARNING):
        empty_indices = homogenisation.validate_subdomain_coverage(subproblems)

    np.testing.assert_array_equal(empty_indices, [0])
    assert "bulk_candidates=1" in caplog.text
    assert "gmsh_id=20" in caplog.text
    assert "macro_element_scale=1.5" in caplog.text
    assert "required_macro_element_scale=3.75" in caplog.text
    assert "macro_center_containers=[0]" in caplog.text
    assert "their values will be interpolated" in caplog.text


def test_subdomain_create_retains_empty_selection(monkeypatch) -> None:
    """Represent an empty averaging domain instead of stopping homogenization."""
    macro_mesh = _Mesh([_MacroElement(np.vstack([np.zeros(3), np.eye(3)]))], [10])
    micro_mesh = _Mesh([_MacroElement(10.0 + np.vstack([np.zeros(3), np.eye(3)]))], [20])
    empty_selection = homogenisation.SubdomainSelection(
        candidate_indices=np.asarray([], dtype=int),
        element_indices=np.asarray([], dtype=int),
        interaction_weights=np.asarray([], dtype=float),
    )
    monkeypatch.setattr(homogenisation.Subdomain, "select", lambda *_args: empty_selection)

    subdomain = homogenisation.Subdomain.create(
        homogenisation.MacroTetra(rel_radius=1.0),
        micro_mesh,
        macro_mesh,
        0,
        np.ones(len(micro_mesh.elements)),
    )

    assert subdomain.is_empty
    assert subdomain.el_indices == []


def test_assembly_average_matrix_skips_empty_subdomains() -> None:
    """Keep empty rows in the matrix while assembling weights only for supported rows."""
    element = _MacroElement(np.vstack([np.zeros(3), np.eye(3)]))
    mesh = _Mesh([element, element], [20, 21])
    subproblem = homogenisation.SubMeshSubproblem(
        macro_mesh=mesh,
        micro_mesh=mesh,
        macro_el_shape=homogenisation.MacroTetra(rel_radius=1.0),
        aabb=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        macro_elements=np.asarray([0, 1]),
        i_subdomains=np.asarray([0, 1]),
        cross_section=np.ones(2),
    )
    subproblem._subdomains_mesh = mesh
    subproblem._subdomains = [
        homogenisation.Subdomain(mesh, 0, [0, 1], [1.0, 1.0], np.ones(2)),
        homogenisation.Subdomain(mesh, 1, [], [], np.ones(2)),
    ]

    average_matrix = subproblem.assembly_average_matrix(mesh)

    np.testing.assert_allclose(average_matrix.toarray(), [[0.5, 0.5], [0.0, 0.0]])
    np.testing.assert_array_equal(subproblem.empty_subdomain_indices(mesh), [1])


def test_assembly_average_matrix_caches_cross_section_weights() -> None:
    """Apply cross sections once and reuse both element weights and the sparse matrix."""
    element = _MacroElement(np.vstack([np.zeros(3), np.eye(3)]))
    mesh = _Mesh([element, element, element], [20, 21, 22])
    cross_section = np.asarray([1.0, 3.0, 0.0])
    subproblem = homogenisation.SubMeshSubproblem(
        macro_mesh=mesh,
        micro_mesh=mesh,
        macro_el_shape=homogenisation.MacroTetra(rel_radius=1.0),
        aabb=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        macro_elements=np.asarray([0]),
        i_subdomains=np.asarray([0]),
        cross_section=cross_section,
    )
    subdomain = homogenisation.Subdomain(
        mesh,
        0,
        [0, 1, 2],
        [1.0, 1.0, 1.0],
        cross_section,
    )
    subproblem._subdomains_mesh = mesh
    subproblem._subdomains = [subdomain]

    weights = subdomain.weights
    average_matrix = subproblem.assembly_average_matrix(mesh)

    np.testing.assert_allclose(weights, [0.25, 0.75, 0.0])
    np.testing.assert_allclose(average_matrix.toarray(), [[0.25, 0.75, 0.0]])
    assert subdomain.weights is weights
    assert subproblem.assembly_average_matrix(mesh) is average_matrix


def test_assembly_average_matrix_accepts_all_empty_subdomains() -> None:
    """Build a correctly shaped sparse matrix when every averaging domain is empty."""
    element = _MacroElement(np.vstack([np.zeros(3), np.eye(3)]))
    mesh = _Mesh([element], [20])
    subproblem = homogenisation.SubMeshSubproblem(
        macro_mesh=mesh,
        micro_mesh=mesh,
        macro_el_shape=homogenisation.MacroTetra(rel_radius=1.0),
        aabb=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        macro_elements=np.asarray([0]),
        i_subdomains=np.asarray([0]),
        cross_section=np.ones(1),
    )
    subproblem._subdomains_mesh = mesh
    subproblem._subdomains = [homogenisation.Subdomain(mesh, 0, [], [], np.ones(1))]

    average_matrix = subproblem.assembly_average_matrix(mesh)

    assert average_matrix.shape == (1, 1)
    assert average_matrix.nnz == 0


def test_micro_postprocess_interpolates_empty_subdomains(monkeypatch) -> None:
    """Average supported rows and nearest-interpolate scalar and vector values for empty rows."""
    offsets = 0.02 * np.vstack([np.zeros(3), np.eye(3)])
    micro_centers = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    micro_elements = [
        _MacroElement(center + offsets - np.mean(offsets, axis=0))
        for center in micro_centers
    ]

    class _OutputMesh(_Mesh):
        def get_static_p0_values(self, field_name: str) -> np.ndarray:
            assert field_name == "response"
            return np.asarray([10.0, 20.0])

    output_mesh = _OutputMesh(micro_elements, [20, 21])
    macro_centers = np.asarray([[5.0, 0.0, 0.0], [9.9, 0.0, 0.0]])
    macro_mesh = _Mesh([
        _MacroElement(center + offsets - np.mean(offsets, axis=0))
        for center in macro_centers
    ], [10, 11])

    class _AveragingSubproblem:
        macro_elements = np.asarray([0, 1])

        def __init__(self) -> None:
            self.macro_mesh = macro_mesh

        def assembly_average_matrix(self, _output_mesh: _OutputMesh) -> np.ndarray:
            return np.asarray([[0.5, 0.5], [0.0, 0.0]])

        def empty_subdomain_indices(self, _output_mesh: _OutputMesh) -> np.ndarray:
            return np.asarray([1])

    load_values = np.asarray([[2.0, 0.0, 0.0], [4.0, 2.0, 0.0]])
    monkeypatch.setattr(macro_flow_model, "load_mesh", lambda _path: output_mesh)
    monkeypatch.setattr(
        macro_flow_model,
        "get_load_data",
        lambda _cfg, _mesh, _response: load_values,
    )
    monkeypatch.setattr(
        macro_flow_model,
        "_interpolate_empty_subdomains",
        lambda _subproblem, _mesh, _empty, values: np.asarray(values)[[1]],
    )
    micro_model = SimpleNamespace(hydro=SimpleNamespace(spatial_file="flow_fields.msh"))

    load_averages, response_averages = macro_flow_model.micro_postprocess(
        SimpleNamespace(response_field_p0="response"),
        _AveragingSubproblem(),
        micro_model,
    )

    np.testing.assert_allclose(load_averages, [[3.0, 1.0, 0.0], [4.0, 2.0, 0.0]])
    np.testing.assert_allclose(response_averages, [15.0, 20.0])


def test_macro_conductivity_runs_enabled_coverage_preflight(monkeypatch) -> None:
    """Run the coverage preflight before dispatching a microscale load."""
    sentinel = object()
    monkeypatch.setenv("ENDORSE_DISABLE_MEMOIZE", "1")

    def fail_preflight(subproblems) -> None:
        assert subproblems is sentinel
        raise homogenisation.SubdomainCoverageError("preflight called")

    def create_subproblems(
            _macro_mesh,
            _homogenized_els,
            _micro_mesh,
            macro_shape,
            _subdivision,
            _cross_section,
    ):
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
        macro_flow_model.macro_conductivity(
            cfg,
            None,
            None,
            [],
            {"cross_section": np.ones(1)},
        )


@pytest.mark.parametrize("batched", [False, True])
def test_voigt_maps_and_converters(batched: bool) -> None:
    """Check both maps and converters against independently numbered tensors."""
    vectors = np.array([[2., 3., 4., .4, .3, .2], [5., 6., 7., -.8, .7, -.6]])
    tensors = np.array([
        [[2., .2, .3], [.2, 3., .4], [.3, .4, 4.]],
        [[5., -.6, .7], [-.6, 6., -.8], [.7, -.8, 7.]],
    ])
    np.testing.assert_array_equal(homogenisation.VOIGT_TO_TENSOR, [0, 5, 4, 5, 1, 3, 4, 3, 2])
    np.testing.assert_array_equal(homogenisation.TENSOR_TO_VOIGT, [0, 4, 8, 5, 2, 1])
    if not batched:
        vectors, tensors = vectors[0], tensors[0]
    expanded = vectors[..., homogenisation.VOIGT_TO_TENSOR].reshape(tensors.shape)
    flattened = tensors.reshape(vectors.shape[:-1] + (9,))
    np.testing.assert_array_equal(expanded, tensors)
    np.testing.assert_array_equal(flattened[..., homogenisation.TENSOR_TO_VOIGT], vectors)
    np.testing.assert_array_equal(homogenisation.voigt_to_tensor(vectors), tensors)
    np.testing.assert_array_equal(homogenisation.tensor_to_voigt(tensors), vectors)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("n_elements", [1, 2])
@pytest.mark.parametrize("load_rows", [
    np.eye(3),
    np.array([[1., 2., 0.], [0., 1., 3.], [2., 0., 1.]]),
    np.array([[1., 2., 0.], [0., 1., 3.], [2., 0., 1.], [-1., 2., 1.]]),
], ids=["coordinate", "mixed", "four_loads"])
def test_tensor_recovery_preserves_component_numbering(n_elements: int, load_rows: np.ndarray, strided: bool) -> None:
    """Recover literal Voigt components from exact matrix-vector responses."""
    tensors = np.array([
        [[2., .2, .3], [.2, 3., .4], [.3, .4, 4.]],
        [[5., -.6, .7], [-.6, 6., -.8], [.7, -.8, 7.]],
    ])[:n_elements]
    expected = np.array([[2., 3., 4., .4, .3, .2], [5., 6., 7., -.8, .7, -.6]])[:n_elements]

    loads = load_rows.T                            # (3, n_loads): columns are load vectors
    responses = tensors @ loads[None, :, :]        # (n_elements, 3, n_loads)
    load_field = np.stack([load_rows] * n_elements, axis=1)
    response_field = responses.transpose(2, 0, 1)  # (n_loads, n_elements, 3)
    subproblems = homogenisation.Subproblems(
        None, np.arange(n_elements), [SimpleNamespace(i_subdomains=list(range(n_elements)))])
    if strided:
        load_storage = np.zeros(load_field.shape[:-1] + (6,))
        response_storage = np.zeros(response_field.shape[:-1] + (6,))
        load_storage[..., ::2] = load_field
        response_storage[..., ::2] = response_field
        load_field, response_field = load_storage[..., ::2], response_storage[..., ::2]
        assert not load_field.flags.c_contiguous
    actual = subproblems.equivalent_tensor_field(load_field, response_field)

    assert actual.shape == (n_elements, 6)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_voigt_two_batch_axes() -> None:
    """Preserve literal tensor components through two leading batch dimensions."""
    vectors = np.broadcast_to([2., 3., 4., .4, .3, .2], (2, 3, 6))
    tensors = np.broadcast_to([[2., .2, .3], [.2, 3., .4], [.3, .4, 4.]], (2, 3, 3, 3))
    np.testing.assert_array_equal(homogenisation.voigt_to_tensor(vectors), tensors)
    np.testing.assert_array_equal(homogenisation.tensor_to_voigt(tensors), vectors)
    assert not homogenisation.VOIGT_TO_TENSOR.flags.writeable
    assert not homogenisation.TENSOR_TO_VOIGT.flags.writeable


def test_tensor_recovery_clamping() -> None:
    """Keep the existing response-scaled eigenvalue floor for indefinite fits.
        K = diag(-2, 3, 4) -> replace negative eig -2 to eps
    """
    loads = np.eye(3)[:, None, :]
    responses = np.diag([-2., 3., 4.])[:, None, :]
    subproblems = homogenisation.Subproblems(None, np.array([42]), [SimpleNamespace(i_subdomains=[0])])
    actual = subproblems.equivalent_tensor_field(loads, responses)
    np.testing.assert_allclose(actual, [[3e-6, 3., 4., 0., 0., 0.]], atol=1e-12)


def test_tensor_recovery_poor_fit(caplog) -> None:
    """Identify only the element whose excess load is inconsistent with the fitted tensor."""
    loads = np.stack([np.vstack([np.eye(3), [1., 1., 1.]])] * 2, axis=1)
    responses = 2 * loads
    responses[-1, 1, 0] += 100.
    subproblems = homogenisation.Subproblems(None, np.array([42, 71]), [SimpleNamespace(i_subdomains=[0, 1])])
    with caplog.at_level(logging.WARNING):
        subproblems.equivalent_tensor_field(loads, responses)
    assert "element_index=1 macro_element=71" in caplog.text
    assert "element_index=0" not in caplog.text


def test_voigt_macro_export(monkeypatch, tmp_path) -> None:
    """Inspect anisotropic nine-component fields passed to both macro exporters."""
    monkeypatch.setenv("ENDORSE_DISABLE_MEMOIZE", "1")
    captured = []
    (tmp_path / "fields.msh").touch()
    mesh = SimpleNamespace(
        elements=[None, None],
        write_fields=lambda path, fields: captured.append(fields["conductivity_tn"].copy()),
        write_fields_vtu=lambda path, fields: captured.append(fields["conductivity_tn"].copy()),
    )
    subproblems = SimpleNamespace(equivalent_tensor_field=lambda *_: np.array([[2., 3., 4., .4, .3, .2]]))
    monkeypatch.setattr(macro_flow_model.Subproblems, "create", lambda *_: subproblems)
    monkeypatch.setattr(macro_flow_model, "micro_load_response", lambda *_: (None, None))
    cfg = common.dotdict.create({
        "homogenization": {"macro_element_scale": 1.},
        "transport_microscale": {"pressure_loads": [[1., 0., 0.]]},
        "transport_macroscale": {"default_conductivity": 9., "input_fields_file": str(tmp_path / "fields.msh")},
    })
    macro_flow_model.macro_conductivity(cfg, None, mesh, [1], {"cross_section": np.ones(1)})
    assert len(captured) == 2
    for values in captured:
        np.testing.assert_array_equal(values, [[9., 0., 0., 0., 9., 0., 0., 0., 9.],
                                               [2., .2, .3, .2, 3., .4, .3, .4, 4.]])


@pytest.mark.parametrize("field_name", ["load", "response"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_tensor_recovery_reports_nonfinite_input(
    field_name: str, bad_value: float, caplog, monkeypatch
) -> None:
    """Report invalid field coordinates and stop before entering SVD."""
    loads = np.stack([np.eye(3)] * 2, axis=1)
    responses = loads.copy()
    fields = {"load": loads, "response": responses}
    fields[field_name][1, 1, 2] = bad_value
    subproblems = homogenisation.Subproblems(
        None, np.array([42, 71]), [SimpleNamespace(i_subdomains=[0, 1])])

    def unexpected_svd(*args, **kwargs):
        pytest.fail("Non-finite input reached SVD")

    monkeypatch.setattr(np.linalg, "svd", unexpected_svd)
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="Non-finite tensor recovery input"):
        subproblems.equivalent_tensor_field(loads, responses)
    assert f"field={field_name} count=1" in caplog.text
    assert "load_index=1 element_index=1 macro_element=71 component=z" in caplog.text
