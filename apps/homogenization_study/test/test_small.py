"""Real Flow123d integration cases for the deterministic Stage A conductivity field."""

import pytest

from pathlib import Path
import json
import logging
import numpy as np

from copy import deepcopy
from collections.abc import Callable, Iterator
from time import perf_counter

from bgem.gmsh import gmsh
from bgem.stochastic import Fracture, RectangleShape

from endorse import common
from endorse.mesh import mesh_tools, fracture_tools

import homogenization_study.exception_wrapper as exp
from homogenization_study.create_mesh import (
    fragment, geometry_finalize,
    make_geometry_homo,
    make_mesh
)
from homogenization_study import job
from homogenization_study.model import ConductivityFieldResult, StudyMeshFiles
from homogenization_study.problem import prepare_conductivity_homogenization, run_homogenization_problem


def validate_conductivity_field(result: ConductivityFieldResult) -> np.ndarray:
    """Check positive definiteness of all homogenized conductivity tensors.

    The Voigt order is ``[Kxx, Kyy, Kzz, Kxy, Kxz, Kyz]``. A symmetric
    conductivity tensor should be positive definite for the three pressure loads.
    """
    xx, yy, zz, xy, xz, yz = result.conductivity.T
    tensors = np.empty((result.n_elements, 3, 3), dtype=float)
    tensors[:, 0, 0] = xx
    tensors[:, 1, 1] = yy
    tensors[:, 2, 2] = zz
    tensors[:, 0, 1] = tensors[:, 1, 0] = xy
    tensors[:, 0, 2] = tensors[:, 2, 0] = xz
    tensors[:, 1, 2] = tensors[:, 2, 1] = yz

    eigenvalues = np.linalg.eigvalsh(tensors)
    invalid = np.flatnonzero(np.min(eigenvalues, axis=1) <= 0.0)
    assert invalid.size == 0, (
        "Homogenized conductivity tensors must be positive definite; "
        f"invalid Gmsh elements: {result.gmsh_id[invalid].tolist()}"
    )
    return eigenvalues

@exp.rethrow_as(exp.GeomException, "Geometry exception")
def make_geometry_micro(
    factory: gmsh.GeometryOCC,
    cfg_mesh: common.dotdict,
    fracture_set: list[fracture_tools.RegionFracture],
    include_boundary: bool = True,
) -> gmsh.ObjectSet:
    """Mesh two adjacent half-boxes and an optional clipped vertical fracture."""
    start = perf_counter()
    cfg_geom = cfg_mesh.geometry

    # Prepare objects
    vol_dict = {
        "fractures_group": None,
        "box_A": None,
        "box_B": None
    }
    bnd_dict = {}

    box, box_sides_dict = mesh_tools.box_with_sides(factory, cfg_geom.box_dimensions, cfg_geom.box_center)
    bnd_dict = {**bnd_dict, **box_sides_dict}

    dims = cfg_geom.box_dimensions
    rect_A = factory.rectangle([dims[0] / 2, dims[1]], center=[-dims[0] / 4, 0, -dims[2]/2])
    rect_B = factory.rectangle([dims[0] / 2, dims[1]], center=[ dims[0] / 4, 0, -dims[2]/2])
    box_A = rect_A.extrude(vector=[0, 0, dims[2]])[3]
    box_B = rect_B.extrude(vector=[0, 0, dims[2]])[3]

    vol_dict["box_A"] = box_A.set_region("box_A")
    vol_dict["box_B"] = box_B.set_region("box_B")

    if "fractures" in cfg_geom.include:
        fractures = fracture_tools.create_fractures_rectangles(factory, fracture_set, cfg_geom.box_center, factory.rectangle())
        vol_dict["fractures_group"] = factory.group(*fractures).intersect(box)
        vol_dict["fractures_group"].mesh_step(cfg_mesh.fracture_mesh_step) # .set_region("fractures")

    [print(k, v) for k, v in vol_dict.items()]
    factory.synchronize()
    # factory.show()
    # exit(0)

    # Step 5: Map results back to their labels
    fr_dict, fr_bnd_dict = fragment(factory, vol_dict, bnd_dict)

    [print(k, v) for k, v in fr_dict.items()]
    [print(k, v) for k, v in fr_bnd_dict.items()]

    # include all volumetric fragments
    geometry_set = list(fr_dict.values())

    # get box surface
    if include_boundary:
        for side_name in box_sides_dict.keys():
            fr_bnd_dict[side_name+"_fr"] \
                .set_region('.'+side_name) \
                .mesh_step(cfg_mesh.boundary_mesh_step)
            geometry_set.append(fr_bnd_dict[side_name+"_fr"])

        if "fractures" in cfg_geom.include:
            # fractures and its boundary
            b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
            b_fractures_out = b_fractures_fr \
                .select_by_intersect(box.get_boundary().deepcopy()) \
                .set_region(".fractures_out") \
                .mesh_step(cfg_mesh.boundary_mesh_step)
            geometry_set.append(b_fractures_out)

    geometry = geometry_finalize(factory, geometry_set)
    Path("geometry_timing.json").write_text(json.dumps({"geometry_seconds": perf_counter() - start}))
    return geometry



def run_small_case(
    work_dir: Path, case: str, fracture_angle_degrees: float | None = None,
) -> ConductivityFieldResult:
    """Run one deterministic variant and preserve its conductivity field and diagnostics."""
    if case not in ("uniform", "two_materials", "fractured"):
        raise ValueError(f"Unknown small-test case: {case}")
    test_dir = Path(__file__).resolve().parent
    cfg = common.config.load_config(str(test_dir / "test_small.yaml"))
    cfg.transport_microscale.input_template = str(test_dir / cfg.transport_microscale.input_template)

    if fracture_angle_degrees is not None:
        cfg.transport_microscale.fracture_angle_degrees = fracture_angle_degrees
    fracture_angle_degrees = float(cfg.transport_microscale.fracture_angle_degrees)
    if not np.isfinite(fracture_angle_degrees):
        raise ValueError("Fracture angle must be finite.")

    work_dir = work_dir.resolve()
    if (work_dir / "homogenization_mesh.msh2").exists():
        raise FileExistsError(f"Use a fresh work directory to avoid stale mesh and solver outputs: {work_dir}")

    job.set_workdir(work_dir, input_dir=test_dir)
    with common.workdir(work_dir, clean=False):
        homo_cfg = deepcopy(cfg.mesh)
        homo_cfg.mesh_name = "homogenization_mesh"
        homo_cfg.boundary_mesh_step = cfg.homogenization.boundary_mesh_step
        start = perf_counter()
        homogenization_mesh = make_mesh(homo_cfg, [], 1, make_geometry_homo)
        homo_seconds = perf_counter() - start
        case_cfg = deepcopy(cfg)
        case_cfg.mesh.geometry.include = ["fractures"] if case == "fractured" else []
        if case == "uniform":
            case_cfg.transport_microscale.region_conductivity.box_B = cfg.transport_microscale.region_conductivity.box_A

        fractures = []
        if case == "fractured":
            angle = np.deg2rad(fracture_angle_degrees)
            normal = np.array([np.cos(angle), np.sin(angle), 0.0])
            fracture = Fracture(RectangleShape.id, (200.0, 200.0), np.zeros(3), normal, family=0)
            fractures = [fracture_tools.RegionFracture(fracture, gmsh.Region.get("fr_0"))]

        case_dir = work_dir / case
        with common.workdir(case_dir, clean=False):
            common.config.dump_config(case_cfg, Path("config.yaml"))
            start = perf_counter()
            micro_mesh = make_mesh(case_cfg.mesh, fractures, 1, make_geometry_micro)
            mesh_seconds = perf_counter() - start
            meshes = StudyMeshFiles(Path(micro_mesh.path), Path(homogenization_mesh.path))
            assert meshes.micro_mesh != meshes.homogenization_mesh

            start = perf_counter()
            prepared = prepare_conductivity_homogenization(case_cfg, meshes, fractures=fractures)
            preparation_seconds = perf_counter() - start

            assert prepared.empty_macro_elements.size == 0
            assert prepared.subproblems.n_subdomains == prepared.homogenized_elements.size > 0
            mesh = prepared.micro_mesh
            physical = mesh.gmsh_io.physical
            barycenters = mesh.el_barycenters()
            region_ids = mesh.get_physical_ids()
            volumes = mesh.el_volumes
            for name, sign in (("box_A", -1), ("box_B", 1)):
                selected = region_ids == physical[name][0]
                assert selected.any(), name
                np.testing.assert_allclose(volumes[selected].sum(), 500000.0, rtol=1e-10)
                assert np.all(sign * barycenters[selected, 0] > 0), name
                np.testing.assert_allclose(prepared.micro_fields["conductivity"][selected],
                                           case_cfg.transport_microscale.region_conductivity[name])
            np.testing.assert_allclose(mesh.nodes.min(axis=0), [-50] * 3, atol=1e-8)
            np.testing.assert_allclose(mesh.nodes.max(axis=0), [50] * 3, atol=1e-8)
            bulk = mesh.el_dim_slice(dim=3)
            np.testing.assert_allclose(volumes[bulk].sum(), 1e6)
            assert (prepared.micro_fields["cross_section"][bulk] == 1).all()
            if fractures:
                selected = region_ids == physical["fractures"][0]
                assert selected.any()
                np.testing.assert_allclose(barycenters[selected] @ normal, 0, atol=1e-8)
                np.testing.assert_allclose(prepared.micro_fields["conductivity"][selected],
                                           cfg.transport_microscale.fracture_conductivity)
                np.testing.assert_allclose(prepared.micro_fields["cross_section"][selected],
                                           cfg.transport_microscale.fracture_aperture)
            Path("coverage.json").write_text(json.dumps({
                "macro_elements": int(prepared.homogenized_elements.size),
                "empty_macro_elements": prepared.empty_macro_elements.tolist(),
                "regions": physical,
                }, indent=2))

            start = perf_counter()
            result = run_homogenization_problem(case_cfg, case_dir, meshes, prepared)
            solve_seconds = perf_counter() - start

            eigenvalues = validate_conductivity_field(result)
            summary = {
                "homogenization_mesh_seconds": homo_seconds, "micro_mesh_seconds": mesh_seconds,
                "preparation_seconds": preparation_seconds, "solve_seconds": solve_seconds,
                "minimum_eigenvalue": float(eigenvalues.min()),
                "tensor_min": result.conductivity.min(axis=0).tolist(),
                "tensor_max": result.conductivity.max(axis=0).tolist(),
                "interface_tensor_mean": result.conductivity[
                    np.abs(result.el_barycenter[:, 0]) < 20].mean(axis=0).tolist(),
            }
            Path("summary.json").write_text(json.dumps(summary, indent=2))
            if case == "uniform":
                expected = np.array([1, 1, 1, 0, 0, 0]) * cfg.transport_microscale.region_conductivity.box_A
                np.testing.assert_allclose(result.conductivity, np.broadcast_to(expected, result.conductivity.shape),
                                           rtol=0, atol=expected[0] * cfg.transport_microscale.uniform_relative_tolerance)
            element_indices = prepared.homogenized_elements
            np.testing.assert_array_equal(
                result.gmsh_id, np.asarray(prepared.homogenization_mesh.el_ids)[element_indices])
            np.testing.assert_array_equal(
                result.el_barycenter, prepared.homogenization_mesh.el_barycenters()[element_indices])
    return result


@pytest.fixture
def small_case_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> Iterator[Callable[[str], ConductivityFieldResult]]:
    """Give each selected test fresh outputs and retain logs even when the calculation fails."""
    monkeypatch.setenv("ENDORSE_DISABLE_MEMOIZE", "1")
    caplog.set_level(logging.INFO)
    # Each pytest invocation has a fresh retained workspace, avoiding stale mesh/solver caches.
    def run(case: str) -> ConductivityFieldResult:
        return run_small_case(tmp_path / "small_homogenization", case)

    try:
        yield run
    finally:
        (tmp_path / "run.log").write_text(caplog.text)


def test_uniform(small_case_runner: Callable[[str], ConductivityFieldResult]) -> None:
    """Recover the prescribed isotropic conductivity with equal half-box materials."""
    small_case_runner("uniform")


def test_two_materials(small_case_runner: Callable[[str], ConductivityFieldResult]) -> None:
    """Generate a positive-definite field for the two-material box without fractures."""
    small_case_runner("two_materials")


def test_fractured(small_case_runner: Callable[[str], ConductivityFieldResult]) -> None:
    """Generate a positive-definite field for the two-material box with one angled fracture."""
    small_case_runner("fractured")
