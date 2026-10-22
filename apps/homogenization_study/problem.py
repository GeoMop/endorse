"""Three-load homogenization problem for the conductivity field study."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Any, Sequence

import attrs
import numpy as np

from endorse.apply_fields import fr_fields_repo
from endorse import common
from endorse.common import dotdict, File, memoize
from endorse.homogenisation import Subproblems, validate_subdomain_coverage
from endorse.macro_flow_model import configured_macro_tetra, micro_load_response
from endorse.fullscale_transport import fracture_map
from endorse.mesh_class import Mesh, load_mesh

from .model import ConductivityFieldResult, StudyMeshFiles


@attrs.frozen
class PreparedConductivityHomogenization:
    """Meshes and averaging domains required by the three-load calculation."""

    micro_mesh: Mesh
    homogenization_mesh: Mesh
    homogenized_elements: np.ndarray
    subproblems: Subproblems
    pressure_loads: np.ndarray
    empty_macro_elements: np.ndarray
    micro_fields: dict

@memoize
def merge_fr_regions(mesh: Mesh, fr_set):
    """Merge fracture element tags and expose the shared region as ``fractures``."""
    if not fr_set:
        return mesh, {}

    el_to_ifr = fracture_map(mesh, fr_set, n_large=0, dim=3)
    # rename region
    fracture_region = mesh.gmsh_io.physical.pop("fr_small")
    mesh.gmsh_io.physical["fractures"] = fracture_region

    mesh_modified_filepath = Path(mesh.file.path).stem + "_modified.msh2"
    mesh_modified_file = mesh.write_fields(mesh_modified_filepath)
    mesh.file = mesh_modified_file
    return mesh, el_to_ifr


def set_field_value_on_regions(
        mesh: Mesh,
        field: np.ndarray,
        value: float,
        region_names: Sequence[str],
):
    """Assign given value on the selected mesh regions."""
    if len(field) != len(mesh.elements):
        raise ValueError(f"Field has {len(field)} values for a mesh with {len(mesh.elements)} elements.")

    selected_region_names = set(region_names)

    selected_regions = {
        (region_id, region_dim)
        for physical_name, (region_id, region_dim) in mesh.gmsh_io.physical.items()
        if physical_name in selected_region_names
    }
    selected_elements = np.asarray([
        (element.tags[0], len(element.node_indices) - 1) in selected_regions
        for element in mesh.elements
    ], dtype=bool)
    logging.info(
        "Assigning field value %s to regions %s: matched_regions=%s matched_elements=%s.",
        value,
        list(region_names),
        len(selected_regions),
        int(np.count_nonzero(selected_elements)),
    )
    field[selected_elements] = float(value)


def set_fracture_conductivity(fields:dict, cfg: dotdict, mesh: Mesh, fractures:Sequence[Any] | None, el_to_fr):
    if not fractures:
        return
    fracture_elements = mesh.el_dim_slice(dim=2)
    default_fracture = len(fractures)
    fracture_map_slice = [
        el_to_fr.get(element_index, default_fracture)
        for element_index in range(fracture_elements.start, fracture_elements.stop)
    ]
    fracture_field_params = dotdict.create(
        # relative permeability with respect to ideal case given by cubic law
        {"permeability_factor": 1.0e-2},
    )
    fr_cond, fr_cross, _ = fr_fields_repo(
        cfg["mesh"]["fractures"],
        fracture_field_params,
        mesh.elements[fracture_elements],
        fracture_map_slice,
        fractures,
    )
    fields["conductivity"][fracture_elements] = fr_cond
    fields["cross_section"][fracture_elements] = fr_cross


@memoize
def prepare_conductivity_homogenization(
    cfg: dotdict,
    meshes: StudyMeshFiles,
    fractures: Sequence[Any] | None = None,
) -> PreparedConductivityHomogenization:
    """Load meshes, assign the configured micro conductivity, and construct averaging domains.

    ``homogenization.micro_conductivity`` initializes all micro elements and
    ``homogenization.region_conductivity`` overrides the four material regions.
    """
    pressure_loads = cfg["homogenization"]["pressure_loads"]
    if np.asarray(pressure_loads).shape != (3, 3):
        raise ValueError(f"homogenization.pressure_loads must have shape (3, 3), got {pressure_loads.shape}.")

    for mesh_path in (meshes.micro_mesh, meshes.homogenization_mesh):
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Homogenization mesh file does not exist: {mesh_path}")

    micro_mesh = load_mesh(File(str(meshes.micro_mesh)), heal_tol=None)
    micro_mesh, el_to_fr = merge_fr_regions(micro_mesh, fractures)
    homogenization_mesh = load_mesh(File(str(meshes.homogenization_mesh)), heal_tol=None)
    micro_fields = {"conductivity": np.full(len(micro_mesh.elements), fill_value=0.0, dtype=float),
                    "cross_section": np.full(len(micro_mesh.elements), fill_value=0.0, dtype=float),
                    "region_id": micro_mesh.get_physical_ids()}

    cond_A = 1e-11
    cond_B = 1e-12
    set_field_value_on_regions(micro_mesh, micro_fields["conductivity"], cond_A, ["box_A", "box_thin"])
    set_field_value_on_regions(micro_mesh, micro_fields["conductivity"], cond_B, ["box_B", "box_thick"])
    set_fracture_conductivity(micro_fields, cfg, mesh=micro_mesh, fractures=fractures, el_to_fr=el_to_fr)

    if cfg.debug_output:
        micro_mesh.write_fields_vtu("micro_fields.vtu", micro_fields)

    macro_bulk = homogenization_mesh.el_dim_slice(dim=3)
    homogenized_elements = np.arange(macro_bulk.start, macro_bulk.stop, dtype=int)
    if not homogenized_elements.size:
        raise ValueError(f"Homogenization mesh has no bulk elements: {meshes.homogenization_mesh}")

    subproblems = Subproblems.create(
        homogenization_mesh,
        homogenized_elements,
        micro_mesh,
        configured_macro_tetra(cfg),
        np.ones(3, dtype=int),
    )
    empty_macro_elements = np.empty(0, dtype=int)
    if cfg["homogenization"].get("coverage_preflight", False):
        logging.info(
            "Checking averaging-domain coverage for %s homogenization elements.",
            homogenized_elements.size,
        )
        empty_macro_elements = validate_subdomain_coverage(subproblems)

    logging.info(
        "Prepared conductivity homogenization: micro_elements=%s macro_bulk_elements=%s "
        "empty_macro_elements=%s loads=%s.",
        len(micro_mesh.elements),
        homogenized_elements.size,
        empty_macro_elements.size,
        pressure_loads,
    )
    return PreparedConductivityHomogenization(
        micro_mesh=micro_mesh,
        homogenization_mesh=homogenization_mesh,
        homogenized_elements=homogenized_elements,
        subproblems=subproblems,
        pressure_loads=pressure_loads,
        empty_macro_elements=empty_macro_elements,
        micro_fields=micro_fields,
    )


@memoize
def run_homogenization_problem(
    cfg: dotdict,
    workdir: Path,
    meshes: StudyMeshFiles,
    prepared: PreparedConductivityHomogenization | None = None,
    fractures: Sequence[Any] | None = None,
) -> ConductivityFieldResult:
    """Run three Flow123d loads and return the homogenized tensor field.

    The prepared subproblems are reused for every load.  Flow123d returns
    elementwise load and response fields; their subdomain averages are solved
    together into the six-component Voigt tensor used by the study result.
    """
    if prepared is None:
        prepared = prepare_conductivity_homogenization(cfg, meshes, fractures=fractures)

    loads = prepared.pressure_loads
    averaged_loads = []
    averaged_responses = []
    with common.workdir(workdir, clean=False):
        for load_index, load in enumerate(loads):
            logging.info("Starting homogenization pressure load %s/%s: %s", load_index + 1, len(loads), load)
            load_average, response_average = micro_load_response(
                cfg,
                prepared.subproblems,
                load_index,
                load,
                prepared.micro_fields,
            )
            averaged_loads.append(load_average)
            averaged_responses.append(response_average)

    cond_voigt_tensors = prepared.subproblems.equivalent_tensor_field(
        np.asarray(averaged_loads),
        np.asarray(averaged_responses),
    )

    # Heterogeneous conductiity tensor stored in Voigt notation.
    dflt_cond = -1e-16
    element_indices = prepared.homogenized_elements
    n_elements = len(element_indices)

    if cfg.debug_output:
        conductivity_tn = np.empty((n_elements, 9))
        conductivity_tn[:, :] = np.array([dflt_cond, 0, 0, 0, dflt_cond, 0, 0, 0, dflt_cond])
        voigt_indices = [0, 5, 4, 5, 1, 3, 4, 3, 2]
        conductivity_tn[element_indices[:], :] = cond_voigt_tensors[:, voigt_indices[:]]
        prepared.homogenization_mesh.write_fields_vtu("homogenization.vtu",
                                                      dict(conductivity_tn=conductivity_tn))

    homogenization_mesh = prepared.homogenization_mesh
    result = ConductivityFieldResult(
        conductivity=cond_voigt_tensors,
        el_barycenter=homogenization_mesh.el_barycenters()[element_indices],
        gmsh_id=np.asarray(homogenization_mesh.el_ids, dtype=np.int64)[element_indices],
    )
    logging.info("Completed conductivity homogenization for %s elements.", result.n_elements)
    return result
