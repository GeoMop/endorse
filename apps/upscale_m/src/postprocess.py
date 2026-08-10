
from typing import Dict, List, Tuple

import numpy as np

from endorse.homogenisation import Subdomain
from endorse.mesh_class import Mesh


def hooke_strain(sigma: np.ndarray, young: float, poisson: float) -> np.ndarray:
    """Isotropic inverse Hooke law eps = ((1+nu) sigma - nu tr(sigma) I) / E, on (N, 3, 3)."""
    trace = np.trace(sigma, axis1=1, axis2=2)
    return ((1.0 + poisson) * sigma - poisson * trace[:, None, None] * np.eye(3)) / young


def strain_stress_fields(output_mesh: Mesh, bulk_region_id: int, fracture_region_ids: List[int],
                         young_rock: float, poisson_rock: float,
                         young_fracture: float, poisson_fracture: float,
                         stress_field: str, region_id_field: str, output_time: float = 0.0
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (region_id, strain, stress) per element of output_mesh, the last two as flat 9-vectors.

    Flow123d does not output strain, so it is reconstructed here from the solved stress -- with
    the compliance matrix of the element's own region, rock in the bulk and the softer fracture
    material in the 2D elements
    """
    region_id = np.asarray(output_mesh.get_p0_values(region_id_field, output_time)).astype(int).ravel()
    stress9 = np.asarray(output_mesh.get_p0_values(stress_field, output_time), dtype=float)
    assert stress9.shape[1] == 9, f"expected 9-component stress, got {stress9.shape}"

    eps9 = np.zeros_like(stress9)
    bulk_rows = region_id == bulk_region_id
    assert bulk_rows.any(), f"No bulk cells with region id {bulk_region_id} in the output."
    eps9[bulk_rows] = hooke_strain(
        stress9[bulk_rows].reshape(-1, 3, 3), young_rock, poisson_rock).reshape(-1, 9)
    if fracture_region_ids:
        frac_rows = np.isin(region_id, fracture_region_ids)
        eps9[frac_rows] = hooke_strain(
            stress9[frac_rows].reshape(-1, 3, 3), young_fracture, poisson_fracture).reshape(-1, 9)
    return region_id, eps9, stress9


def subdomain_measures(sub: Subdomain, mesh: Mesh, region_id_field: np.ndarray,
                       bulk_region_id: int, fracture_region_ids: List[int],
                       aperture_by_region_id: Dict[int, float]
                       ) -> Tuple[Subdomain, np.ndarray]:

    relevant_ids = {bulk_region_id, *fracture_region_ids}
    keep = np.isin(region_id_field[sub.el_indices], list(relevant_ids))
    el_indices = np.asarray(sub.el_indices)[keep]
    intersect_weights = np.asarray(sub.intersect_weights)[keep]
    sub = Subdomain(sub.mesh, sub.macro_el_idx, list(el_indices), list(intersect_weights))

    is_bulk = region_id_field[sub.el_indices] == bulk_region_id
    aperture_scale = np.ones(len(sub.el_indices))
    frac_mask = ~is_bulk
    if frac_mask.any():
        aperture_scale[frac_mask] = [aperture_by_region_id[r]
                                     for r in region_id_field[sub.el_indices][frac_mask]]
    measures = mesh.el_volumes[sub.el_indices] * aperture_scale
    return sub, measures


def weighted_sum(sub: Subdomain, field: np.ndarray, measures: np.ndarray) -> np.ndarray:
    weights = measures * np.asarray(sub.intersect_weights)
    return np.sum(weights[:, None] * np.asarray(field)[sub.el_indices], axis=0)
