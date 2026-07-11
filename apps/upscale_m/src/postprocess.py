"""
Volume integration of stress and strain over the inner (averaging) cube.

Port of R. Siddall's GeneralComputationClass (src_internship_cvut) restricted to the buffered-mesh
regions of micro_mesh.py:

  <sigma> = (1/V) [ sum_bulk sigma_e V_e  +  sum_frac sigma_f A_f delta ]
  <eps>   = (1/V) [ sum_bulk eps_e   V_e  +  sum_frac eps_f   A_f delta ],   eps = M : sigma

Stress is taken directly from the Flow123d VTU cell data; strain is reconstructed per element from
Hooke's law with the analytic isotropic compliance of the respective material (report sec. 3.4.2).
Fracture elements are thin elastic plates weighted by the aperture delta (report sec. 3.1); the
adequacy of this treatment vs the explicit displacement-jump term J is an open question in PLAN.md.

Element selection: bulk = region_id of 'rock_inner' (tag 1); fracture elements = region_id of
'fractures' (tag 3) with barycenter inside the inner cube — exact because the mesh conforms to the
averaging-volume boundary (micro_mesh.py). The reference volume is V = L_inner^3.

Voigt convention (report): sigma = [11, 22, 33, 23, 13, 12]; strain shear rows carry engineering
shear gamma = 2 eps.
"""
from typing import List, Tuple

import numpy as np
import pyvista as pv

REGION_ROCK_INNER = 1
REGION_FRACTURES = 3


def build_compliance_matrix_voigt(E: float, nu: float) -> np.ndarray:
    """
    Isotropic compliance M (6x6), Voigt: eps_v = M @ sigma_v with
    sigma_v = [s11, s22, s33, s23, s13, s12], eps_v = [e11, e22, e33, g23, g13, g12].
    """
    M = np.zeros((6, 6))
    M[0, 0] = M[1, 1] = M[2, 2] = 1.0 / E
    M[0, 1] = M[1, 0] = M[0, 2] = M[2, 0] = M[1, 2] = M[2, 1] = -nu / E
    M[3, 3] = M[4, 4] = M[5, 5] = 2.0 * (1.0 + nu) / E
    return M


def strain_from_stress(sigma_tensors: np.ndarray, M_voigt: np.ndarray) -> np.ndarray:
    """eps = M : sigma elementwise. Input (N,3,3) stress, returns (N,3,3) tensor strain."""
    s = sigma_tensors
    sigma_voigt = np.stack([s[:, 0, 0], s[:, 1, 1], s[:, 2, 2],
                            s[:, 1, 2], s[:, 0, 2], s[:, 0, 1]], axis=1)
    eps_voigt = sigma_voigt @ M_voigt.T
    eps = np.zeros_like(s)
    eps[:, 0, 0] = eps_voigt[:, 0]
    eps[:, 1, 1] = eps_voigt[:, 1]
    eps[:, 2, 2] = eps_voigt[:, 2]
    eps[:, 1, 2] = eps[:, 2, 1] = eps_voigt[:, 3] / 2.0
    eps[:, 0, 2] = eps[:, 2, 0] = eps_voigt[:, 4] / 2.0
    eps[:, 0, 1] = eps[:, 1, 0] = eps_voigt[:, 5] / 2.0
    return eps


def parse_stress_field(raw: np.ndarray) -> np.ndarray:
    """Raw stress data -> (N,3,3). Supports (N,3,3), row-major (N,9) and Voigt (N,6)."""
    raw = np.asarray(raw)
    n = raw.shape[0]
    if raw.ndim == 3 and raw.shape[1:] == (3, 3):
        return raw.astype(float).copy()
    if raw.ndim == 2 and raw.shape[1] == 9:
        return raw.reshape(n, 3, 3).astype(float)
    if raw.ndim == 2 and raw.shape[1] == 6:
        S = np.zeros((n, 3, 3))
        S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = raw[:, 0], raw[:, 1], raw[:, 2]
        S[:, 1, 2] = S[:, 2, 1] = raw[:, 3]
        S[:, 0, 2] = S[:, 2, 0] = raw[:, 4]
        S[:, 0, 1] = S[:, 1, 0] = raw[:, 5]
        return S
    raise ValueError(f"Unsupported stress field shape: {raw.shape}")


def _read_stress(submesh) -> np.ndarray:
    if "stress" in submesh.cell_data:
        raw = submesh.cell_data["stress"]
    elif "stress" in submesh.point_data:
        raw = submesh.point_data_to_cell_data().cell_data["stress"]
    else:
        raise ValueError("No 'stress' field in the VTU output.")
    return parse_stress_field(raw)


def average_boxes(vtu_file: str, boxes: List[Tuple[np.ndarray, np.ndarray]],
                  young_rock: float, poisson_rock: float,
                  young_fracture: float, poisson_fracture: float,
                  aperture: float,
                  region_rock_inner: int = REGION_ROCK_INNER,
                  region_fractures: int = REGION_FRACTURES,
                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Volume-average strain and stress over each axis-aligned box (lo_xyz, hi_xyz) — one
    (eps_avg, sigma_avg) 3x3 pair per box. The VTU is read once; element selection is by
    barycenter, EXACT because the mesh conforms to every box boundary (micro_mesh imprints the
    subdomain cells). Bulk elements must additionally carry region_rock_inner (buffer excluded);
    fracture elements carry region_fractures and are weighted by area * aperture.
    The reference volume of each box is its own geometric volume (macro_conductivity's
    per-subdomain averaging, restricted to conforming = exact weights).
    """
    mesh = pv.read(vtu_file)
    assert "region_id" in mesh.cell_data, "VTU output lacks 'region_id' cell data."
    rid = np.asarray(mesh.cell_data["region_id"]).astype(int).ravel()
    centers = mesh.cell_centers().points

    M_rock = build_compliance_matrix_voigt(young_rock, poisson_rock)
    M_frac = build_compliance_matrix_voigt(young_fracture, poisson_fracture)

    # Overall bounds of the box set: the topmost faces stay INCLUSIVE so nothing on the outer
    # rim of the averaging volume is lost; interior cell interfaces are half-open [lo, hi) so a
    # fracture element lying exactly ON an interface belongs to exactly ONE cell (no double count).
    all_hi = np.max(np.stack([np.asarray(h, dtype=float) for _, h in boxes]), axis=0)

    results = []
    for lo3, hi3 in boxes:
        lo3 = np.asarray(lo3, dtype=float)
        hi3 = np.asarray(hi3, dtype=float)
        tol = 1e-8 * max(1.0, float(np.max(hi3 - lo3)))
        upper_rim = np.isclose(hi3, all_hi, rtol=0.0, atol=tol)
        hi_eff = np.where(upper_rim, hi3 + tol, hi3 - tol)
        in_box = np.all((centers > lo3 - tol) & (centers < hi_eff), axis=1)

        # --- bulk: rock_inner region, barycenter in this box --------------------------------
        bulk_idx = np.where((rid == region_rock_inner) & in_box)[0]
        assert bulk_idx.size, f"No rock_inner cells in subdomain box {lo3} .. {hi3}."
        bulk = mesh.extract_cells(bulk_idx)
        V_e = np.abs(bulk.compute_cell_sizes(length=False, area=False, volume=True)
                     .cell_data["Volume"])
        sigma_e = _read_stress(bulk)
        eps_e = strain_from_stress(sigma_e, M_rock)

        # --- fractures in this box (exact: mesh conforms to the box) ------------------------
        frac_in_idx = np.where((rid == region_fractures) & in_box)[0]
        if frac_in_idx.size:
            frac = mesh.extract_cells(frac_in_idx)
            A_f = np.abs(frac.compute_cell_sizes(length=False, area=True, volume=False)
                         .cell_data["Area"])
            V_f = A_f * aperture
            sigma_f = _read_stress(frac)
            eps_f = strain_from_stress(sigma_f, M_frac)

            sigma_all = np.concatenate((sigma_e, sigma_f), axis=0)
            eps_all = np.concatenate((eps_e, eps_f), axis=0)
            V_all = np.concatenate((V_e, V_f), axis=0)
        else:
            sigma_all, eps_all, V_all = sigma_e, eps_e, V_e

        V_ref = float(np.prod(hi3 - lo3))
        sigma_avg = np.einsum("kij,k->ij", sigma_all, V_all) / V_ref
        eps_avg = np.einsum("kij,k->ij", eps_all, V_all) / V_ref
        results.append((eps_avg, sigma_avg))
    return results


def average_inner_cube(vtu_file: str, inner_box: Tuple[float, float], L_inner: float,
                       young_rock: float, poisson_rock: float,
                       young_fracture: float, poisson_fracture: float,
                       aperture: float,
                       region_rock_inner: int = REGION_ROCK_INNER,
                       region_fractures: int = REGION_FRACTURES,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Volume-average strain and stress over the whole inner cube [lo, hi]^3 (single-box wrapper
    around average_boxes). Returns (eps_avg, sigma_avg) as 3x3.
    """
    lo, hi = inner_box
    box = (np.array([lo, lo, lo]), np.array([hi, hi, hi]))
    return average_boxes(vtu_file, [box], young_rock, poisson_rock,
                         young_fracture, poisson_fracture, aperture,
                         region_rock_inner, region_fractures)[0]


def stress_to_voigt(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[1, 2], T[0, 2], T[0, 1]])


def strain_to_voigt(T: np.ndarray) -> np.ndarray:
    """Engineering shear on the shear rows (gamma = 2 eps)."""
    return np.array([T[0, 0], T[1, 1], T[2, 2],
                     2.0 * T[1, 2], 2.0 * T[0, 2], 2.0 * T[0, 1]])
