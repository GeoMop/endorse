"""
Assembly of the effective elastic tensor from the six load-case results.

Method: least-squares identification of the linear map responses = C @ loads via the endorse
equivalent-tensor machinery (`endorse.equivalent_field.eq_tensor`, dimension-generic; used here
with D = 6 on the Voigt vectors). With exactly 6 independent loads the LS solution IS the exact
6-case inversion C = Sigma @ E^-1 (report sec. 2.5.2); more (e.g. mixed) load states extend it
to a genuine LS fit with no code change. BOTH matrices are MEASURED: column q of E and Sigma
holds the volume-averaged strain/stress over the inner cube for load case q. The prescribed load
values are deliberately NOT used as results: with fractures intersecting the outer boundary the
average strain theorem fails (displacement jumps contribute on the boundary), so <eps> over the
averaging volume is the only consistent strain measure. The prescribed formula remains what it
physically is — just the boundary condition.

(The SYMMETRY-constrained fit — 21 unknowns — needs a dim-6 Voigt-pair table in endorse
eq_tensor.tn_homo_kernel_sym, which currently stops at dim 3: upstream question in PLAN.md.)

The report format mirrors R. Siddall's original GenerateKinematicEffectiveElasticTensor output
(sections Sigma, E, C_k, S_k; same prefixes and rulers).
"""
import os
from datetime import datetime
from typing import List

import numpy as np

from endorse.equivalent_field import eq_tensor


def assemble_matrices(results: List["LoadCaseResult"], i_sub: int = 0):
    """
    Column-stack the measured per-case Voigt vectors of subdomain `i_sub`:
    returns (E, Sigma), each 6x6, column q = load case q.
    """
    E = np.column_stack([np.atleast_2d(r.eps_voigt)[i_sub] for r in results])
    Sigma = np.column_stack([np.atleast_2d(r.sigma_voigt)[i_sub] for r in results])
    return E, Sigma


def equivalent_tensor(loads: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """
    Effective tensor C (6x6) from load/response COLUMN matrices (responses = C @ loads),
    via endorse eq_tensor LS (rows of eq_tensor input = load cases, hence the transposes).
    Exactly 6 independent loads -> unique solution = the exact inversion.
    """
    assert loads.shape[0] == 6 and loads.shape == responses.shape, \
        f"need 6-row Voigt column matrices, got {loads.shape} / {responses.shape}"
    eq = eq_tensor(dim=6)  # unconstrained 36-unknown kernel; 'sym' pending upstream (PLAN.md)
    C_flat = eq.flat(loads.T, responses.T)
    return eq.to_full_tn(C_flat).reshape(6, 6)


def _write_matrix(f, title, symbol, M, title_indent=26):
    f.write("=" * 118 + "\n")
    f.write(" " * title_indent + title + "\n")
    f.write("=" * 118 + "\n\n")
    for i in range(6):
        prefix = f"{symbol} =".ljust(8) if i == 2 else " " * 8
        row = " ".join([f"{M[i, j]:>16.6e}" for j in range(6)])
        f.write(f"{prefix} [ {row} ]\n")
    f.write("\n\n")


def write_report(path: str, results, E, Sigma, C, meta: dict,
                 bc_label: str = "kinematic boundary conditions", symbol_suffix: str = "k"):
    """
    Sigma, E (measured), C_<suffix>, S_<suffix>, source VTU list.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    S = np.linalg.inv(C)
    c_sym, s_sym = f"C_{symbol_suffix}", f"S_{symbol_suffix}"
    with open(path, "w") as f:
        f.write(f"upscale_m effective elastic tensor report ({bc_label})\n")
        f.write(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
        for key, value in meta.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        _write_matrix(f, f"Macroscopic stress matrix Sigma for {bc_label}", "Sigma", Sigma)
        _write_matrix(f, f"Macroscopic deformation matrix E for {bc_label}", "E", E)
        _write_matrix(f, f"Effective Elastic Tensor {c_sym} (with dash) for {bc_label}",
                      c_sym, C, title_indent=18)
        _write_matrix(f, f"Compliance Tensor {s_sym} (inverse of {c_sym}) for {bc_label}",
                      s_sym, S, title_indent=18)

        f.write("-" * 118 + "\n")
        f.write("Result computed using the following Flow123d output files:\n")
        for r in results:
            f.write(f"- [{r.name}] {r.spatial_file}\n")
    print(f"[upscale_m] tensor report written: {os.path.abspath(path)}")
