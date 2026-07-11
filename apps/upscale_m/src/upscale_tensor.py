"""
Assembly of the effective elastic tensor from the six load-case results.

Method: exact 6-case inversion C = Sigma @ E^-1 (report sec. 2.5.2). BOTH matrices are MEASURED:
column q of E and Sigma holds the volume-averaged strain/stress over the inner cube for load case
q. The prescribed load values are deliberately NOT used as results: with fractures intersecting
the outer boundary the average strain theorem fails (displacement jumps contribute on the
boundary), so <eps> over the averaging volume is the only consistent strain measure. The
prescribed formula remains what it physically is — just the boundary condition.

The report format mirrors R. Siddall's original GenerateKinematicEffectiveElasticTensor output
(sections Sigma, E, C_k, S_k; same prefixes and rulers).

The symmetric least-squares fit (21 unknowns, 6x6+ equations) will replace method='exact' later;
the interface already accepts general (loads, responses) column matrices.
"""
import os
from datetime import datetime
from typing import List

import numpy as np


def assemble_matrices(results: List["LoadCaseResult"], i_sub: int = 0):
    """
    Column-stack the measured per-case Voigt vectors of subdomain `i_sub`:
    returns (E, Sigma), each 6x6, column q = load case q.
    """
    E = np.column_stack([np.atleast_2d(r.eps_voigt)[i_sub] for r in results])
    Sigma = np.column_stack([np.atleast_2d(r.sigma_voigt)[i_sub] for r in results])
    return E, Sigma


def equivalent_tensor(loads: np.ndarray, responses: np.ndarray, method: str = "exact") -> np.ndarray:
    """
    Effective tensor C from load/response column matrices (responses = C @ loads).
    method 'exact': C = responses @ loads^-1 (requires exactly 6 independent loads).
    method 'ls_symmetric': symmetric least-squares fit — pending
    """
    if method == "exact":
        assert loads.shape == (6, 6), f"exact inversion needs 6x6 loads, got {loads.shape}"
        return responses @ np.linalg.inv(loads)
    elif method == "ls_symmetric":
        raise NotImplementedError("Symmetric LS fit pending.")
    raise ValueError(f"Unknown method: {method!r}")


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
        f.write("Result computed using the following .vtu files:\n")
        for r in results:
            f.write(f"- [{r.name}] {r.vtu_file}\n")
    print(f"[upscale_m] tensor report written: {os.path.abspath(path)}")
