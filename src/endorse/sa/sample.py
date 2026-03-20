from SALib.sample.sobol import sample as sobol
from SALib.sample.saltelli import sample as saltelli
from SALib.sample.morris import sample as morris
from SALib.sample.latin import sample as latin
from SALib.sample.finite_diff import sample as finite_diff
from SALib.sample.ff import sample as frac_fact

import numpy as np

def prepare_problem_defition(parameters: list):
    """
    Create problem dict as input for salib.

    Available distributions and its parameters:
    unif - interval given by bounds
    logunif,
    triang - [lower_bound, upper_bound, mode_fraction]
    norm,  bounds : [mean, std]
    truncnorm, bounds : [lower_bound, upper_bound, mean, std_dev]
    lognorm, bounds: [mean, std]  # mean and std of the log(X)

    :param parameters: list of parameters with dict values: {"type": str, "name": str, "bounds": list(float)}
    :return: dict
    """
    for par in parameters:
        # process "seed" parameter: uniform dist [0,1]
        if par["type"] == "seed":
            par["type"] = "unif"
            par["bounds"] = [0,1]

    problem = {
        'num_vars': len(parameters),
        'names': [p["name"] for p in parameters],
        'dists': [p["type"] for p in parameters],
        'bounds': [p["bounds"] for p in parameters]
    }

    return problem


def _num_blocks(D: int, second_order: bool=True, groups=None) -> int:
    Dg = D if groups is None else len(np.unique(groups))
    return (2 * Dg + 2) if second_order else (Dg + 2)

def saltelli_qmc_idx(N: int, D: int, *, second_order: bool = True, groups=None) -> np.ndarray:
    """
    Per-row Sobol/QMC base index (0..N-1), tiled once per block.
    """
    nb = _num_blocks(D, second_order, groups)
    return np.tile(np.arange(N), nb)

def saltelli_block_idx(N: int, D: int, *, second_order: bool = True, groups=None) -> np.ndarray:
    """
    Per-row block id: 0=A, 1=B, 2..=A_Bi, (if 2nd) Dg+2..=B_Ai.
    Repeats each block id for N consecutive rows.
    """
    nb = _num_blocks(D, second_order, groups)
    return np.repeat(np.arange(nb), N)


def saltelli_base_idx(N: int, D: int, second_order: bool = True, groups=None) -> np.ndarray:
    """
    Return vector of base sample indices for each Saltelli-generated sample row.

    Parameters
    ----------
    N : int
        Base sample size.
    D : int
        Number of parameters.
    second_order : bool, default True
        Whether second-order samples are included.

    Returns
    -------
    np.ndarray
        Array of length (N * nb), where each entry is in [0, N-1].
        This tells you which base row (from A/B) the generated row corresponds to.
    """
    nb = 2 + D if not second_order else 2 + 2 * D
    total_rows = N * nb

    # Row r always corresponds to base index r % N
    # return np.arange(total_rows) % N
    return np.repeat(np.arange(N), nb)


# def saltelli_qmc_index(N, D, *, second_order: bool = True, groups=None):
#     """
#     Build the vector that tells you which Sobol (QMC) point each row of a
#     Saltelli design comes from.
#
#     Parameters
#     ----------
#     N : int
#         Base sample size you passed to `saltelli.sample`.
#     D : int
#         Number of original parameters  (problem['num_vars']).
#     second_order : bool, default True
#         Same flag you gave to `saltelli.sample`.
#     groups : list[str] or None
#         Optional grouping list (same length as D).  Leave None if you
#         didn’t use variable groups.
#
#     Returns
#     -------
#     idx : ndarray, shape (rows,)
#         Integer Sobol index for every row of the Saltelli matrix.
#     """
#
#     if groups is None:
#         Dg = D
#     else:
#         Dg = len(set(groups))       # one block per *group*, not per variable
#
#     block = (2 * Dg + 2) if second_order else (Dg + 2)
#     return np.repeat(np.arange(N), block)
#
#
#
def saltelli_ab_mask(N: int, D: int, second_order: bool = True, groups=None) -> np.ndarray:
    """
    Generate a mask matrix for a Saltelli sample.

    Each entry is:
      0 → parameter from matrix A
      1 → parameter from matrix B

    Parameters
    ----------
    N : int
        Base sample size.
    D : int
        Number of parameters.
    second_order : bool, default True
        Whether second-order samples are included.

    Returns
    -------
    np.ndarray
        Array of shape (N * num_blocks, D) with 0/1 mask.
    """
    nb = _num_blocks(D, second_order, groups)
    mask = np.zeros((N * nb, D), dtype=int)

    # Block 0: A → all zeros (already done by init)
    # Block 1: B → all ones
    mask[N:2 * N, :] = 1

    # Blocks 2..D+1: A_Bi → column i = 1 (from B), others = 0
    for i in range(D):
        block_start = (2 + i) * N
        block_end = block_start + N
        mask[block_start:block_end, i] = 1

    if second_order:
        # Blocks D+2..2D+1: B_Ai → column i = 0 (from A), others = 1
        for i in range(D):
            block_start = (D + 2 + i) * N
            block_end = block_start + N
            mask[block_start:block_end, :] = 1
            mask[block_start:block_end, i] = 0

    return mask


# def saltelli_block_labels(N, D, *, second_order: bool = True, groups=None):
#     """
#     Return a 1‑D array whose length equals the number of rows produced by
#     SALib's saltelli.sample().  Each element is 0, 1, 2, or 3 corresponds to A, B, A_Bi, B_Ai groups.
#
#     Parameters
#     ----------
#     N : int
#         Base sample size you passed to saltelli.sample().
#     D : int
#         Number of scalar variables (problem['num_vars']).
#     second_order : bool, default True
#         Same flag you gave to saltelli.sample().
#     groups : list[str] or None
#         The optional grouping list used in the problem dict.  Leave None if you
#         did not use groups.
#
#     Returns
#     -------
#     labels : ndarray, dtype=int8, shape (rows,)
#     """
#     # One block per *group*, not per individual column
#     Dg = D if groups is None else len(set(groups))
#
#     # Pattern for one Sobol point
#     if second_order:
#         pattern = np.array([0, 1] + [2] * Dg + [3] * Dg, dtype=np.int8)
#     else:
#         pattern = np.array([0, 1] + [2] * Dg,            dtype=np.int8)
#
#     # Tile it N times
#     return np.tile(pattern, N)