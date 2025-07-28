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


def saltelli_qmc_index(N, D, *, calc_second_order=True, groups=None):
    """
    Build the vector that tells you which Sobol (QMC) point each row of a
    Saltelli design comes from.

    Parameters
    ----------
    N : int
        Base sample size you passed to `saltelli.sample`.
    D : int
        Number of original parameters  (problem['num_vars']).
    calc_second_order : bool, default True
        Same flag you gave to `saltelli.sample`.
    groups : list[str] or None
        Optional grouping list (same length as D).  Leave None if you
        didn’t use variable groups.

    Returns
    -------
    idx : ndarray, shape (rows,)
        Integer Sobol index for every row of the Saltelli matrix.
    """

    if groups is None:
        Dg = D
    else:
        Dg = len(set(groups))       # one block per *group*, not per variable

    block = (2 * Dg + 2) if calc_second_order else (Dg + 2)
    return np.repeat(np.arange(N), block)


def saltelli_block_labels(N, D, *, calc_second_order=True, groups=None):
    """
    Return a 1‑D array whose length equals the number of rows produced by
    SALib's saltelli.sample().  Each element is 0, 1, 2, or 3 as defined above.

    Parameters
    ----------
    N : int
        Base sample size you passed to saltelli.sample().
    D : int
        Number of scalar variables (problem['num_vars']).
    calc_second_order : bool, default True
        Same flag you gave to saltelli.sample().
    groups : list[str] or None
        The optional grouping list used in the problem dict.  Leave None if you
        did not use groups.

    Returns
    -------
    labels : ndarray, dtype=int8, shape (rows,)
    """
    # One block per *group*, not per individual column
    Dg = D if groups is None else len(set(groups))

    # Pattern for one Sobol point
    if calc_second_order:
        pattern = np.array([0, 1] + [2] * Dg + [3] * Dg, dtype=np.int8)
    else:
        pattern = np.array([0, 1] + [2] * Dg,            dtype=np.int8)

    # Tile it N times
    return np.tile(pattern, N)