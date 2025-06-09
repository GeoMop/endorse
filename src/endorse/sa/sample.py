from SALib.sample.sobol import sample as sobol
from SALib.sample.saltelli import sample as saltelli
from SALib.sample.morris import sample as morris
from SALib.sample.latin import sample as latin
from SALib.sample.finite_diff import sample as finite_diff
from SALib.sample.ff import sample as frac_fact


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
