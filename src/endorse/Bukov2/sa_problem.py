import numpy as np


def log_bounds(cfg_lognorm_params):
    exp_mean, exp_std = cfg_lognorm_params
    return exp_mean, exp_std

def sa_dict(cfg_sim):
    params = cfg_sim['parameters']
    problem = {
        'num_vars': len(params),
        'names': [p["name"] for p in params],
        'dists': [p["type"] for p in params],
        # available distributions:
        # unif - interval given by bounds
        # logunif,
        # triang - [lower_bound, upper_bound, mode_fraction]
        # norm,  bounds : [mean, std]
        # truncnorm, bounds : [lower_bound, upper_bound, mean, std_dev]
        # lognorm, bounds: [mean, std]  # mean and std of the log(X)
        #'bounds': [log_bounds(p["bounds"]) for p in params],
        'second_order': cfg_sim.get('second_order_sa', True)    # Interpreted by endorse optimize.eval_from_chambers_sa
    }
    return problem