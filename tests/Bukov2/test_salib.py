import random

from endorse.sa import sample, analyze
import numpy as np
import matplotlib.pyplot as plt
import pytest
import multiprocessing
import math
import time

def single_param_model(params):
    P1, = params
    epsilon = np.random.normal(scale=1)  # Adding random noise
    return 10 * P1 + epsilon



def forward_model(params, eps):
    P1, P2, P3, P4 = params
    #epsilon = np.random.normal(scale=eps)  # Adding random noise

    # Toy forward model equation with modified interaction terms
    Y = 5 * P1 + 32 * P2 + 10 * P3 + 0.1 * P4 +  10 * P1 * P4
    # if P1 < 0.01:
    #     Y = np.NaN
    """
    Result: replacing the NaN observations with the mean have nearly no impact on the results for 10% of NaNs
    but have very pronounced impact for 50% of Nans.
    """
    return Y


def calculate_and_format_ci(value, ci_width):
    # Calculate confidence interval as a percentage of the value
    lower_bound = value - 0.5 * ci_width
    upper_bound = value + 0.5 * ci_width

    # Format the result as a string
    result_str = f"{value:.4f} +/- {ci_width:.4f} ({100 * ci_width / value:.2f}%) [{lower_bound:.4f}, {upper_bound:.4f}]"

    return result_str


def print_sensitivity_results(problem, sobol_indices):
    # Print first-order Sobol indices with confidence intervals
    print("First-order Sobol indices with confidence intervals (95%):")
    for i, name in enumerate(problem['names']):
        S1_value = sobol_indices['S1'][i]
        S1_conf_width = sobol_indices['S1_conf'][i]

        result_str = calculate_and_format_ci(S1_value, S1_conf_width)
        print(f"{name}: {result_str}")

    # Print total-order Sobol indices with confidence intervals
    print("\nTotal-order Sobol indices with confidence intervals (95%):")
    for i, name in enumerate(problem['names']):
        ST_value = sobol_indices['ST'][i]
        ST_conf_width = sobol_indices['ST_conf'][i]

        result_str = calculate_and_format_ci(ST_value, ST_conf_width)
        print(f"{name}: {result_str}")

    # Print second-order Sobol indices with confidence intervals
    print("\nSecond-order Sobol indices:")
    for i, name1 in enumerate(problem['names']):
        for j, name2 in enumerate(problem['names']):
            if i < j:
                index_pair = (i, j)
                S2_value = sobol_indices['S2'][index_pair]
                S2_conf_width = sobol_indices['S2_conf'][index_pair]

                result_str = calculate_and_format_ci(S2_value, S2_conf_width)
                print(f"{name1}-{name2}: {result_str}")


def test_salib_small():
    # Define the problem for SALib
    problem = {
        'num_vars': 4,
        'names': ['P1', 'P2', 'P3', 'P4'],
        'bounds': [[0, 1], [0, 1], [0, 1], [0, 1]]

    }
    eps=10

    # Generate Saltelli samples
    param_values = sample.saltelli(problem, 1000, calc_second_order=True)

    # Evaluate the model for each set of parameters
    model_evaluations = np.array([forward_model(params, eps) for params in param_values])
    mean = np.nanmean(model_evaluations)
    model_evaluations[np.isnan(model_evaluations)] = mean

    # Perform Sobol sensitivity analysis
    print("\n")
    sobol_indices = analyze.sobol(
        problem, model_evaluations,
        calc_second_order=True, print_to_console=False)

    print_sensitivity_results(problem, sobol_indices)


def large_out_model(X, params):
    P1, P2, P3, P4 = params
    epsilon = np.random.normal(scale=0.1, size=len(X))  # Adding random noise

    Y = P1 + P2 * X + P3 * X**P4 + epsilon

    return Y


def plot_sampler(sampler):
    # Define the problem using ProblemSpec with bounds
    problem = {
        'names': ['P1', 'P2'],
        'bounds': [[0, 1], [0, 1]],
        'dists': ['norm', 'norm'],
        'num_vars': 2,
    }

    # Generate Saltelli samples
    param_values = sampler(problem, 1000)

    # Scatter plot
    plt.scatter(param_values[:, 0], param_values[:, 1], s=0.1)
    plt.title(f'{sampler.__module__} Samples, N: {len(param_values)}')
    plt.xlabel('Parameter P1')
    plt.ylabel('Parameter P2')
    plt.show()


@pytest.mark.skip
def test_samplers():
    plot_sampler(sample.sobol)
    plot_sampler(sample.saltelli)
    # These two gives nearly same results. N=6000
    plot_sampler(sample.latin)
    # Seems to produce smaller number of samples, N=1000
    plot_sampler(sample.morris)
    # Produce sort of regular gird of samples, independent of sample size.
    # N=3000
    plot_sampler(sample.finite_diff)
    # Interresting patterns.
    # N=3000
    plot_sampler(sample.frac_fact)
    # Empty
    # N=4


def large_model(X, params):
    """
    Somewhat inefficient to mimict performance of parallel execution with
    a simple model.
    :param X:
    :param params:
    :return:
    """
    np.random.seed(int(params[7] * 1000000))
    cond_eps = np.random.normal(0, 0.5)
    eps = np.random.normal(0, 0.1)
    stress = params[3] / (X + 2) + params[4]
    cond_x = params[0]  +  params[1] * X  + params[2] * 0.5 * (3 * X * X - 1)
    cond = np.exp( cond_x - params[5] * stress + cond_eps)
    flux = (stress + params[6]) * cond
    return flux * math.exp(eps)


class Analyze:
    def __init__(self, problem):
        self.problem = problem
    def single(self, output):
        return analyze.sobol(self.problem, output, calc_second_order=True, print_to_console=False)

@pytest.mark.skip
def test_salib_large_output():
    # Define the problem for SALib
    problem = {
        'num_vars': 8,
        'names': ['P1', 'P2', 'P3', 'P4',
                  'P5','P6','P7','P8'],
        'dists': ['norm', 'norm', 'norm',
                  'lognorm', 'lognorm',
                  'norm', 'norm', 'unif'],
        # available distributions:
        # unif - interval given by bounds
        # logunif,
        # triang - [lower_bound, upper_bound, mode_fraction]
        # norm,  bounds : [mean, std]
        # truncnorm, bounds : [lower_bound, upper_bound, mean, std_dev]
        # lognorm, bounds: [mean, std]  # mean and std of the log(X)
        'bounds': [[-9, 1], [0, 1], [0, 1], [0, 1],
                   [0, 1], [0, 1], [0, 1], [0, 1]]
    }

    # Generate Saltelli samples
    param_values = sample.saltelli(problem, 100, calc_second_order=True)

    X = np.linspace(-1, 1, 1000)

    t = time.perf_counter()
    # Evaluate the model for each set of parameters
    combined_values  = ( (X, p) for p in param_values )
    #with multiprocessing.Pool(4) as pool:
    #    Y_samples = pool.starmap(large_model, combined_values)
    Y_samples = [large_model(*c) for c in combined_values]
    print(f"Sample time: {time.perf_counter() - t}")

    t = time.perf_counter()
    # Perform Sobol sensitivity analysis
    print("\n")
    analyze = Analyze(problem)
    with multiprocessing.Pool(4) as pool:
        indices = pool.map(analyze.single, np.array(Y_samples).T)
    print(f"Analyze time: {time.perf_counter() - t}")

    # TODO: Plot indices spatial variability, using histograms
    # Use max of variability over time.
    # Unfortunately analysis takes quite some time:
    # Assuming linear dependence on number of outputs and number of samples we have:
    # 



#############
def forward_time_model(params, n_times):
    P1, P2, P3, P4 = params
    epsilon = np.random.normal(scale=0.1)  # Adding random noise
    t = np.linspace(0, 1, n_times)
    # Toy forward model equation with modified interaction terms
    Y = (5 * P1 + 1 * P2) *t  + (10 * P3 + 0.1 * P4 +  3 * P3 * P4) + epsilon
    #if P1 < 0.1:
    #    Y = np.NaN
    return Y


def fit(values):
    # values: (times, samples)
    a0 = np.mean(values, axis=0)
    a1 = np.mean(values[1:, :] - values[0:-1,:], axis=0)
    return np.stack((a0, a1))

#@pytest.mark.skip
def test_time_series_sa():
    # Define the problem for SALib
    problem = {
        'num_vars': 4,
        'names': ['P1', 'P2', 'P3', 'P4'],
        'bounds': [[0, 1], [0, 1], [0, 1], [0, 1]],
        'second_order': True
    }

    # Generate Saltelli samples
    param_values = sample.saltelli(problem, 1000, calc_second_order=True)

    # Evaluate the model for each set of parameters
    n_times = 10
    model_evaluations = np.array([forward_time_model(params, n_times) for params in param_values]).T
    #mean = np.nanmean(model_evaluations)
    #model_evaluations[np.isnan(model_evaluations)] = mean

    # Sensitivity of inversion
    fit(model_evaluations)
    sobol_array = analyze.sobol_vec(model_evaluations, problem)
    sobol_indices = analyze.sobol_max(sobol_array)
    print_sensitivity_results(problem, analyze.arr_to_dict(sobol_indices))

    # Perform Sobol sensitivity analysis, directly
    print("\n")
    sobol_indices = analyze.sobol_max(analyze.sobol_vec(model_evaluations, problem, second_order=True))
    print_sensitivity_results(problem, analyze.arr_to_dict(sobol_indices))

    """
    Results: perfect match
    So we conclude, that taking max of sensitivities from linearly derived values of the time series
    is the same as taking max from the time seties directly. 
    In particular if te derivative of the time series is sensitive to some parameter so would be the 
    time series itself. 
    """

