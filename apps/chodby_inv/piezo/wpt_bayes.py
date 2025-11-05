import logging
import os
from operator import inv
import pickle

import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag
import pandas as pd
from sys import argv, exit

# Import the borehole pressure model module.
from . import PoroElasticSolver
from chodby_inv import input_data, piezo
from endorse import common
from . import plot_idata, get_generic_name, save_idata_to_file, read_idata_from_file

# Import TinyDA (assumes TinyDA is installed; adjust the import if needed)
import xarray as xr
import tinyDA as tda




def exponential_covariance(n, dx, correlation_length, variance):
    """
    Build an n x n covariance matrix using an exponential decay model.
    
    Cov(i,j) = variance * exp(-|x_i - x_j| / correlation_length)
    
    Parameters:
      n: number of parameters.
      dx: spacing between adjacent elements (assumed uniform).
      correlation_length: spatial correlation length [same units as dx].
      variance: variance of the log(k) field.
    
    Returns:
      cov: n x n covariance matrix.
    """
    coords = np.arange(n) * dx
    cov = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            distance = np.abs(coords[i] - coords[j])
            cov[i, j] = variance * np.exp(-distance / correlation_length)
    return cov


def borehole_section_inversion(inv_cfg, ):
    df = piezo.full_flat_df()
    epoch_df = piezo.get_epoch(df, inv_cfg)
    #epoch_df.set_index('time_days', inplace=True)
    return _run_inversion(inv_cfg, epoch_df)

def interpolate_pressure_series(epoch_df, dt):
    # create new time series
    # minimum and maximum is set from epoch_df
    time_series = epoch_df.time_days.values
    target_points = pd.DataFrame({
        "time_days": np.arange(time_series.min(), time_series.max(), dt)
    })

    print(target_points)

    # extend the pressure series to include target points and remove duplicates
    # new values have pressure NaN, which will be interpolated later
    merged = pd.concat([epoch_df, target_points]) \
        .drop_duplicates(subset="time_days", keep="first") \
        .sort_values("time_days")

    # interpolate missing pressure values
    merged["pressure"] = merged["pressure"].interpolate(method="linear")
    # df contains both old and new time values - extract only the target points
    target_subset = merged[merged["time_days"].isin(target_points["time_days"])]

    # return the pressure series and the time series
    return target_subset["pressure"].values * 1000, target_subset["time_days"].values


@common.memoize
def _run_inversion(inv_cfg, epoch_df):

    #dt = 6*60 * 60  # dt = 6 hours
    dt = inv_cfg.get("dt", 6 * 60 * 60)  # Default to 6 hours if not specified
    dt_days = dt / (24 * 3600)  # Convert to days
    regular_pb_measured_extended, time_series = interpolate_pressure_series(epoch_df, dt_days)

    print(regular_pb_measured_extended)
    #print(time_series)
    regular_pb_measured = regular_pb_measured_extended[time_series >= 0]
    print(regular_pb_measured)
    tests = load_pressure_tests()
    #selected_test = tests[inv_cfg["section"]]
    selected_test = next((t for t in tests if t["vrt"] == inv_cfg["borehole"] and t["sekce"] == inv_cfg["section"]), None)
    if selected_test is None:
        return None

    # Geometry and time-stepping parameters.
    r_b = 0.076  # Borehole radius [m]
    R = 2  # Outer domain radius [m]
    N = 5  # Number of finite elements (⇒ N+1 nodes)
    geom_power = 2 # 1 = even spacing of elements, >1 = concentrated at borehole
    T_final = dt * (len(regular_pb_measured) - 1)  # Total simulation time: 1 day [s]
    #p_b0 = 1000* 1000  # Elevated borehole pressure (node 0) [Pa]
    #p_b0 = selected_test["tlak"]
    p_b0 = regular_pb_measured[0]
    # load p_far_prior from config, if available
    # or default to the last measured pressure
    # same for deviation, default to 10kPa if not specified
    p_far_prior = inv_cfg.get("p_far_prior", regular_pb_measured[-1])
    p_far_std = inv_cfg.get("p_far_std", 10 * 1000)
    k_prior = 1e-9

    # Rock and fluid parameters.
    biot = 0.2
    phi = 0.02  # Porosity (dimensionless)
    E_prior = 50e9  # Young's modulus [Pa]
    nu = 0.25  # Poisson's ratio
    solver = PoroElasticSolver(r_b, R, N, geom_power, dt, T_final, p_b0)
    def forward_model(param_vec):
        k_field = np.power(10, param_vec[:N])  # Convert log(k) to k
        E_field = np.power(10, param_vec[N:2*N])
        p_far = param_vec[-1]
        try:
            t,p,p_b = solver.simulate(biot, phi, E_field, nu, p_far, k_field)
            flux = -solver.C_b[0] * (p_b[1] - p_b[0]) / dt
            return np.concatenate((np.log10([flux]), p_b))
        except Exception as e:
            print(f"Simulation failed for parameters: {param_vec}, error: {e}")
            # Return a large penalty value to indicate failure
            return np.full(len(regular_pb_measured) + 1, 1e10)


    # L = 2     # [m] Length of the borehole section
    # S = 2 * np.pi * r_b * L
    # c_f = 1e-9  # Fluid compressibility [Pa^-1]
    # C_b = np.pi * r_b ** 2 * (c_f + (2 * (1 - nu) * (1 - nu ** 2)) / E) * L
    # times = np.arange(0, T_final + dt, dt)
    # def forward_model(param):
    #     k, p_far = param
    #     #k = np.mean(k_field)
    #     k = np.exp(k)
    #     lmbd = k * S / C_b
    #     print(f"params: {k}, {p_far}, {lmbd}")
    #
    #     p_t = (p_b0 - p_far) * np.exp(-lmbd * times) + p_far
    #     return p_t

    # ==============================================================================
    # 3. Prior, Likelihood, and Covariance Setup
    # ==============================================================================

    # We will invert for the N parameters (one per finite element) in log-space.
    param_dim = N
    # Estimate element spacing in the radial direction.
    dx = (R - r_b) / N
    k_correlation_length = 0.1  # [m] (adjust as needed)
    k_prior_std_log10 = 2   # Variance of the log(k) field (adjust based on your prior belief)
    E_correlation_length = 0.1  # [m] (adjust as needed)
    E_prior_std_log10 = 1.2  # Variance of the log(k) field (adjust based on your prior belief)
    mean_prior = np.concatenate([
        np.full(param_dim, np.log10(k_prior)),
        np.full(param_dim, np.log10(E_prior)),
        np.array([p_far_prior])
    ])

    #mean_prior[12]

    k_prior_variance = (k_prior_std_log10) ** 2
    E_prior_variance = (E_prior_std_log10) ** 2
    cov_prior = block_diag(
        exponential_covariance(param_dim, dx, k_correlation_length,
                               k_prior_variance),
        exponential_covariance(param_dim, dx, E_correlation_length,
                               E_prior_variance),
        np.array([[p_far_std**2]])
    )

    prior = multivariate_normal(mean_prior, cov_prior)


    if piezo.to_datetime(inv_cfg["origin"]).year > 2024:
        flow_rate_observed = np.array([selected_test["spotreba"]])
        #flow_rate_sigma = np.array([1e-6])
        #flow_rate_sigma = np.array([selected_test["spotreba_sigma"]])
        flow_rate_sigma = np.log10(100 / 94) / 3
        if flow_rate_sigma == 0:
            # cover cases when sigma is zero
            flow_rate_sigma = np.array([1e-11])
        plot_observed_flow = True
    else:
        flow_rate_sigma = 10000
        flow_rate_observed = np.array([1e6])
        plot_observed_flow = False

    pressure_output_sigma = 3 * 1000
    pressure_output_sigma = np.full(len(regular_pb_measured), pressure_output_sigma)

    observed = np.concatenate([
        np.log10(flow_rate_observed),
        regular_pb_measured
    ])

    sigma = np.concatenate([
        #np.log10(flow_rate_sigma),
        np.array([flow_rate_sigma]),
        pressure_output_sigma
    ])

    cov_likelihood = sigma ** 2 * np.eye(len(observed))
    print(cov_likelihood)

    loglike = tda.GaussianLogLike(observed, cov_likelihood)
    posterior = tda.Posterior(prior, loglike, forward_model)

    # ==============================================================================
    # 4. Configuring and Running the Bayesian Inversion using TinyDA
    # ==============================================================================

    # Here we assume that TinyDA provides a class (for example, BayesInversion)
    # where you can pass the forward model, measured data, prior mean/covariance, 
    # and noise covariance. The exact API may differ; adjust according to your TinyDA version.

    iterations = 30000
    burnin = 10000
    chains = 20


    # rwmh_cov = np.eye(len(mean_prior)) * 0.2
    # rwmh_cov = np.diag(np.power(mean_prior * 0.1, 2), 0)
    # rmwh_scaling = 0.1
    # rwmh_adaptive = True
    # my_proposal = tda.AdaptiveMetropolis(C0=rwmh_cov,
    #                                      period=50,
    #                                      adaptive=rwmh_adaptive)

    # m0 = 10000 # initial archive size
    # delta = 5 # number of pairs to compute jump vector
    # adaptive = True
    # adaptivity_period = 50  # Period for adaptation
    # nCR = 15 # up to how many parameters can change in a proposal
    # my_proposal = tda.DREAMZ(
    #     m0,
    #     delta,
    #     nCR=nCR,
    #     adaptive=adaptive,
    #     period=adaptivity_period
    # )

    # old params
    # m0 = 10000 # initial archive size
    # delta = 5 # number of pairs to compute jump vector
    # adaptive = True
    # adaptivity_period = 50  # Period for adaptation
    # nCR = 15 # up to how many parameters can change in a proposal
    # Z_method = "random"
    # b = 0.05
    # b_star = 1e-6


    # new params
    m0 = 1000             # should be sufficient in combination with 'lhs'
    delta = 1              
    # number of jump pairs added to consturuct proposal jump
    # at most 2 to have meaning proposal jumps.
    Z_method = 'lhs'      # better prior coverage for initial archive
    nCR = 3              
    # Unintuitive parameter. Too large values leads to very sparse jump vectors
    # nCR=1 means select all parameters, nCR=3 chood parameter probability from discrete set {1/3, 1/2, 1}
    adaptive = True
    # Less aggressive adaptivity, could avoid potential problems with ergodicity. 
    # More diagnostics needed for reasonable choice.
    adaptivity_period = 100   # keep same
    gamma = 1.01             # more aggresive adaptivity
    b = 0.1
    # maximal relative prolongation / contraction of the jump
    # 0.1 still needs about 10-40 steps to explore space between two isoolated parameter sets 
    b_star = 1e-5         
    # a bit more aggresive, but mainly have no impact

    sync_rate = 1000
    stuck_checking_start = 3000
    stuck_checking_period = 1000


    my_proposal = tda.DREAM(
        m0,
        delta,
        nCR=nCR,
        adaptive=adaptive,
        period=adaptivity_period,
        sync_rate=sync_rate,
        stuck_checking_start=stuck_checking_start,
        stuck_checking_period=stuck_checking_period,
        Z_method=Z_method,
        b=b,
        b_star=b_star
    )

    # pcn_scaling = 0.1
    # pcn_adaptive = False
    # my_proposal = tda.CrankNicolson(scaling=pcn_scaling, adaptive=pcn_adaptive)

    # am_cov = np.eye(2)
    # am_t0 = 2000
    # am_sd = 1
    # am_epsilon = 1e-6
    # my_kernel = tda.AdaptiveMetropolis(C0=am_cov, t0=am_t0, sd=am_sd, epsilon=am_epsilon)

    #my_proposal = tda.MultipleTry(my_kernel, 3)
    my_chains = tda.sample(posterior, my_proposal, iterations=iterations, n_chains=chains)
      # define input variable names for inferencedata
    #parameter_names = [f"k_{n}" for n in range(param_dim)] +  ["P_far"]
    parameter_names = \
        [f"log_k_{n}" for n in range(param_dim)] + \
        [f"log_E_{n}" for n in range(param_dim)] + \
        ["p_far"]

    # construct idata object from the chains and include info about burn in
    idata = tda.to_inference_data(my_chains, parameter_names=parameter_names)
    idata.attrs["burnin"] = burnin

    # add the observed data to the InferenceData object
    idata["sample_stats"].attrs["observed_timeseries"] = time_series
    idata["sample_stats"].attrs["observed_pressure"] = regular_pb_measured
    idata["sample_stats"].attrs["observed_pressure_sigma"] = pressure_output_sigma
    idata["sample_stats"].attrs["observed_flow"] = np.log10(flow_rate_observed)
    idata["sample_stats"].attrs["observed_flow_sigma"] = flow_rate_sigma
    idata["sample_stats"].attrs["observed_extended"] = regular_pb_measured_extended

    # add prior information to the InferenceData object
    idata["posterior"].attrs["prior_mean"] = mean_prior
    idata["posterior"].attrs["prior_cov"] = cov_prior

    # add metadata to the InferenceData object
    idata.attrs["borehole"] = inv_cfg["borehole"]
    idata.attrs["section"] = inv_cfg["section"]
    idata.attrs["year"] = piezo.to_datetime(inv_cfg["origin"]).year
    idata.attrs["month"] = piezo.to_datetime(inv_cfg["origin"]).month
    idata.attrs["plot_observed_flow"] = plot_observed_flow
    idata.attrs["solver_radii"] = solver.r

    return idata

def load_pressure_tests(path=input_data.wpt_multipacker):
    try:
        df = pd.read_excel(path, sheet_name="data (2)")
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        return None

    col_keys = df.columns.tolist()
    
    zkouska_starts = df[df["čas"] == 0].index.tolist()

    vodivost_true_idx = int(df.columns.get_loc("hydraulická vodivost") + 1)

    zkousky = []
    minule_datum = None

    for i, start in enumerate(zkouska_starts):
        if i < len(zkouska_starts) - 1:
            end = zkouska_starts[i + 1] - 1
        else:
            end = len(df) - 1
        
        while np.any([
            pd.isna(df.iloc[end]["čas"]),
            pd.isna(df.iloc[end]["spotřeba"]),
            pd.isna(df.iloc[end]["hydraulická vodivost"])
        ]):
            # If the end row has NaN values, adjust the end index
            end -= 1
            if end < start or end <= 0:
                print(f"Skipping invalid section from {start} to {end}.")
                continue

        spotreba = df.iloc[end]["spotřeba.1"]
        spotreba_sigma = df.iloc[end]["spotřeba sigma"]
        vodivost = df.iloc[end]["hydraulická vodivost"]
        vodivost_true = df.iloc[end][vodivost_true_idx]
        etaz = df.iloc[start]["etáž"]
        vrt = df.iloc[start]["vrt"]
        sekce = df.iloc[start]["sekce"]
        tlak = df.iloc[end]["tlak v intervalu"] * 1e3 # convert from kPa to Pa

        datum = df.iloc[start]["datum a čas"]
        if not pd.isna(datum):
            datum = pd.to_datetime(datum, format="%m.%d.%Y %H:%M:%S")
        else:
            datum = minule_datum


        zkousky.append({
            "date": datum,
            "vrt": vrt,
            "sekce": sekce,
            "etaz": etaz,
            "spotreba": spotreba,
            "spotreba_sigma": spotreba_sigma,
            "tlak": tlak,
            "vodivost": vodivost,
            "vodivost_true": vodivost_true
        })

        minule_datum = datum

    return zkousky

if __name__ == '__main__':
    try:
        selected_test = int(argv[1])
    except:
        print("No test index provided, exiting...")
        exit(1)

    events = common.load_config(input_data.events_yaml)['water_pressure_tests']
    if selected_test >= len(events):
        print(f"Test index {selected_test} out of range, exiting...")
        exit(1)
    
    wpt_cfg = events[selected_test]
    idata = borehole_section_inversion(wpt_cfg)
    save_idata_to_file(idata, f"{get_generic_name(idata)}.idata")
    
    idata_loaded = read_idata_from_file(f"{get_generic_name(idata)}.idata")
    plot_idata(idata_loaded)