import numpy as np
import yaml
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.stats import multivariate_normal, norm
from scipy.linalg import block_diag
import pandas as pd
from pprint import pprint
import logging

"""

"""


# Import the borehole pressure model module.
from . import PoroElasticSolver
from chodby_inv import input_data, piezo
from endorse import common

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

@common.memoize
def _run_inversion(inv_cfg, epoch_df):

    dt = 6*60 * 60  # dt = 6 hours
    #time_delta = pd.Timedelta(dt, unit='s')
    dt_days = dt / (24 * 3600)  # Convert to days
    time_days = epoch_df.time_days.values
    p_b_measured = epoch_df.pressure.values
    def smooth_fn(x):
        t_diff = time_days - x.time_days
        mask = (t_diff >  - dt_days/2) & (t_diff <=  dt_days/2)
        return p_b_measured[mask].mean() * 1000

    df_reg = pd.DataFrame({
        "time_days": np.arange(time_days.min(), time_days.max(), dt_days)
    })
    regular_pb_measured = df_reg.apply(smooth_fn, axis=1).values

    tests = load_pressure_tests()
    selected_test = tests[inv_cfg["section"]]

    # Geometry and time-stepping parameters.
    r_b = 0.076  # Borehole radius [m]
    R = 2  # Outer domain radius [m]
    N = 10  # Number of finite elements (⇒ N+1 nodes)
    T_final = dt * (len(regular_pb_measured) - 1)  # Total simulation time: 1 day [s]
    #p_b0 = 1000* 1000  # Elevated borehole pressure (node 0) [Pa]
    p_b0_prior = selected_test["tlak"]
    p_b0_sd = 20000
    p_far_prior = 300* 1000  # Far-field Dirichlet pressure (last node) [Pa]
    p_far_sd = 5000 # 5kPa
    k_prior = 1e-13

    # Rock and fluid parameters.
    biot = 0.2
    phi = 0.02  # Porosity (dimensionless)
    E = 30e9  # Young's modulus [Pa]
    nu = 0.25  # Poisson's ratio

    solver = PoroElasticSolver(r_b, R, N, dt, T_final, p_b0_prior)
    compress_prior_mean = solver.estimate_complience(biot, phi, E, nu)

    def forward_model(param_vec):
        compress, K, p_init, p_far = param_vec
        C = np.exp(compress)  # Convert log(C) to C
        K = np.exp(K)  # Convert log(K) to K
        k_field = K * np.ones(N)  # Assuming uniform hydraulic conductivity
        t,p,p_b = solver.simulate(biot, phi, E, nu, p_init,  p_far, k_field, C)
        flux = K * (p_init - p_far) * (r_b**2) / (2 * r_b)
        # This is  over simplification, we should take presurization history to account.
        output = [flux, *p_b]
        return np.array(output)


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
    correlation_length = 0.01  # [m] (adjust as needed)
    prior_std_log10 = 0.5   # Variance of the log(k) field (adjust based on your prior belief)
    # mean_prior = np.concatenate([
    #     np.full(param_dim, np.log(k_prior)),
    #     np.array([p_far_prior])
    # ])
    # prior_variance = (prior_std_log10 * np.log(10)) ** 2
    # cov_prior = block_diag(
    #     exponential_covariance(param_dim, dx, correlation_length,
    #                            prior_variance),
    #     np.array([[p_far_sd**2]])
    # )

    mean_prior = np.array([
        compress_prior_mean, # C?
        np.log(k_prior), # K
        p_b0_prior, # P_init
        p_far_prior # P_far
    ])

    prior_variance = (prior_std_log10 * np.log(10)) ** 2
    cov_prior = block_diag(
        prior_variance, # C?
        prior_variance, # K?
        p_b0_sd, # P_init
        p_far_sd # P_far
    )

    prior = multivariate_normal(mean_prior, cov_prior)


    # [ln k, p_far]
    # mean_prior = np.array([np.log(k_prior), p_far])
    # cov_prior = np.diag([0.5**2, 5000**2])
    # prior = multivariate_normal(mean_prior, cov_prior)

    flow_rate_observed = np.array([selected_test["spotreba"]])
    flow_rate_sigma = np.array([1e-7])
    pressure_output_sigma = 2000 # 2 kPa

    observed = np.concatenate([
        flow_rate_observed,
        regular_pb_measured
    ])

    sigma = np.concatenate([
        flow_rate_sigma, 
        np.full(len(regular_pb_measured), pressure_output_sigma) # 2 kPa
    ])

    cov_likelihood = sigma ** 2 * np.eye(len(observed))
    
    loglike = tda.GaussianLogLike(observed, cov_likelihood)
    posterior = tda.Posterior(prior, loglike, forward_model)

    # ==============================================================================
    # 4. Configuring and Running the Bayesian Inversion using TinyDA
    # ==============================================================================

    # Here we assume that TinyDA provides a class (for example, BayesInversion)
    # where you can pass the forward model, measured data, prior mean/covariance, 
    # and noise covariance. The exact API may differ; adjust according to your TinyDA version.

    rwmh_cov = np.eye(len(mean_prior)) * 0.2
    rmwh_scaling = 0.1
    rwmh_adaptive = True
    my_proposal = tda.AdaptiveMetropolis(C0=rwmh_cov,
                                         period=50,
                                         adaptive=rwmh_adaptive)

    # pcn_scaling = 0.1
    # pcn_adaptive = False
    # my_proposal = tda.CrankNicolson(scaling=pcn_scaling, adaptive=pcn_adaptive)

    # am_cov = np.eye(2)
    # am_t0 = 2000
    # am_sd = 1
    # am_epsilon = 1e-6
    # my_kernel = tda.AdaptiveMetropolis(C0=am_cov, t0=am_t0, sd=am_sd, epsilon=am_epsilon)

    #my_proposal = tda.MultipleTry(my_kernel, 3)
    iterations = 2000
    burnin = 50
    my_chains = tda.sample(posterior, my_proposal, iterations=iterations, n_chains=4)

    # define input variable names for inferencedata
    #parameter_names = [f"k_{n}" for n in range(param_dim)] +  ["P_far"]
    parameter_names = [
        "C",  # Compliance
        "K",  # Hydraulic conductivity
        "P_init",  # Initial borehole pressure
        "P_far"  # Far-field pressure
    ]

    # construct idata object from the chains
    idata = tda.to_inference_data(my_chains, burnin=burnin, parameter_names=parameter_names)

    # add the observed data to the InferenceData object
    idata["sample_stats"].attrs["observed_data"] = regular_pb_measured
    idata["sample_stats"].attrs["observed_cov"] = cov_likelihood

    # add prior information to the InferenceData object
    idata["posterior"].attrs["prior_mean"] = mean_prior
    idata["posterior"].attrs["prior_cov"] = cov_prior

    return idata

    # # ==============================================================================
    # # 5. Postprocessing and Comparison of Results
    # # ==============================================================================
    #
    # # Convert the MAP estimate from log space to the physical conductivity field.
    # estimated_k_field = np.exp(posterior_mean)
    #
    # # Plot the true versus estimated conductivity field.
    # plt.figure(figsize=(10, 6))
    # element_indices = np.arange(1, N+1)  # element numbering (or use radial midpoints if desired)
    # plt.plot(element_indices, conductivity_true, label='True Conductivity')
    # plt.plot(element_indices, estimated_k_field, label='Estimated Conductivity', linestyle='--')
    # plt.xlabel('Element Index')
    # plt.ylabel('Hydraulic Conductivity [m²/s]')
    # plt.legend()
    # plt.title('True vs Estimated Hydraulic Conductivity Field')
    # plt.grid(True)
    # plt.show()
    #
    # # Compare the measured borehole pressure history with the model prediction using the MAP estimate.
    # p_b_estimated = forward_model(posterior_mean)
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_vec / 3600, p_b_measured, 'o', label='Measured Borehole Pressure')
    # plt.plot(time_vec / 3600, p_b_estimated, '-', label='Model Prediction (MAP)')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Borehole Pressure (Pa)')
    # plt.legend()
    # plt.title('Borehole Pressure Relaxation: Measured vs Predicted')
    # plt.grid(True)
    # plt.show()

def plot_idata(idata):

    #az.style.use("arviz-doc")


    plot_observe(idata, bins=80)
    plt.savefig("observe_plot.pdf", dpi=300)
    #plt.show()

    az.summary(idata)

    # plot trace and force axis to use scientitic notation
    plot_trace_modified(idata, figsize=(16, 18))
    plt.savefig("trace_plot.pdf", dpi=300)

    # plot posterior distributions and corresponding prior distributions
    plot_posterior_modified(idata, figsize=(16, 18))

    plt.savefig("posterior_plot.pdf", dpi=300)
    plot_observe(idata)

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
        vodivost = df.iloc[end]["hydraulická vodivost"]
        vodivost_true = df.iloc[end][vodivost_true_idx]
        etaz = df.iloc[start]["etáž"]
        vrt = df.iloc[start]["vrt"]
        tlak = df.iloc[end]["tlak v intervalu"] * 1e3 # convert from kPa to Pa

        datum = df.iloc[start]["datum a čas"]
        if not pd.isna(datum):
            datum = pd.to_datetime(datum, format="%m.%d.%Y %H:%M:%S")
        else:
            datum = minule_datum


        zkousky.append({
            "date": datum,
            "vrt": vrt,
            "etaz": etaz,
            "spotreba": spotreba,
            "tlak": tlak,
            "vodivost": vodivost,
            "vodivost_true": vodivost_true
        })

        minule_datum = datum

    return zkousky

def plot_observe(idata, p_obs=None, ax=None, bins=100):
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 9))

    if p_obs is None:
        # attempt to load data directly from idata
        if idata.sample_stats.attrs["observed_data"] is not None:
            p_obs = idata.sample_stats.attrs["observed_data"]

    observe = idata.posterior_predictive
    observe_vars = sorted([v for v in observe.data_vars if v.startswith("obs_")],
                      key=lambda s: int(s.split("_", 1)[1]))

    observe_list = [observe[v] for v in observe_vars]
    observe_arr = xr.concat(observe_list, dim="time")
    chains = observe_arr.sizes["chain"]
    draws = observe_arr.sizes["draw"]

    observe_arr = observe_arr.stack(flat_dim=("time", "chain", "draw")).reset_index("flat_dim", drop=True)
    observe_length = len(observe_list)

    #print(observe_arr.shape)

    hist2d_x = np.repeat(np.arange(observe_length), chains * draws)

    ax.hist2d(hist2d_x, observe_arr.values, bins=[observe_length, bins], cmap="viridis", cmin=1e-10)
    ax.plot(np.arange(observe_length - 1), p_obs, "r-", label="Predicted observation", lw=2)
    ax.set_ylim(
        [
            np.min([observe_arr.min(), p_obs.min()]),
            np.max([observe_arr.max(), p_obs.max()])
        ]
    )
    ax.set_xlabel("Time (integer steps)")
    ax.set_ylabel("Pressure")
    ax.legend()
    plt.colorbar(ax.collections[0], ax=ax, label="Counts")

    return ax

def plot_trace_modified(idata, *args, **kwargs):
    axes = az.plot_trace(idata, *args, **kwargs)

    for ax_row in axes:
        for ax in ax_row:
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

    fig = plt.gcf()
    fig.set_constrained_layout(True)

    return axes

def plot_posterior_modified(idata, *args, **kwargs):
    axes = az.plot_posterior(idata, *args, **kwargs)

    # add prior, if available
    if np.all([
        idata.posterior.attrs["prior_mean"] is not None,
        idata.posterior.attrs["prior_cov"] is not None
    ]):
        prior_mean = idata.posterior.attrs["prior_mean"]
        prior_cov = idata.posterior.attrs["prior_cov"]

        assert isinstance(prior_mean, np.ndarray), "Prior mean should be a numpy array."
        assert isinstance(prior_cov, np.ndarray), "Prior covariance should be a numpy array."
        assert prior_mean.ndim == 1, "Prior mean should be a 1D array."
        assert prior_cov.ndim == 2, "Prior covariance should be a 2D array."
        assert prior_mean.shape[0] == prior_cov.shape[0] == prior_cov.shape[1], \
            "Prior mean and covariance dimensions do not match."
        
        prior_sd = np.sqrt(np.diag(prior_cov))

        # iterate across axes and add corresponding prior plots
        for x, ax_row in enumerate(axes):
            if isinstance(ax_row, list):
                for y, ax in enumerate(ax_row):
                    idx = x * len(ax_row) + y
                    # if empty axis, skip
                    if not (ax.lines or ax.images or ax.collections or ax.patches):
                        continue
                    mean = prior_mean[idx]
                    sd = prior_sd[idx]
                    xvals = np.linspace(mean - 3 * sd, mean + 3 * sd, 100)
                    yvals = norm.pdf(xvals, mean, sd)
                    ax.plot(xvals, yvals, color="red", linestyle="dashed", label="Původní odhad")
            else:
                idx = x
                # if empty axis, skip
                if not (ax_row.lines or ax_row.images or ax_row.collections or ax_row.patches):
                    continue
                mean = prior_mean[idx]
                sd = prior_sd[idx]
                xvals = np.linspace(mean - 3 * sd, mean + 3 * sd, 100)
                yvals = norm.pdf(xvals, mean, sd)
                ax_row.plot(xvals, yvals, color="red", linestyle="dashed", label="Původní odhad")

    # set scientific notation for axes
    for ax_row in axes:
        if isinstance(ax_row, list):
            for ax in ax_row:
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
        else:
            # single axis
            ax_row.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax_row.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

    # constrained layout for better spacing
    fig = plt.gcf()
    fig.set_constrained_layout(True)

    return axes

if __name__ == '__main__':
    wpt_cfg = common.load_config(input_data.events_yaml)['water_pressure_tests'][0]
    #bh_inv_cfg = yaml.load(bh_inv_cfg_yaml)
    idata = borehole_section_inversion(wpt_cfg)
    plot_idata(idata)