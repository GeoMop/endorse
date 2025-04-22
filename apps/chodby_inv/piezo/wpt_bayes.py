import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag
import pandas as pd
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

def borehole_section_inversion(inv_cfg):
    dt = 6*60 * 60  # dt = 6 hours
    #time_delta = pd.Timedelta(dt, unit='s')
    dt_days = dt / (24 * 3600)  # Convert to days
    df = piezo.full_flat_df()
    epoch_df = piezo.get_epoch(df, inv_cfg)
    #epoch_df.set_index('time_days', inplace=True)
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

    # Geometry and time-stepping parameters.
    r_b = 0.076  # Borehole radius [m]
    R = 2  # Outer domain radius [m]
    N = 10  # Number of finite elements (⇒ N+1 nodes)
    T_final = dt * (len(regular_pb_measured) - 1)  # Total simulation time: 1 day [s]
    p_b0 = 1000* 1000  # Elevated borehole pressure (node 0) [Pa]
    p_far_prior = 300* 1000  # Far-field Dirichlet pressure (last node) [Pa]
    p_far_sd = 5000 # 5kPa
    k_prior = 1e-13

    # Rock and fluid parameters.
    biot = 0.2
    phi = 0.02  # Porosity (dimensionless)
    E = 30e9  # Young's modulus [Pa]
    nu = 0.25  # Poisson's ratio
    solver = PoroElasticSolver(r_b, R, N, dt, T_final, p_b0)
    def forward_model(param_vec):

        k_field = np.exp(param_vec[:-1])  # Convert log(k) to k
        p_far = param_vec[-1]
        t,p,p_b = solver.simulate(biot, phi, E, nu, p_far, k_field)
        return p_b


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
    correlation_length = 0.5  # [m] (adjust as needed)
    prior_variance = 0.7**2   # Variance of the log(k) field (adjust based on your prior belief)
    mean_prior = np.concatenate([
        np.full(param_dim, np.log(k_prior)),
        np.array([p_far_prior])
    ])
    cov_prior = block_diag(
        exponential_covariance(param_dim, dx, correlation_length, prior_variance),
        np.array([[p_far_sd**2]])
    )
    prior = multivariate_normal(mean_prior, cov_prior)


    # [ln k, p_far]
    # mean_prior = np.array([np.log(k_prior), p_far])
    # cov_prior = np.diag([0.5**2, 5000**2])
    # prior = multivariate_normal(mean_prior, cov_prior)

    sigma = 2000 # 2 kPa
    cov_likelihood = sigma ** 2 * np.eye(len(regular_pb_measured))
    loglike = tda.GaussianLogLike(regular_pb_measured, cov_likelihood)
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
    idata = tda.to_inference_data(my_chains, burnin=burnin)



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

def plot_idata(idata, p_obs):
    import arviz as az
    az.style.use("arviz-doc")


    # 1) pick off all your per‑time PPC variables
    ppc_ds = idata.posterior_predictive
    obs_vars = sorted([v for v in ppc_ds.data_vars if v.startswith("obs_")],
                      key=lambda s: int(s.split("_", 1)[1]))

    # 2) concat them into one DataArray of shape (chain, draw, time)
    ppc_list = [ppc_ds[v] for v in obs_vars]
    ppc_arr = xr.concat(ppc_list, dim="time")
    # give it a name and meaningful coords
    ppc_arr = ppc_arr.assign_coords(
        time=("time", df_reg.time_days.values)  # or whatever your timestamps are
    )

    ppc_arr.name = "pressure"  # new var name

    # 1) Stack chain & draw into one "sample" dimension
    ppc_stacked = ppc_arr.stack(sample=("chain", "draw"))  # now dims = ("time","sample")

    # 2) Define integer time steps 0,1,2,...,T-1
    time = np.arange(ppc_arr.sizes["time"])

    # 3) Plot each posterior draw
    fig, ax = plt.subplots(figsize=(10, 6))
    for vals in ppc_stacked.values.T:  # iterate over samples
        ax.plot(time, vals, color="C0", alpha=0.05, linewidth=0.5)

    # 4) Overlay your observed (smoothed) series
    ax.plot(time, regular_pb_measured, color="k", linewidth=2, label="Observed")

    # 5) Label axes
    ax.set_xlabel("Time (integer steps)")
    ax.set_ylabel("Pressure")
    ax.legend()

    plt.show()

    az.summary(idata)

    az.plot_trace(idata)
    plt.show()




if __name__ == '__main__':
    wpt_cfg = common.load_config(input_data.events_yaml)['water_pressure_tests'][0]
    #bh_inv_cfg = yaml.load(bh_inv_cfg_yaml)
    borehole_section_inversion(wpt_cfg)
