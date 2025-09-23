import numpy as np
import arviz as az
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, SymmetricalLogLocator, LogFormatter, FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, lognorm
import logging


def get_generic_name(idata):
    return f"WPT_{idata.attrs['borehole'][3:]}_{idata.attrs['section']}_{idata.attrs['year']}_{idata.attrs['month']:02d}"

def plot_idata(idata):

    burnin = idata.attrs["burnin"] if "burnin" in idata.attrs else 0
    idata_cut = idata.sel(draw=slice(burnin, None)) # cut first 2000 draws

    #az.style.use("arviz-doc")
    az.rcParams["plot.max_subplots"] = 50

    generic_name = get_generic_name(idata)

    #save_plots_pdf_pages("observe_plot.pdf", plot_observe(idata_cut, bins=150, generic_name=generic_name))
    #plt.savefig("observe_plot.pdf", dpi=300)
    #plt.show()

    print(az.summary(idata_cut))

    # plot trace and force axis to use scientitic notation
    #plot_trace_modified(idata_cut, figsize=(16, 36), generic_name=generic_name)
    #plt.savefig("trace_plot.pdf", dpi=300)

    # plot posterior distributions and corresponding prior distributions
    #plot_posterior_modified(idata_cut, figsize=(16, 18), generic_name=generic_name)

    #plt.savefig("posterior_plot.pdf", dpi=300)

    #save_plots_pdf_pages("likelihood_plot.pdf", plot_likelihood(idata_cut, generic_name=generic_name))

    plot_merged(idata_cut, idata)

def plot_observe(idata, ax=None, bins=100, generic_name="WPT", kind="both"):
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 18), nrows=2)


    borehole = idata.attrs["borehole"]
    section = idata.attrs["section"]

    if idata.sample_stats.attrs["observed_pressure"] is None:
        logging.warning("No observed data found in InferenceData object.")
        return ax

    if kind not in ["both", "pressure", "flow"]:
        logging.warning("Unknown type of observe plot")
        return ax

    pressure_output = idata.sample_stats.attrs["observed_pressure"]
    pressure_output_extended = idata.sample_stats.attrs["observed_extended"]
    pressure_output_sigma = idata.sample_stats.attrs["observed_pressure_sigma"]

    flow_rate_observed = idata.sample_stats.attrs["observed_flow"]
    flow_rate_sigma = idata.sample_stats.attrs["observed_flow_sigma"]

    observe = idata.posterior_predictive
    pressure_vars = sorted([v for v in observe.data_vars if v.startswith("obs_") and v not in ["obs_0"]],
                      key=lambda s: int(s.split("_", 1)[1]))

    pressure_list = [observe[v] for v in pressure_vars]
    observe_arr = xr.concat(pressure_list, dim="time")

    flow_values = observe["obs_0"].values.flatten()
    flow_values = np.clip(flow_values, -20, 20)

    chains = observe_arr.sizes["chain"]
    draws = observe_arr.sizes["draw"]

    observe_arr = observe_arr.stack(flat_dim=("chain", "draw", "time")).reset_index("flat_dim", drop=True)
    observe_length = len(pressure_list)

    #print(idata.sample_stats)
    likelihood_data = idata.sample_stats["likelihood"].stack(flat_dim=("chain", "draw")).reset_index("flat_dim", drop=True).values
    best_fit_idx = np.argmax(likelihood_data)
    #posterior_data = idata.posterior["K"].stack(flat_dim=("chain", "draw")).reset_index("flat_dim", drop=True).values
    #print(posterior_data[best_fit_idx])
    best_fit = observe_arr.isel(flat_dim=slice(best_fit_idx * observe_length, (best_fit_idx + 1) * observe_length)).values

    figs = []

    if kind in ["both", "pressure"]:
        if kind in ["both"]:
            ax_pressure = ax[0]
        else:
            ax_pressure = ax

        hist2d_x = np.tile(np.arange(observe_length), chains * draws)

        ax_pressure.hist2d(hist2d_x, observe_arr.values, bins=[observe_length, bins], cmap="viridis", cmin=1e-7)
        origin_offset = observe_length - len(pressure_output_extended) # compute origin offset to plot extended data
        ax_pressure.plot(np.arange(origin_offset, observe_length), pressure_output_extended, "r-", label="Predicted observation (extended)", lw=1)
        ax_pressure.plot(np.arange(observe_length), best_fit, "k--", label="Best fit", lw=1)
        ax_minima =  [
            np.min([observe_arr.min(), pressure_output_extended.min()]),
            np.max([observe_arr.max(), pressure_output_extended.max()])
        ]
        ax_pressure.set_ylim(ax_minima)

        ax_pressure.set_xlim([origin_offset, observe_length])
        ax_pressure.set_xlabel("Time (integer steps)")
        ax_pressure.set_ylabel("Pressure")
        ax_pressure.legend()
        plt.suptitle(f"{generic_name} - distibution of pressure series values")
        plt.colorbar(ax.collections[0], ax=ax, label="Counts")
        figs.append(ax.get_figure())

    if kind in ["both", "flow"]:
        if kind in ["both"]:
            ax_flow = ax[1]
        else:
            ax_flow = ax
        plt.suptitle(f"{generic_name} - flow rate distribution")
        ax_flow.set_xlabel("Flow rate [m^3/s]")
        ax_flow.set_ylabel("Counts")
        ax_flow.xaxis.set_major_formatter(FuncFormatter(exp_formatter))

        # plot flow rate distribution
        counts, bin_edges, _ = ax_flow.hist(flow_values, alpha=0.7, bins=bins, color="orange", label="Flow rate fit")
        total_area = (bin_edges[1] - bin_edges[0]) * np.sum(counts)

        # plot observed flow rate distribution
        if idata.attrs["plot_observed_flow"]:
            observed_xvals = np.linspace(flow_rate_observed - 3 * flow_rate_sigma, flow_rate_observed + 3 * flow_rate_sigma, bins)
            observed_yvals = norm.pdf(observed_xvals, flow_rate_observed, flow_rate_sigma)
            ax_flow.plot(observed_xvals, observed_yvals * total_area, color="red", linestyle="dashed", label="Observed flow rate distribution")
        
        ax_flow.legend()
        figs.append(ax_flow.get_figure())

    return figs

def plot_trace_modified(idata, generic_name="WPT", *args, **kwargs):
    axes = az.plot_trace(idata, *args, **kwargs)
    borehole = idata.attrs["borehole"]
    section = idata.attrs["section"]
    plt.suptitle(f"{generic_name} - trace plot")

    for ax_row in axes:
        for ax in ax_row:
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

    fig = plt.gcf()
    fig.set_constrained_layout(True)

    return axes

def exp_formatter(x, pos):
     return f"{np.power(10, x):.1e}"

def plot_posterior_modified(idata, generic_name="WPT", *args, **kwargs):
    axes = az.plot_posterior(idata, *args, **kwargs)
    # constrained layout for better spacing
    fig = plt.gcf()
    fig.set_constrained_layout(True)

    borehole = idata.attrs["borehole"]
    section = idata.attrs["section"]
    plt.suptitle(f"{generic_name} - posterior distributions")
    
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
        if not isinstance(axes, np.ndarray):
            param_names = list(idata.posterior.data_vars)
            idx = param_names.index(kwargs["var_names"][0])
            mean = prior_mean[idx]
            sd = prior_sd[idx]
            print(sd)
            xvals = np.linspace(mean - 3 * sd, mean + 3 * sd, 100)
            yvals = norm.pdf(xvals, mean, sd)
            axes.plot(xvals, yvals, color="red", linestyle="dashed", label="Původní odhad")

            if axes.get_title() not in ["p_far"]:
                axes.xaxis.set_major_formatter(FuncFormatter(exp_formatter))
            else:
                axes.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axes.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
            #ax.set_xscale('symlog', linthresh=1, base=10)
            axes.tick_params(axis="x", labelsize=6)
            return axes

        for x, ax_row in enumerate(axes):
            if isinstance(ax_row, np.ndarray):
                for y, ax in enumerate(ax_row):
                    idx = x * len(ax_row) + y
                    # if empty axis, skip
                    if not (ax.lines or ax.images or ax.collections or ax.patches):
                        continue
                    mean = prior_mean[idx]
                    sd = prior_sd[idx]
                    print(sd)
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
                print(sd)
                xvals = np.linspace(mean - 3 * sd, mean + 3 * sd, 100)
                yvals = norm.pdf(xvals, mean, sd)
                ax_row.plot(xvals, yvals, color="red", linestyle="dashed", label="Původní odhad")

    # set scientific notation for axes
    for ax_row in axes:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                if ax.get_title() not in ["p_far"]:
                    ax.xaxis.set_major_formatter(FuncFormatter(exp_formatter))
                else:
                    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
                #ax.set_xscale('symlog', linthresh=1, base=10)
                ax.tick_params(axis="x", labelsize=6)
        else:
            # single axis
            if ax_row.get_title() not in ["p_far"]:
                ax_row.xaxis.set_major_formatter(FuncFormatter(exp_formatter))
            else:
                ax_row.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax_row.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
            #ax.set_xscale('symlog', linthresh=1, base=10)
            #ax_row.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax_row.tick_params(axis="x", labelsize=6)

    return axes

def plot_likelihood(idata: az.InferenceData, idata_uncut: az.InferenceData, cutoff=None, generic_name="WPT") -> list:
    if cutoff is None:
        cutoff = -1e8
    # cut data
    draws = idata.posterior.sizes["draw"]
    chains = idata.posterior.sizes["chain"]
    likelihoods = np.clip(idata["sample_stats"]["likelihood"], cutoff, None)
    prior = np.clip(idata["sample_stats"]["prior"], cutoff, None)
    posterior = np.clip(idata["sample_stats"]["posterior"], cutoff, None)
    datasets = [likelihoods, prior, posterior]
    labels = ["log-likelihood", "log-prior", "log-posterior"]
    x_axis = np.arange(0, draws)

    # uncut data
    draws_uncut = idata_uncut.posterior.sizes["draw"]
    chains_uncut = idata_uncut.posterior.sizes["chain"]
    likelihoods_uncut = np.clip(idata_uncut["sample_stats"]["likelihood"], cutoff, None)
    prior_uncut = np.clip(idata_uncut["sample_stats"]["prior"], cutoff, None)
    posterior_uncut = np.clip(idata_uncut["sample_stats"]["posterior"], cutoff, None)
    datasets_uncut = [likelihoods_uncut, prior_uncut, posterior_uncut]
    x_axis_uncut = np.arange(0, draws_uncut)

    cutoff = draws_uncut - draws # where to display cutoff line

    figs = []

    for dataset, dataset_uncut, label in zip(datasets, datasets_uncut, labels):
        fig_progression, axes_progression = plt.subplots(2, 1, figsize=(16, 9))
        fig_progression.suptitle(f"{generic_name} - progression of {label} (values under {cutoff} cut off)")
        axes_progression[0].set_xlabel("Iteration in chain")
        axes_progression[0].set_ylabel(f"{label}")
        for chain in np.arange(0, chains):
            axes_progression[0].plot(x_axis, dataset[chain, :], label=f"Chain {chain}")
        #axes_progression[0].axvline(x=0, color='red', linestyle='--', label='Cutoff point')
        axes_progression[0].legend(ncol=2, loc="lower right")
        axes_progression[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes_progression[0].set_yscale('symlog', linthresh=1)

        dataspan = np.log10(-dataset.min()) - np.log10(-dataset.max()) # negative values
        print(dataspan)
        if dataspan >= 1.5:
            subs_major = [1, 5]
            subs_minor = np.arange(1, 10)
        elif dataspan >= 0.6:
            subs_major = [1, 3, 5, 7]
            subs_minor = np.arange(1, 10)
        elif dataspan >= 0.3:
            subs_major = np.arange(1, 10, 0.5)
            subs_minor = np.arange(1, 10, 0.1)
        else:
            subs_major = np.arange(1, 10, 0.2)
            subs_minor = np.arange(1, 10, 0.05)

        axes_progression[0].yaxis.set_major_locator(SymmetricalLogLocator(base=10.0, subs=subs_major, linthresh=1))
        axes_progression[0].yaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, subs=subs_minor, linthresh=1))
        axes_progression[0].yaxis.set_major_formatter(get_symlog_formatter())

        # mean = np.mean(dataset, axis=0)
        # median = np.median(dataset, axis=0)
        # min = np.min(dataset, axis=0)
        # axes_progression[1].set_xlabel("Iteration in chain")
        # axes_progression[1].set_ylabel(f"")
        # axes_progression[1].plot(x_axis, mean, label=f"Mean {label}")
        # axes_progression[1].plot(x_axis, median, label=f"Median {label}")
        # axes_progression[1].plot(x_axis, min, label=f"Minimum {label}")
        # axes_progression[1].legend(ncol=2, loc="lower right")
        # axes_progression[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        # axes_progression[1].set_yscale('symlog', linthresh=1)
        # axes_progression[1].yaxis.set_major_locator(SymmetricalLogLocator(base=10.0, subs=subs_major, linthresh=1))
        # axes_progression[1].yaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, subs=subs_minor, linthresh=1))
        # axes_progression[1].yaxis.set_major_formatter(get_symlog_formatter())

        # uncut data plot
        axes_progression[1].set_xlabel("Iteration in chain (uncut data)")
        axes_progression[1].set_ylabel(f"{label}")
        for chain in np.arange(0, chains_uncut):
            axes_progression[1].plot(x_axis_uncut, dataset_uncut[chain, :], label=f"Chain {chain}")
        axes_progression[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes_progression[1].set_yscale('symlog', linthresh=1)
        axes_progression[1].axvline(x=cutoff, color='red', linestyle='--', label='Cutoff point')
        axes_progression[1].legend(ncol=2, loc="lower right")

        axes_progression[1].yaxis.set_major_locator(SymmetricalLogLocator(base=10.0, subs=subs_major, linthresh=1))
        axes_progression[1].yaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, subs=subs_minor, linthresh=1))
        axes_progression[1].yaxis.set_major_formatter(get_symlog_formatter())


        figs += [fig_progression]

        fig_hist, axes_hist = plt.subplots(figsize=(16, 9))
        fig_hist.suptitle(f"{generic_name} - histogram {label} (values below {cutoff} cut off)")
        axes_hist.set_xlabel(f"{label}")
        axes_hist.set_ylabel("Počet")
        logbins = np.multiply(np.logspace(np.log10(-dataset.max()), np.log10(-dataset.min()), 100), -1)
        axes_hist.hist(dataset.values.flatten(), bins=logbins[::-1])
        axes_hist.set_xscale('symlog', linthresh=1)

        dataspan_uncut = np.log10(-dataset_uncut.min()) - np.log10(-dataset_uncut.max()) # negative values
        print(dataspan_uncut)
        if dataspan >= 1.5:
            subs_major = [1, 5]
            subs_minor = np.arange(1, 10)
        elif dataspan >= 0.6:
            subs_major = [1, 3, 5, 7]
            subs_minor = np.arange(1, 10)
        elif dataspan >= 0.3:
            subs_major = np.arange(1, 10, 0.5)
            subs_minor = np.arange(1, 10, 0.1)
        else:
            subs_major = np.arange(1, 10, 0.2)
            subs_minor = np.arange(1, 10, 0.05)

        axes_hist.xaxis.set_major_locator(SymmetricalLogLocator(base=10.0, subs=subs_major, linthresh=1))
        axes_hist.xaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, subs=subs_minor, linthresh=1))
        axes_hist.xaxis.set_major_formatter(get_symlog_formatter())

        figs += [fig_hist]

    return figs

def get_symlog_formatter():
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    return formatter

def save_plots_pdf_pages(
        filename: str,
        figs: list) -> None:

    if not figs:
        return

    try:
        with PdfPages(filename) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)
        logging.info("Succesfully saved plot %s", filename)
    except:
        logging.error("Failed to save plot at %s", filename)

def plot_merged(idata, idata_uncut):

    az.rcParams["plot.max_subplots"] = 400
    generic_name = get_generic_name(idata)

    # order: 
    # observe (both)
    # likelihood (just chains, just likelihood(?))
    # posterior + trace (right side)
    # new posterior plot (2d hist)
    # pair (might not fit)
    # all others

    figs = []
    figs += plot_observe(idata, bins=150, generic_name=generic_name)
    likelihood_figs = plot_likelihood(idata, idata_uncut, generic_name=generic_name) # order - likelihood, prior, posterior
    figs += [likelihood_figs[0]]

    trace_ax = plot_trace_modified(idata, figsize=(16, 36), generic_name=generic_name)
    trace_fig = trace_ax[0, 0].figure

    k_params = [v for v in idata.posterior.data_vars if v.startswith("log_k")]
    E_params = [v for v in idata.posterior.data_vars if v.startswith("log_E")]

    k_vals = np.concatenate([idata.posterior[v].values.flatten() for v in k_params])
    E_vals = np.concatenate([idata.posterior[v].values.flatten() for v in E_params])

    k_min, k_max = np.min(k_vals), np.max(k_vals)
    E_min, E_max = np.min(E_vals), np.max(E_vals)

    for i, var_name in enumerate(idata.posterior.data_vars):
        trace_ax[i, 0].clear()
        with plt.rc_context({'axes.labelsize': 12, "axes.titlesize": 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12}):
            plot_posterior_modified(idata, var_names=[var_name], generic_name=generic_name, ax=trace_ax[i, 0])
        if var_name not in ["p_far"]:
            trace_ax[i, 1].yaxis.set_major_formatter(FuncFormatter(exp_formatter))
        
        trace_ax[i, 1].set_ylim([k_min, k_max] if var_name.startswith("log_k") else [E_min, E_max] if var_name.startswith("log_E") else None)
    
    figs += [trace_fig]

    poserior_ax = plot_posterior_hist_2d(idata, generic_name=generic_name, figsize=(16, 18))
    poserior_fig = poserior_ax[0].figure
    figs += [poserior_fig]

    pair_ax = az.plot_pair(idata, figsize=(16, 16), marginals=True, kind="kde")
    pair_fig = pair_ax[0, 0].figure
    figs += [pair_fig]

    figs += likelihood_figs[1:] # add prior and posterior likelihood histograms

    save_plots_pdf_pages(f"{generic_name}_summary.pdf", figs)

def plot_posterior_hist_2d(idata, generic_name="WPT", *args, **kwargs):

    if "figsize" in kwargs:
        figsize = kwargs.pop("figsize")
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f"{generic_name} - posterior distributions of log_k and log_E")

    posterior = idata.posterior
    k_params = [idx for idx in posterior.data_vars if idx.startswith("log_k_")]
    k_values = posterior[k_params].to_array().values.reshape(len(k_params), -1)
    E_params = [idx for idx in posterior.data_vars if idx.startswith("log_E_")]
    E_values = posterior[E_params].to_array().values.reshape(len(E_params), -1)

    n_samples, n_params = k_values.shape

    x = np.repeat(np.arange(n_samples), n_params)

    y = k_values.flatten()
    axes[0].hist2d(x, y, bins=[n_samples, 100], cmap="viridis", cmin=1e-7)
    axes[0].set_xlabel("index of k parameter")
    axes[0].set_ylabel("parameter value")
    
    y = E_values.flatten()
    axes[1].hist2d(x, y, bins=[n_samples, 100], cmap="viridis", cmin=1e-7)
    axes[1].set_xlabel("index of E parameter")
    axes[1].set_ylabel("parameter value")

    return axes