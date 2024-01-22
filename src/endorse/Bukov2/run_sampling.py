import sys
import os.path

import matplotlib.pyplot as plt
from endorse.sa import sample, analyze
from endorse.bayes_orig import aux_functions
from endorse.bayes_orig import run_all as bayes_run_all
import endorse.Bukov2.sample_storage as sample_storage

import numpy as np
import csv


sensitivity_dirname = "sensitivity"
param_dirname = "parameters"
empty_hdf_dirname = "empty_hdfs"
pbs_job_dirname = "pbs_jobs"

sampled_data_dirname = "sampled_data"

class Analyze:
    def __init__(self, problem):
        self.problem = problem

    def single(self, output):
        return analyze.sobol(self.problem, output, calc_second_order=True, print_to_console=False)


def prepare_pbs_scripts(sens_config_dict, output_dir_in, np):
    endorse_root = sens_config_dict["rep_dir"]
    met = sens_config_dict["metacentrum"]

    pbs_dir = os.path.join(output_dir_in, pbs_job_dirname)
    aux_functions.force_mkdir(pbs_dir, force=True)

    def create_common_lines(id):
        name = met["name"] + "_" + id
        common_lines = [
            '#!/bin/bash',
            '#PBS -S /bin/bash',
            # '#PBS -l select=' + str(met["chunks"]) + ':ncpus=' + str(met["ncpus_per_chunk"]) + ':mem=' + met["memory"],
            # scratch: charon ssd: 346/20 = 17,3 GB
            '#PBS -l select=1:ncpus=1:mem=' + met["memory"] + ":scratch_local=17gb",
            # '#PBS -l place=scatter',
            '#PBS -l walltime=' + str(met["walltime"]),
            '#PBS -q ' + met["queue"],
            '#PBS -N ' + name,
            '#PBS -j oe',
            '#PBS -o ' + os.path.join(pbs_dir, name + '.out'),
            #'#PBS -e ' + os.path.join(pbs_dir, name + '.err'),
            '\n',
            'set -x',
            'export TMPDIR=$SCRATCHDIR',
            '\n# absolute path to output_dir',
            'output_dir="' + output_dir + '"',
            'workdir=$SCRATCHDIR',
            #'\n',
            #'SWRAP="' + met["swrap"] + '"',
            #'IMG_MPIEXEC="/usr/local/mpich_4.0.3/bin/mpirun"',
            #'SING_IMAGE="' + os.path.join(endorse_root, 'endorse.sif') + '"',
            #'\n',
            'cd $output_dir',
            #'SCRATCH_COPY=$output_dir',
            #'python3 $SWRAP/smpiexec_prepare.py -i $SING_IMAGE -s $SCRATCH_COPY -m $IMG_MPIEXEC'
        ]
        return common_lines

    pbs_file_list = []
    for n in range(np):
        id = solver_id(n)
        sensitivity_dir = os.path.join("$workdir", sensitivity_dirname)
        csv_file = "params_" + id + ".csv"

        sample_subdir = os.path.join(sensitivity_dir, "samples_" + id)
        sampled_data_out = os.path.join("$workdir", sampled_data_hdf(n))
        # prepare PBS script
        common_lines = create_common_lines(id)
        rsync_cmd = " ".join(["rsync -av",
                              #"--include " + os.path.join(sensitivity_dirname, empty_hdf_dirname, sampled_data_hdf(n)),
                              #"--include " + os.path.join(sensitivity_dirname, param_dirname, "params_" + id + ".csv"),
                              "--exclude *.h5",
                              "--exclude *.pdf",
                              "--exclude " + os.path.join(sensitivity_dirname, empty_hdf_dirname),
                              "--exclude " + os.path.join(sensitivity_dirname, param_dirname),
                              "--exclude " + os.path.join(sensitivity_dirname, pbs_job_dirname),
                              "$output_dir" + "/",
                              "$workdir"])
        lines = [
            *common_lines,
            '\n',
            rsync_cmd,
            ' '.join(['cp',
                      os.path.join(sensitivity_dirname, empty_hdf_dirname, sampled_data_hdf(n)),
                      "$workdir"]),
            ' '.join(['cp',
                      os.path.join(sensitivity_dirname, param_dirname, csv_file),
                      "$workdir"]),
            'cd $workdir',
            'pwd',
            'ls -la',
            '\n# finally gather the full command',
            os.path.join(endorse_root, "bin", "endorse-bayes") + " "
                + ' '.join(["-t", "set", "-o", "$workdir", "-p", csv_file, "-x", sample_subdir, "-s", id]),
            # 'zip -r samples.zip solver_*', # avoid 'bash: Argument list too long'
            # 'find . -name "solver_*" -print0 | xargs -0 tar -zcvf samples.tar.gz',
            # 'find . -name "solver_*" -print0 | xargs -0 rm -r',
            # '\n' + ' '.join(['tar', '-zcvf', 'samples_' + id + '.tar.gz', sample_subdir]),
            # ' '.join(['rm', '-r', sample_subdir]),
            'ls -la',
            'mkdir -p $output_dir/sampled_data',
            'time cp ' + sampled_data_out + ' $output_dir/sampled_data',
            #'time cp -r sensitivity $output_dir',
            'clean_scratch',
            'echo "FINISHED"'
        ]
        pbs_file = os.path.join(pbs_dir, "pbs_job_" + id + ".sh")
        with open(pbs_file, 'w') as f:
            f.write('\n'.join(lines))
        pbs_file_list.append(pbs_file)

    return pbs_file_list


def solver_id(i):
    return str(i).zfill(2)


def sampled_data_hdf(i):
    return 'sampled_data_' + solver_id(i) + '.h5'


def prepare_sets_of_params(parameters, output_dir_in, n_processes, par_names):
    no_samples, no_parameters = np.shape(parameters)
    rows_per_file = no_samples // n_processes
    rem = no_samples % n_processes

    param_dir = os.path.join(output_dir_in, param_dirname)
    aux_functions.force_mkdir(param_dir, force=True)
    empty_hdf_dir = os.path.join(output_dir_in, empty_hdf_dirname)
    aux_functions.force_mkdir(empty_hdf_dir, force=True)

    sample_idx = 0
    off_start = 0
    off_end = 0
    for i in range(n_processes):
        off_start = off_end
        off_end = off_end + rows_per_file
        # add sample while there is still remainder after rows_per_file division
        if rem > 0:
            off_end = off_end + 1
            rem = rem - 1
        subset_matrix = parameters[off_start:off_end, :]

        param_file = os.path.join(param_dir, "params_" + solver_id(i) + ".csv")
        with open(param_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['idx', *par_names])
            for row in subset_matrix:
                writer.writerow([sample_idx, *row])
                sample_idx = sample_idx+1

        # Prepare HDF, write parameters
        output_file = os.path.join(empty_hdf_dir, sampled_data_hdf(i))
        n_params = parameters.shape[0]
        n_times = config_dict["sample_shape"][0]
        n_elements = config_dict["sample_shape"][1]
        sample_storage.create_chunked_dataset(output_file,
                                              shape=(n_params, n_times, n_elements),
                                              chunks=(1, 1, n_elements))
        sample_storage.append_new_dataset(output_file, "parameters", parameters)

    # for i, mat in enumerate(sub_parameters):
    #     output_file = f"parameters_{str(i+1).zfill(2)}.npy"
    #     np.save(output_file, mat)
    #     print(f"Saved {output_file}")


def fpermeability(k0, eps, delta, gamma, sigma0, a, b, c, sigma_m, sigma_tres):
    sigma_vm = 170/120*np.abs(sigma_m)
    x = 0
    y = 30
    z = 30
    lin = (1 + 0.1*(a * x / 35 + b * y / 30 + c * z / 30))
    kr = 1/eps *k0
    k = kr + delta * np.exp(np.log((k0 - kr) / delta) * sigma_m / sigma0)
    return np.where(sigma_vm < sigma_tres, k, k * np.exp(gamma * (sigma_vm - sigma_tres)/sigma_tres)) * lin


def fconductivity(perm):
    return 1000*9.81/0.001*perm


def plot_conductivity(config_dict, params):
    import matplotlib.pyplot as plt
    parnames = [p["name"] for p in config_dict["parameters"]]

    # init_sigma_m = -(42 + 19 + 14)*1e6/3
    sigma_tres = 55e6
    # sigma_mean is positive for graph, but goes negative into the permeability function
    sigma_mean = np.linspace(-7e6,120e6,200)
    init_sigma_m = -(params[:,parnames.index("init_stress_x")] +
                     params[:,parnames.index("init_stress_y")] +
                     params[:,parnames.index("init_stress_z")]) / 3
    # cond = conductivity(p[parnames.index("perm_k0")],
    #                     p[parnames.index("perm_eps")],
    #                     p[parnames.index("perm_delta")],
    #                     p[parnames.index("perm_gamma")],
    #                     init_sigma_m,
    #                     p[parnames.index("conductivity_a")],
    #                     p[parnames.index("conductivity_b")],
    #                     p[parnames.index("conductivity_c")],
    permeability = np.zeros((params.shape[0],len(sigma_mean)))

    # fig_cond, ax_cond = plt.subplots()
    plt.rcParams['text.usetex'] = True
    fig, ax1 = plt.subplots()
    xax = sigma_mean / 1e6

    for i in range(params.shape[0]):
        p = params[i,:]
        # if p[4] >= p[5]:
        #     continue
        # p[[4,5]] = p[[5,4]]
        perm = fpermeability(p[parnames.index("perm_k0")],
                            p[parnames.index("perm_eps")],
                            p[parnames.index("perm_delta")],
                            p[parnames.index("perm_gamma")],
                            init_sigma_m[i],
                            p[parnames.index("conductivity_a")],
                            p[parnames.index("conductivity_b")],
                            p[parnames.index("conductivity_c")],
                            -sigma_mean, sigma_tres)
        permeability[i,:] = perm
        # cond = fconductivity(perm)
        # ax_cond.plot(sigma_mean/1e6, cond)
        # ax_cond.scatter(init_sigma_m[i]/1e6, fconductivity(p[parnames.index("perm_k0")]))

    # plot N random samples:
    from random import randrange
    rids = [randrange(0, params.shape[0]) for i in range(0, 50)]
    for rid in rids:
        ax1.plot(xax, permeability[rid, :], linewidth=0.5, color="black", alpha=0.2)

    # plot permeability range
    q_mean = np.mean(permeability, axis=0)
    q_up = np.quantile(permeability, q=0.95, axis=0)
    q_down = np.quantile(permeability, q=0.05, axis=0)
    # q_99 = np.max(permeability, axis=0)
    # q_01 = np.min(permeability, axis=0)

    ax1.set_xlabel(r'$\sigma_m \mathrm{[MPa]}$')
    ax1.set_ylabel(r'$\kappa \mathrm{[m^2]}$', color='black')
    ax1.fill_between(xax, q_down, q_up,
                     color="red", alpha=0.2, label=None)

    # plot vertical lines of interest
    def add_vline(x, label):
        ax1.axvline(x=x / 1e6, linewidth=0.5)
        ax1.text(x / 1e6 + 0.75, 0.015, label, rotation=0, transform=ax1.get_xaxis_transform())
    def add_hline(y, label):
        ax1.axhline(y=y, linewidth=0.5)
        ax1.text(0.01, y*1.4, label, rotation=0, transform=ax1.get_yaxis_transform())

    add_vline(sigma_tres, r'$\sigma_{\mathrm{VM}c}$')
    add_vline(0, '')
    # add_vline(np.min(-init_sigma_m), r'$\sigma_0$')
    init_sigma_m_mean = np.mean(-init_sigma_m)
    add_vline(init_sigma_m_mean, r'$\sigma_{m0}$')
    # add_vline(np.max(-init_sigma_m), r'$\sigma_0$')

    # plot permeability
    ax1.plot(xax, q_mean, color="red")

    def add_annotated_point(x,y,label):
        ax1.scatter(x, y, facecolors='none', edgecolors='limegreen', marker='o',
                    zorder=100)
        ax1.annotate(label, xy=(x + 2, y),
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none',
                               boxstyle='round,pad=0.2,rounding_size=0.2'))

    perm_delta_mean = np.mean(params[:, parnames.index("perm_delta")])
    add_annotated_point(0, perm_delta_mean, r'$[0, \kappa_\delta]$')
    # ax1.scatter(0, perm_delta_mean, facecolors='none', edgecolors='limegreen', marker='o', zorder=100)
    # ax1.text(0, perm_delta_mean, r'$[0, \kappa_\delta]$', rotation=0, transform=ax1.get_yaxis_transform())
    # ax1.annotate(r'$[0, \kappa_\delta]$', xy=(0+2, perm_delta_mean),
    #              bbox=dict(facecolor='white', alpha=0.4, edgecolor='none'))
    perm_k0_mean = np.mean(params[:, parnames.index("perm_k0")])
    add_annotated_point(init_sigma_m_mean/1e6, perm_k0_mean, r'$[\sigma_{m0}, \kappa_0]$')

    perm_kr_mean = np.mean(params[:, parnames.index("perm_k0")]/params[:, parnames.index("perm_eps")])
    # perm_kr_mean = np.mean(params[:, parnames.index("perm_k0")]) / np.mean(params[:, parnames.index("perm_eps")])
    # print(perm_kr_mean)
    # ax1.scatter(init_sigma_m_mean / 1e6, perm_kr_mean, facecolors='none', edgecolors='limegreen', marker='o',
    #             zorder=100)
    add_hline(perm_kr_mean, label=r'$\kappa_r$')

    ax1.text(0.52,0.9,
             # r'$\sigma_{VM}=\frac{170}{120}\sigma_m$\\$\kappa_r=\kappa_0/\epsilon,\qquad \epsilon>1.1$',
             r'\begin{eqnarray*} \sigma_{\mathrm{VM}c} &=& \frac{170}{120}\sigma_m \\'
             r'\kappa_r &=& \kappa_0/\epsilon,\qquad \epsilon>1.1 \end{eqnarray*}',
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5,rounding_size=0.2', linewidth=0.5)
             )

    # finialize figure
    ax1.set_xlim(np.min(xax), np.max(xax))
    ax1.set_yscale('log')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig("permeability.pdf")

    # ax_cond.set_yscale('log')
    # fig_cond.savefig('conductivity.pdf')

    # plt.scatter(range(params.shape[0]), params[:,5])
    # plt.savefig('eps.pdf')


if __name__ == "__main__":

    # default parameters
    output_dir = None

    len_argv = len(sys.argv)
    assert len_argv > 1, "Specify output dir."
    if len_argv > 1:
        output_dir = os.path.abspath(sys.argv[1])

    # aux_functions.force_mkdir(output_dir, force=True)
    # shutil.copyfile("../test_data/config_sim_A04hm_V1_03.yaml", os.path.join(output_dir, "config.yaml"))
    # setup paths and directories
    config_dict = bayes_run_all.setup(output_dir, can_overwrite=False, clean=False)
    # add repository dir
    config_dict["rep_dir"] = os.path.abspath(os.path.join(config_dict["script_dir"], "../../.."))

    # Define the problem for SALib
    # Bayes Inversion borehole_V1/sim_A04hm_V1_04_20230713a
    params = config_dict["parameters"]
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
        'bounds': [p["bounds"] for p in params]
    }

    print(problem)

    # Generate Saltelli samples
    param_values = sample.saltelli(problem, config_dict["n_samples"], calc_second_order=config_dict["second_order_sa"])
    # param_values = sample.sobol(problem, n_samples, calc_second_order=True)
    print(param_values.shape)

    # plot requires LaTeX installed
    plot_conductivity(config_dict, param_values)
    # exit(0)

    sensitivity_dir = os.path.join(output_dir, sensitivity_dirname)
    aux_functions.force_mkdir(sensitivity_dir, force=True)

    # plan sample parameters a prepare them in CSV
    prepare_sets_of_params(param_values, sensitivity_dir, config_dict["n_processes"], problem["names"])
    # exit(0)

    # plan parallel sampling, prepare PBS jobs
    pbs_file_list = prepare_pbs_scripts(config_dict, sensitivity_dir, config_dict["n_processes"])
