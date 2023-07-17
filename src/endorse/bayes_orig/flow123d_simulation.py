import os
import subprocess
import time

import numpy as np
import itertools
import collections
import shutil
import csv
import yaml
from typing import List
import traceback

import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

from endorse import common


def generate_time_axis(config_dict):
    end_time = float(config_dict["end_time"])
    output_times = config_dict["output_times"]

    # create time axis
    times = []
    for dt in output_times:
        b = float(dt["begin"])
        s = float(dt["step"])
        e = float(dt["end"])
        times.extend(np.arange(b, e, s))
    times.append(end_time)
    return times


class endorse_2Dtest():

    def __init__(self, config, clean):

        # TODO: set work dir
        self.work_dir = config["work_dir"]
        self.clean = clean
        self._config = config
        self.sample_dir = ""
        self.sample_counter = -1

    def set_parameters(self, data_par):
        param_list = self._config["surrDAMH_parameters"]["parameters"]
        assert(len(data_par) == len(param_list))

        for idx, param in enumerate(param_list):
            pname = param["name"]
            assert(pname in self._config["hm_params"])
            self._config["hm_params"][pname] = data_par[idx]

    def get_observations(self):
        try:
            print("get observations from flow_wrapper")
            res = self.calculate(self._config)
            return res
        except ValueError:
            print("flow_wrapper failed for unknown reason.")
            return -1000, []

    def calculate(self, config_dict):
        """
        The program changes to <work_dir> directory.
        does all the data preparation, passing
        running simulation
        extracting results
        """

        # create sample dir
        self.sample_counter = self.sample_counter + 1
        self.sample_dir = os.path.join(config_dict["work_dir"],
                                       "solver_" + str(config_dict["solver_id"]).zfill(2) +
                                       "_sample_" + str(self.sample_counter).zfill(3))
        os.makedirs(self.sample_dir, mode=0o775, exist_ok=True)
        os.chdir(self.sample_dir)

        print("=========================== RUNNING CALCULATION " +
              "solver {} ".format(config_dict["solver_id"]).zfill(2) +
              "sample {} ===========================".format(self.sample_counter).zfill(3),
              flush=True)
        print(self.sample_dir)

        # collect only
        if config_dict["collect_only"]:
            return 2, self.collect_results(config_dict)

        print("Creating mesh...")
        # comp_mesh = self.prepare_mesh(config_dict, cut_tunnel=False)
        comp_mesh = self.prepare_mesh(config_dict, cut_tunnel=True)

        mesh_bn = os.path.basename(comp_mesh)
        config_dict["hm_params"]["mesh"] = mesh_bn

        # endorse_2Dtest.read_physical_names(config_dict, comp_mesh)
        print("Creating mesh...finished")

        if config_dict["mesh_only"]:
            return -10, []  # tag, value_list

        # endorse_2Dtest.prepare_hm_input(config_dict)
        print("Running Flow123d - HM...")
        hm_succeed = self.call_flow(config_dict, 'hm_params', result_files=["flow_observe.yaml"])

        # params = config_dict.tsx_hm_model.hm_params
        # template = os.path.join(flow123d_inputs_path, params.input_template)
        # self.flow_output = common.call_flow(config_dict.flow_env, template, params)

        if not hm_succeed:
            # raise Exception("HM model failed.")
            # "Flow123d failed (wrong input or solver diverged)"
            print("Flow123d failed.")
            return -1, []  # tag, value_list
        print("Running Flow123d - HM...finished")

        if self._config["make_plots"]:
            try:
                self.observe_time_plot(config_dict)
            except:
                print("Making plot of sample results failed:")
                traceback.print_exc()
                return -2, []

        print("Finished computation")

        # collected_values = self.collect_results(config_dict)
        # print("Sample results collected.")
        # return 1, collected_values  # tag, value_list

        try:
            collected_values = self.collect_results(config_dict)
            print("Sample results collected.")
            return 1, collected_values  # tag, value_list
        except:
            print("Collecting sample results failed:")
            traceback.print_exc()
            return -3, []

    # def check_data(self, data, minimum, maximum):
    #     n_times = len(endorse_2Dtest.result_format()[0].times)
    #     if len(data) != n_times:
    #         raise Exception("Data not corresponding with time axis.")
    #
    #     if np.isnan(np.sum(data)):
    #         raise Exception("NaN present in extracted data.")
    #
    #     min = np.amin(data)
    #     if min < minimum:
    #         raise Exception("Data out of given range [min].")
    #     max = np.amax(data)
    #     if max > maximum:
    #         raise Exception("Data out of given range [max].")

    def collect_results(self, config_dict):
        output_dir = config_dict["hm_params"]["output_dir"]
        pressure_points2collect = config_dict["surrDAMH_parameters"]["observe_points"]
        cond_points2collect = config_dict["surrDAMH_parameters"]["conductivity_observe_points"]

        values = np.empty((0,))

        # the times defined in input
        times = np.array(generate_time_axis(config_dict))
        with open(os.path.join(output_dir, "flow_observe.yaml"), "r") as f:
            loaded_yaml = yaml.load(f, yaml.CSafeLoader)

            vals = self.get_from_observe(loaded_yaml, pressure_points2collect, 'pressure_p0', times)
            values = np.concatenate((values, vals), axis=None)

            vals = self.get_from_observe(loaded_yaml, cond_points2collect, 'conductivity', times[-1])
            vals = np.log10(vals)  # consider log10!
            values = np.concatenate((values, vals), axis=None)

        if config_dict["clean_sample_dir"]:
            shutil.rmtree(self.sample_dir)

        # flatten to format: [Point0_all_all_times, Point1_all_all_times, Point2_all_all_times, ...]
        res = values.flatten()
        return res

    def get_from_observe(self, observe_dict, point_names, field_name, select_times=None):
        points = observe_dict['points']
        all_point_names = [p["name"] for p in points]
        print('all_point_names', all_point_names)
        print('point_names', point_names)
        points2collect_indices = []
        for p2c in point_names:
            tmp = [i for i, pn in enumerate(all_point_names) if pn == p2c]
            assert len(tmp) == 1
            points2collect_indices.append(tmp[0])

        print("Collecting results for observe points: ", point_names)
        data = observe_dict['data']
        data_values = np.array([d[field_name] for d in data])
        values = data_values[:, points2collect_indices]
        obs_times = np.array([d["time"] for d in data]).transpose()

        if select_times is not None:
            # check that observe data are computed at all times of defined time axis
            all_times_computed = np.alltrue(np.isin(select_times, obs_times))
            if not all_times_computed:
                raise Exception("Observe data not computed at all times as defined by input!")
            # skip the times not specified in input
            t_indices = np.isin(obs_times, select_times).nonzero()
            values = values[t_indices]
        values = values.transpose()

        # if "smooth_factor" in config_dict.keys():
        #     smooth_factor = config_dict["smooth_factor"]
        #     for i in range(len(values)):
        #         values[i] = self.smooth_ode(times, values[i], smooth_factor)

        return values

    def call_flow(self, config_dict, param_key, result_files):
        """
        Redirect sstdout and sterr, return true on succesfull run.
        :param result_files: Files to be computed - skip computation if already exist.
        :param param_key: config dict parameters key
        :param config_dict:
        :return:
        """

        status = False
        params = config_dict[param_key]
        fname = params["in_file"]
        arguments = ['env', '-i']
        arguments.extend(config_dict["local"]["flow_executable"].copy())
        output_dir = "output_" + fname
        arguments.extend(['--output_dir', output_dir, fname + ".yaml"])
        config_dict[param_key]["output_dir"] = output_dir

        stdout_path = fname + "_stdout"
        stderr_path = fname + "_stderr"
        if all([os.path.isfile(os.path.join(output_dir, f)) for f in result_files]):
            status = True
            completed = subprocess.CompletedProcess(args=arguments, returncode=0)
        else:
            common.substitute_placeholders(
                os.path.join(config_dict["common_files_dir"],
                             fname + '_tmpl.yaml'),
                fname + '.yaml',
                params)

            print("Running: ", " ".join(arguments))
            with open(stdout_path, "w") as stdout:
                with open(stderr_path, "w") as stderr:
                    completed = subprocess.run(arguments, stdout=stdout, stderr=stderr)
            print("Exit status: ", completed.returncode)
            status = completed.returncode == 0

        if status:
            fo = common.flow_call.FlowOutput(completed, stdout_path, stderr_path, output_dir)
            conv_check = fo.check_conv_reasons()
            print("converged: ", conv_check)
            status = conv_check >= 0

        return status

            # if not config_dict["run_on_metacentrum"]:
            #     arguments.extend(['--output_dir', output_dir, fname + ".yaml"])
            #     print("Running: ", " ".join(arguments))
            #     with open(fname + "_stdout", "w") as stdout:
            #         with open(fname + "_stderr", "w") as stderr:
            #             completed = subprocess.run(arguments, stdout=stdout, stderr=stderr)
            #     print("Exit status: ", completed.returncode)
            #     status = completed.returncode == 0
            # else:


            # from mpi4py import MPI
            # # arguments.extend(['--output_dir', os.path.abspath(output_dir), os.path.abspath(fname + ".yaml")])
            # # print("Running: ", " ".join(arguments))
            # # sub_comm = MPI.COMM_SELF.Spawn('flow123d', args=arguments[1:], maxprocs=1)
            #
            # # wrap inside bash script to simulate redirection of stderr and stdout from spawned process
            # arguments.extend(['--output_dir', output_dir, fname + ".yaml"])
            # finished_file = "FINISHED"
            # lines = [
            #     '#!/bin/bash',
            #     'cd ' + self.sample_dir,
            #     'touch ' + finished_file,
            #     'extime=\"$(time (' + " ".join(arguments) + " 2> stderr.log" + " 1> stdout.log" + ") 2>&1 1>/dev/null )\"",
            #     'printf "$extime" >> ' + finished_file
            #     # " ".join(arguments),
            #     # 'printf "finished" >> ' + finished_file
            # ]
            # run_script = "flow123d.sh"
            # run_script = os.path.abspath(run_script)
            # with open(run_script, 'w') as f:
            #     f.write('\n'.join(lines))
            # sub_comm = MPI.COMM_SELF.Spawn('bash', args=[run_script], maxprocs=1)
            # # sub_comm.Barrier()
            # sub_comm.Disconnect()
            # status = False
            # max_time = config_dict["max_time_per_sample"]
            # step = min(2, max_time)
            # cumul_time = 0
            # finished_file = os.path.join(self.sample_dir, finished_file)
            # log_file = os.path.join(self.sample_dir, output_dir, "flow123.0.log")
            # print(finished_file)
            # print(log_file)
            #
            # while cumul_time < max_time:
            #     time.sleep(step)
            #     cumul_time = cumul_time + step
            #     print("cumul_time:", cumul_time, "log size:", os.stat(log_file).st_size)
            #     print("finished size:", os.stat(finished_file).st_size)
            #     # this works:
            #     if os.path.isfile(finished_file) and os.stat(finished_file).st_size != 0:
            #     # this does not:
            #     # if  os.stat(finished_file).st_size != 0:
            #             time.sleep(step)
            #             status = True
            #             break
            #
            #     # if os.path.isfile(log_file):
            #     #     with open(log_file, 'r') as f:
            #     #         for line in f:
            #     #             pass
            #     #         print(line)
            #     #         status = line == "  - O.K."
            #     #     time.sleep(step)
            #         # status = True

            # if status:
            # conv_check = aux_functions.check_conv_reasons(log_file)
            # print("converged: ", conv_check)
            # status = conv_check >= 0
            #
            # return status



    def prepare_mesh(self, config_dict, cut_tunnel):
        mesh_name = config_dict["geometry"]["mesh_name"]
        if cut_tunnel:
            mesh_name = mesh_name + "_cut"
        mesh_file = mesh_name + ".msh"
        mesh_healed = mesh_name + "_healed.msh"

        # suppose that the mesh was created/copied during preprocess
        assert os.path.isfile(os.path.join(config_dict["common_files_dir"], mesh_healed))
        shutil.copyfile(os.path.join(config_dict["common_files_dir"], mesh_healed), mesh_healed)
        return mesh_healed

        # if os.path.isfile(os.path.join(config_dict["common_files_dir"], mesh_healed)):
        #     shutil.copyfile(os.path.join(config_dict["common_files_dir"], mesh_healed), mesh_healed)
        #     return mesh_healed

        # if not os.path.isfile(mesh_healed):
        #     if not os.path.isfile(mesh_file):
        #         self.make_mesh_A(config_dict, mesh_name, mesh_file, cut_tunnel=cut_tunnel)
        #         # self.make_mesh_B(config_dict, mesh_name, mesh_file, cut_tunnel=cut_tunnel)
        #     hm = heal_mesh.HealMesh.read_mesh(mesh_file, node_tol=1e-4)
        #     hm.heal_mesh(gamma_tol=0.01)
        #     hm.stats_to_yaml(mesh_name + "_heal_stats.yaml")
        #     hm.write()
        #     assert hm.healed_mesh_name == mesh_healed
        # return mesh_healed

    # @staticmethod
    # def prepare_hm_input(config_dict):
    #     """
    #     Prepare FieldFE input file for the TH simulation.
    #     :param config_dict: Parsed config.yaml. see key comments there.
    #     """
    #     bc_pressure_csv = 'bc_pressure_tunnel.csv'
    #     if os.path.exists(bc_pressure_csv):
    #         return
    #
    #     end_time = 17
    #     time_step = 1
    #     times = np.arange(0, end_time, time_step)
    #     n_steps = len(times)
    #     times = np.append(times, end_time)
    #
    #     start_val = 300
    #     end_val = 0
    #     val_step = (end_val-start_val)/n_steps
    #     values = np.arange(start_val, end_val, val_step)
    #     values = np.append(values, end_val)
    #
    #     header = "time bc_pressure_tunnel"
    #     fmt = "%g"
    #     list_rows = np.column_stack((times, values))
    #     np.savetxt(bc_pressure_csv, list_rows, fmt=fmt, delimiter=' ', header=header)
    #     # with open(bc_pressure_csv, 'w', newline='') as csv_file:
    #     #     writer = csv.writer(csv_file)
    #     #     writer.writerow(["time", "bc_pressure_tunnel"])
    #     #     for t, v in zip(times, values):
    #     #         writer.writerow([t, v])

        # PREPARE normal traction on tunnel boundary evolving in time






    # @staticmethod
    # def extract_time_series(yaml_stream, regions, extract):
    #     """
    #
    #     :param yaml_stream:
    #     :param regions:
    #     :return: times list, list: for every region the array of value series
    #     """
    #     data = yaml.load(yaml_stream, yaml.CSafeLoader)['data']
    #     times = set()
    #     reg_series = {reg: [] for reg in regions}
    #
    #     for time_data in data:
    #         region = time_data['region']
    #         if region in reg_series:
    #             times.add(time_data['time'])
    #             power_in_time = extract(time_data)
    #             reg_series[region].append(power_in_time)
    #     times = list(times)
    #     times.sort()
    #     series = [np.array(region_series, dtype=float) for region_series in reg_series.values()]
    #     return np.array(times), np.array(series)
    #
    # @staticmethod
    # def extract_th_results(output_dir, out_regions, bc_regions):
    #     with open(os.path.join(output_dir, "energy_balance.yaml"), "r") as f:
    #         power_times, reg_powers = endorse_2Dtest.extract_time_series(f, bc_regions, extract=lambda frame: frame['data'][0])
    #         power_series = -sum(reg_powers)
    #
    #     with open(os.path.join(output_dir, "Heat_AdvectionDiffusion_region_stat.yaml"), "r") as f:
    #         temp_times, reg_temps = endorse_2Dtest.extract_time_series(f, out_regions, extract=lambda frame: frame['average'][0])
    #     with open(os.path.join(output_dir, "water_balance.yaml"), "r") as f:
    #         flux_times, reg_fluxes = endorse_2Dtest.extract_time_series(f, out_regions, extract=lambda frame: frame['data'][0])
    #     sum_flux = sum(reg_fluxes)
    #
    #     reg_temps = reg_temps - endorse_2Dtest.zero_temperature_offset
    #
    #     avg_temp_flux = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux
    #     return avg_temp_flux, power_series

    # @staticmethod
    # def plot_exchanger_evolution(temp_times, avg_temp, power_times, power_series):
    #     year_sec = 60 * 60 * 24 * 365
    #
    #     import matplotlib.pyplot as plt
    #     fig, ax1 = plt.subplots()
    #     temp_color = 'red'
    #     ax1.set_xlabel('time [y]')
    #     ax1.set_ylabel('Temperature [C deg]', color=temp_color)
    #     ax1.plot(temp_times[1:] / year_sec, avg_temp[1:], color=temp_color)
    #     ax1.tick_params(axis='y', labelcolor=temp_color)
    #
    #     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #     pow_color = 'blue'
    #     ax2.set_ylabel('Power [MW]', color=pow_color)  # we already handled the x-label with ax1
    #     ax2.plot(power_times[1:] / year_sec, power_series[1:] / 1e6, color=pow_color)
    #     ax2.tick_params(axis='y', labelcolor=pow_color)
    #
    #     fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #     plt.show()

    def observe_time_plot(self, config_dict):

        output_dir = config_dict["hm_params"]["output_dir"]
        pressure_points2collect = config_dict["surrDAMH_parameters"]["observe_points"]

        with open(os.path.join(output_dir, "flow_observe.yaml"), "r") as f:
            loaded_yaml = yaml.load(f, yaml.CSafeLoader)
            data = loaded_yaml['data']
            times = np.array([d["time"] for d in data]).transpose()

            values = self.get_from_observe(loaded_yaml, pressure_points2collect, 'pressure_p0')

            fig, ax1 = plt.subplots()
            temp_color = ['red', 'green', 'violet', 'blue']
            ax1.set_xlabel('time [d]')
            ax1.set_ylabel('pressure [m]')
            for i in range(0, len(pressure_points2collect)):
                ax1.plot(times, values[i, 0:], color=temp_color[i], label=pressure_points2collect[i])

            ax1.tick_params(axis='y')
            ax1.legend()

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            # plt.show()
            plt.savefig("observe_pressure.pdf")

    def smooth_ode(self, times, values, smooth_factor):

        pw = scipy.interpolate.CubicSpline(times, values, bc_type='natural')

        p0 = values[0]
        tspan = [times[0], times[-1]]

        p0V0 = np.pi * 0.0025 * 1 * p0
        def ode_func(t, y):
            return y*y/p0V0 * smooth_factor * (pw(t)-y)

        sol = scipy.integrate.solve_ivp(fun=ode_func, t_span=tspan, y0=[p0], t_eval=times)
        return sol.y

