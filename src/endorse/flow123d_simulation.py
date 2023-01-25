import os
import numpy as np
import traceback
import matplotlib.pyplot as plt

from . import common
from . import flow123d_inputs_path
from endorse.common.config import dotdict


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


class Edz_HM_TSX_2D():

    def __init__(self, config):
        # TODO: remove config modifications
        self._config = config
        self.sample_counter = -1
        self.flow_output = None

    def set_parameters(self, data_dict: 'dotdict'):
        param_list = self._config.tsx_hm_model.surrDAMH_parameters.parameters

        sub_dict = {p.name: data_dict[p.name] for p in param_list}
        self._config.tsx_hm_model.hm_params.update(sub_dict)

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

        # collect only
        if config_dict["tsx_hm_model"]["collect_only"]:
            return 2, self.collect_results()

        print("Creating mesh...")
        comp_mesh = self.prepare_mesh(cut_tunnel=True)

        mesh_bn = os.path.basename(comp_mesh)
        config_dict["tsx_hm_model"]["hm_params"]["mesh"] = mesh_bn

        # endorse_2Dtest.read_physical_names(config_dict, comp_mesh)
        print("Creating mesh...finished")

        if config_dict.tsx_hm_model.mesh_only:
            # TODO: Just raise exception if could not return correct values.
            return -10, []  # tag, value_list

        # endorse_2Dtest.prepare_hm_input(config_dict)
        print("Running Flow123d - HM...")

        #hm_succeed = self.call_flow(config_dict, 'hm_params', result_files=["flow_observe.yaml"])

        params = config_dict.tsx_hm_model.hm_params
        template = os.path.join(flow123d_inputs_path, params.input_template)
        self.flow_output = common.call_flow(config_dict.flow_env, template, params)

        if not self.flow_output.success:
            # raise Exception("HM model failed.")
            # "Flow123d failed (wrong input or solver diverged)"
            print("Flow123d failed.")
            return -1, []  # tag, value_list
        print("Running Flow123d - HM...finished")

        if self._config.tsx_hm_model.make_plots:
            try:
                self.observe_time_plot(config_dict, self.flow_output)
            except:
                print("Making plot of sample results failed:")
                traceback.print_exc()
                return -2, []

        print("Finished computation")

        # collected_values = self.collect_results(config_dict)
        # print("Sample results collected.")
        # return 1, collected_values  # tag, value_list

        try:
            collected_values = self.collect_results(self.flow_output)
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

    def collect_results(self, flow_output):
        #output_dir = config_dict["hm_params"]["output_dir"]
        points2collect = self._config.tsx_hm_model.surrDAMH_parameters.observe_points

        # the times defined in input
        times = np.array(generate_time_axis(self._config.tsx_hm_model))
        #with open(os.path.join(output_dir, "flow_observe.yaml"), "r") as f:
        #    loaded_yaml = yaml.load(f, yaml.CSafeLoader)

        flow_observe = flow_output.hydro.observe_dict()
        points = flow_observe['points']
        point_names = [p["name"] for p in points]

        points2collect_indices = []
        for p2c in points2collect:
            tmp = [i for i, pn in enumerate(point_names) if pn == p2c]
            assert len(tmp) == 1
            points2collect_indices.append(tmp[0])

        print("Collecting results for observe points: ", points2collect)
        data = flow_observe['data']
        data_values = np.array([d["pressure_p0"] for d in data])
        values = data_values[:, points2collect_indices]
        obs_times = np.array([d["time"] for d in data]).transpose()

        # check that observe data are computed at all times of defined time axis
        all_times_computed = np.alltrue(np.isin(times, obs_times))
        if not all_times_computed:
            raise Exception("Observe data not computed at all times as defined by input!")
        # skip the times not specified in input
        t_indices = np.isin(obs_times, times).nonzero()
        values = values[t_indices].transpose()

        # flatten to format: [Point0_all_all_times, Point1_all_all_times, Point2_all_all_times, ...]
        res = values.flatten()
        return res

    def prepare_mesh(self, cut_tunnel):
        mesh_name = self._config.geometry.tsx_tunnel.mesh_name
        # if cut_tunnel:
        #     mesh_name = mesh_name + "_cut"
        # mesh_file = mesh_name + ".msh"
        # mesh_healed = mesh_name + "_healed.msh"

        # suppose that the mesh was created/copied during preprocess
        # assert os.path.isfile(os.path.join(self._config.common_files_dir, mesh_healed))
        # shutil.copyfile(os.path.join(self._config.common_files_dir, mesh_healed), mesh_healed)
        # return mesh_healed
        import endorse.mesh.tunnel_cross_section as tcs
        meshfile = tcs.make_tunnel_cross_section_mesh(self._config.geometry.tsx_tunnel)
        return meshfile


    def observe_time_plot(self, config_dict, flow_output):
        flow_observe = flow_output.hydro.observe_dict()
        #output_dir = config_dict["hm_params"]["output_dir"]

        #with open(os.path.join(output_dir, "flow_observe.yaml"), "r") as f:
        #flow_observe = yaml.load(f, yaml.CSafeLoader)
        points = flow_observe.points
        point_names = [p["name"] for p in points]
        data = flow_observe.data
        values = np.array([d.pressure_p0 for d in data]).transpose()
        times = np.array([d.time for d in data]).transpose()

        fig, ax1 = plt.subplots()
        temp_color = ['red', 'green', 'violet', 'blue']
        ax1.set_xlabel('time [d]')
        ax1.set_ylabel('pressure [m]')
        for i in range(0, len(point_names)):
            ax1.plot(times, values[i, 0:], color=temp_color[i], label=point_names[i])

        ax1.tick_params(axis='y')
        ax1.legend()

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()
        plt.savefig("observe_pressure.pdf")
