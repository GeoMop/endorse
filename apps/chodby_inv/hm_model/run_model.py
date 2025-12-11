import pathlib
import yaml
from math import exp, log
import numpy as np
from endorse import common
import chodby_inv.input_data as input_data
from scipy.optimize import least_squares, OptimizeResult

from boreholes import Boreholes, ObservePointData
from chodby_inv import piezo
from prepare_model_inputs import prepare_excavation_functions


module_dir = pathlib.Path(__file__).parent
work_dir = input_data.work_dir

def run_init_pressure_calibration(cfg:'dotdict'):
    def f(x, p_target, machine_config, yaml_replacements, fitting_params):

        frac_input_params = []
        for (name,params),xi in zip(fitting_params.items(),x):
            frac_input_params.append( {"region": name, "conductivity": exp(xi), "cross_section": params["cross_section"]} )
        region_str = yaml.dump(frac_input_params, default_flow_style=False, sort_keys=False).rstrip("\n")
        region_str_indented = "\n".join(" "*10 + l for l in region_str.split("\n"))
        yaml_replacements["fracture_input_params"] = region_str_indented

        common.call_flow(machine_config, input_data.config_sim_calibration_tmpl_yaml,
                                       yaml_replacements)

        obs = ObservePointData(bhs, "output/flow_observe.yaml")
        residuals = []
        for ci, cname in enumerate(obs._chamber_names):
            if cname in p_target:
                pressure = np.mean(obs.chamber_pressures(ci, 0))
                residuals.append((p_target[cname]-pressure))

        return np.asarray(residuals)


    def callback(intermediate_result: OptimizeResult):
        r = intermediate_result
        print(f"{r.nit} {r.cost} {np.exp(r.x)}")



    with common.workdir(work_dir):
        excavation_n, excavation_s = prepare_excavation_functions()
        bhs = Boreholes()
        observe_points_str = bhs.make_observe_points()
        yaml_replacements = {"observe_points": observe_points_str,
                             "excavation_north": yaml.dump(excavation_n, default_flow_style=True),
                             "excavation_south": yaml.dump(excavation_s, default_flow_style=True)}

        # get target initial pressures
        df_pressure = piezo.excavation_epoch_df()
        target_pressures = {}
        for bhi in range(bhs.n_boreholes):
            bhname = bhs.bh_name(bhi)
            df_filtered = df_pressure[(df_pressure['borehole'] == bhname) & (df_pressure['time_days'] == 0)]
            pressure = df_filtered['pressure'].to_numpy() / 10 # pressure head in meters
            chamber = df_filtered['section'].to_numpy()
            for ch, p in zip(chamber, pressure):
                target_pressures[f"{bhname}_ch_{ch}"] = p

        # define regions whose conductivity is to be identified
        fitting_regions = {
            "fr1_tunnel" : {"cross_section": 1e-3},
            "fr1_L5-23UR" : {"cross_section": 1e-3},
            "fr1_L5-26R" : {"cross_section": 1e-3},
            "fr1_L5-37R" : {"cross_section": 302.93e-3},
            "fr1_L5-37UR" : {"cross_section": 32.64e-3},
            "fr2_tunnel" : {"cross_section": 1e-3},
            "fr2_L5-22DR" : {"cross_section": 1e-3},
            "fr2_L5-37R" : {"cross_section": 53.09e-3},
            "fr3_tunnel" : {"cross_section": 1e-3},
            "fr3_L5-37R" : {"cross_section": 81.39e-3},
            "fr3_L5-37UR" : {"cross_section": 32.64e-3}
        }

        # # define chambers whose initial pressure is to be matched
        # fitting_chambers = [
        #     ""
        # ]

        # initial guess
        x0 = np.ones(len(fitting_regions))*log(1e-3)

        result = least_squares(f, x0, bounds=(log(5e-13), 0), callback=callback, diff_step=1e-3,
                               kwargs={
                                   "p_target":target_pressures,
                                   "machine_config":cfg.machine_config,
                                   "yaml_replacements":yaml_replacements,
                                   "fitting_params":fitting_regions
                                   }
                               )
        print(result)
        print(f"conductivities: {np.exp(result.x)}")

        obs = ObservePointData(bhs, "output/flow_observe.yaml")
        for ci, cname in enumerate(obs._chamber_names):
            if cname in target_pressures:
                pressure = np.mean(obs.chamber_pressures(ci, 0))
                print(f"{cname} target: {target_pressures[cname]} attained: {pressure}")







if __name__ == "__main__":

    cfg = common.load_config("config.yaml")
    run_init_pressure_calibration(cfg)
    # with common.workdir(work_dir):
    #     excavation_n, excavation_s = prepare_excavation_functions()
    #     observe_points_str = Boreholes().make_observe_points()
    #     flow_output = common.call_flow(cfg.machine_config, input_data.config_sim_tmpl_yaml,
    #                                    {"observe_points":observe_points_str,
    #                                     "excavation_north":yaml.dump(excavation_n, default_flow_style=True),
    #                                     "excavation_south":yaml.dump(excavation_s, default_flow_style=True)})
