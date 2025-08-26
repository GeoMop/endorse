import pathlib
import yaml
from endorse import common
import chodby_inv.input_data as input_data
import chodby_inv.mesh as mesh
from boreholes import Boreholes
from prepare_model_inputs import prepare_excavation_functions


module_dir = pathlib.Path(__file__).parent
work_dir = input_data.work_dir

if __name__ == "__main__":

    cfg = common.load_config("config.yaml")
    with common.workdir(work_dir):
        excavation_n, excavation_s = prepare_excavation_functions()
        observe_points_str = Boreholes().make_observe_points()
        flow_output = common.call_flow(cfg.machine_config, input_data.config_sim_tmpl_yaml,
                                       {"observe_points":observe_points_str,
                                        "excavation_north":yaml.dump(excavation_n, default_flow_style=True),
                                        "excavation_south":yaml.dump(excavation_s, default_flow_style=True)})
