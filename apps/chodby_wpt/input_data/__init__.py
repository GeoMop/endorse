from pathlib import Path
from enum import Enum

# common output directory
__script_dir__ = Path(__file__).parent
work_dir = __script_dir__.parent / "workdir"
work_dir.mkdir(parents=True, exist_ok=True)

# Following is public
input_dir = __script_dir__.parent / "input_data"
bh_cfg_yaml = input_dir / "boreholes.yaml"   # geometry of boreholes and adjacent fractures
mesh_cfg_yaml = input_dir / "mesh.yaml"      # definitions for mesh generation
hm_sim_tmpl_yaml = input_dir / "hm_sim_tmpl.yaml" # HM simulation template config
ep02_ods = input_dir / "EP02_PVP2_ZZ_Tabulka_vysledky_dokumentace_celeb_a_chodeb.ods"
events = input_dir / "events.yaml"
large_model_mesh = input_dir / "3d_model.vtu"
data_2025 = input_dir / "wpt_2025.csv"

# Borehole and Section enums
class Borehole(str, Enum):
    L5_50UL = "L5-50UL"
    L5_49DL = "L5-49DL"
    L5_37UR = "L5-37UR"
    L5_37R  = "L5-37R"
    L5_26R  = "L5-26R"
    L5_24DR = "L5-24DR"
    L5_23UR = "L5-23UR"
    L5_22DR = "L5-22DR"

class Section(Enum):
    CLOSEST = 0
    MIDDLE = 1
    FURTHEST = 2
