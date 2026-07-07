from pathlib import Path

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
