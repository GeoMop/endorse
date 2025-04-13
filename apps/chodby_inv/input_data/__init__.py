import pathlib


__script_dir__ = pathlib.Path(__file__).parent

# Following is public
input_dir = __script_dir__.parent / "input_data"
bh_cfg_file = input_dir / "boreholes.yaml"
