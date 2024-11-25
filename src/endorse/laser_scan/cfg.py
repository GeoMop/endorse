import pathlib\

script_dir = pathlib.Path(__file__).parent
workdir = script_dir.parent.parent / "workdir"

input_obj_file = workdir / "orig_data" / "L5.obj"
origin_global_coords = [-622600.0, -1127800.0, 0.0]
