# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/jb/workspace/endorse-pointcloud/src/endorse/cfg.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 2024-11-23 19:49:30 UTC (1732391370)

import pathlib
script_dir = pathlib.Path(__file__).resolve().parent
workdir = script_dir / 'workdir'
input_dir = script_dir

input_obj_file = workdir.parent / 'scan_meshes' / 'L5.obj'
origin_global_coords = [-622600.0, -1127800.0, 0.0]
