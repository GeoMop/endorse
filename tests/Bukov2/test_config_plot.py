from endorse.Bukov2 import optimize_configs as oc
from pathlib import Path
script_dir = Path(__file__).absolute().parent


configs = \
[[
    ('P +5-25+15', [3,9,14,24]),
    ('P-13+66+12', [6, 13,19,30]),

    ('P-13+59-12', [16,20,28,34]),
    ('P-13+70-15', [16,26,30,31]),
    ('P -6+13-07', [11,18,24,30]),
    ('P +5-09+00', [18, 26, 32, 38])
],
]

cfg_file = script_dir / "3d_model/PE_01_02/Bukov2_mesh.yaml"
for i, cfg in enumerate(configs):
    oc.export_vtk_bh_chamber_set(cfg_file, cfg, f"boreholes_opt_cfg_{i}.vtk", plot=False)
