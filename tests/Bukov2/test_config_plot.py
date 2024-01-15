from endorse.Bukov2 import optimize_configs as oc


configs = [
    ('P +5-25+15', [0,10,50,69]),
    ('P +5-22-14', [10,20,30,40]),
    ('P-13+53-11', [40,45,50,55]),
    ('P-11+48-13', [30,31,32,33]),
    ('P +6-34-20', [20,40,60,69])
]
oc.export_vtk_bh_chamber_set(configs, "boreholes_opt_cfg.test.vtk", plot=False)
