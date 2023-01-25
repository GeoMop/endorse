import os.path

from endorse.common import File, workdir,EndorseCache
from endorse.indicator import indicators
from endorse import plots


def test_indicator():
    #EndorseCache.instance().expire_all()
    #pvd_file = File("test_data/trans_m_01/solute_fields.pvd")
    case = "trans_m_01"
    #pvd_file = File(f"test_data/{case}/solute_fields.pvd")
    for case in ["edz_pos02", "edz_pos10", "noedz_pos02", "noedz_pos10"]:
        for i in range(1, 10):
            path = f"sandbox/{case}/output/L00_S00000{i:02d}/output/solute_fields.pvd"
            if not os.path.isfile(path):
                continue
            pvd_file = File(path)
            with workdir('sandbox'):
                inds = indicators(pvd_file, 'U235_conc', (-27, 27))
                plots.plot_indicators(inds, file=f"indicator_{case}_{i:02d}")
                ind_time_max = [ind.time_max()[1] for ind in inds]
                print(ind_time_max)