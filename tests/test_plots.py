import os
import pytest
from endorse import plots, common, indicator
from bgem.gmsh.gmsh_io import GmshIO
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.skip
def test_plot_mesh():
    mesh = GmshIO('test_data/rectangle_2x5.msh')
    fig, ax = plt.subplots(figsize=(10,10))
    plots.plot_mesh_2d(ax, mesh)
    X = np.linspace(0,1,100)
    Y = X**2
    #ax.plot(X,Y)
    plt.show()

    print('DONE')

@pytest.mark.skip
def test_plot_source():
    conf_file = os.path.join(script_dir, "test_data/config_homo_tsx.yaml")
    cfg = common.load_config(conf_file)
    cfg_fine = cfg.transport_fullscale
    conc_flux = common.File(os.path.join(os.path.dirname(conf_file), cfg_fine.conc_flux_file))
    source_surface = cfg_fine.source_params.source_length * cfg.geometry.borehole.radius * 2 * np.pi
    plots.plot_source(conc_flux, source_surface)

#@pytest.mark.skip
def test_plot_log_errorbar_groups():
    g1 = 'g1'
    g2 = 'g2'
    p1 = "1:5"
    p2 = "2:5"
    data = [
        [g1, p1, -1, 0.3, [-1.3, -0.3, 0.1, -2] ],
        [g1, p2, -11, 2, [-11, -12, -10]],
        [g2, p2, -6, 1, [-6, -6.5, -5]],
        [g2, p1, -3, 0.5, [-3, -4, -3.5]]
    ]
    plots.plot_log_errorbar_groups(data, 'conc ' + r'$[g/m^3]$')

@pytest.mark.skip
def test_plot_slicese():
    case = "trans_m_01"
    pvd_file = common.File(f"test_data/{case}/solute_fields.pvd")
    with common.workdir('sandbox'):
        inds = indicator.indicators(pvd_file, 'U235_conc', (-10, 10))
        plots.plot_indicators(inds, file=case)
        ind_time_max = [ind.time_max()[1] for ind in inds]
        print(ind_time_max)
        itime = indicator.IndicatorFn.common_max_time(inds)  # not splined version, need slice data
        plots.plot_slices(pvd_file, "U235_conc", [-30, 30], [itime-1, itime, itime+1])

