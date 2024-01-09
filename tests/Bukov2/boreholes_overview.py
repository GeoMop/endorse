"""
Sort boreholes by various criteria.
"""
import sys
from pathlib import Path
from endorse.Bukov2 import plot_boreholes, boreholes, sa_problem, bukov_common as bcommon
from endorse import common
import numpy as np
import pyvista as pv

def plot_bh_selection(cfg, bh_set, param_name, bh_tuples):
    #pv.start_xvfb()
    #plotter = pv.Plotter(off_screen=True)
    plotter = pv.Plotter()

    plotter = plot_boreholes.create_scene(plotter, cfg.geometry)
    plot_boreholes.add_cylinders(plotter, bh_set.lateral)
    plot_boreholes.plot_bh_subset(plotter, bh_set, bh_tuples)
    plotter.camera.parallel_projection = True


    # Add your actual objects with custom colors here
    # ...
    # Create a dummy mesh with the same number of points as unique colors
    num_colors = 10
    vals, ids = zip(*bh_tuples)
    x = np.linspace(0, 1, num_colors)
    y = z = np.zeros(num_colors)
    points = np.column_stack((x, y, z))
    dummy_mesh = pv.PolyData(points)

    # Assign scalar values that correspond to unique colors
    dummy_mesh.point_data['scalars'] = np.array(vals)

    # Add the dummy mesh (won't actually display it)
    plotter.add_mesh(dummy_mesh, scalars='scalars', cmap='viridis', show_scalar_bar=False)

    # Add a color bar that reflects the colormap of the dummy mesh
    plotter.add_scalar_bar(title="Custom Colors", n_labels=num_colors, vertical=True)

    #plotter.add_scalar_bar(title="Scalar Data", n_labels=5, title_font_size=20, label_font_size=15)
    plotter.add_text(param_name)
    plotter.show()


def main(workdir):
    bh_packers = bcommon.pkl_read(workdir, "all_bh_configs.pkl")
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    bh_set = boreholes.make_borehole_set(workdir, cfg)

    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    n_params = problem['num_vars']

    # n_boreholes, n_params, n_variants PackerCfg
    n_boreholes = len(bh_packers)
    assert n_params == len(bh_packers[0])
    bh_param_selected = [list() for _ in range(n_params)]
    for ibh, bh in enumerate(bh_packers):
        for i_par, p_variants in enumerate(bh):
            for i_var, pack_cfg in enumerate(p_variants):
                for i_ch, chamber in enumerate(pack_cfg.sobol_indices):
                    for ip, param in enumerate(chamber):
                        bh_param_selected[ip].append((param[0], ibh))

    with open(workdir / "summary.txt", "w") as sum_f:
        for pname, par_list in zip(problem['names'], bh_param_selected):
            tuples = np.array(par_list)
            _, indices = np.unique(tuples[:, 1], return_index=True)

            par_list = tuples[indices].tolist()
            par_list.sort(reverse=True)
            with open(workdir / f"param_{pname}_bh.txt", "w") as f:
                for row in par_list:
                    f.write(f"{row[0]}, {int(row[1])}")
                    f.write("\n")
            sum_f.write(f"parameter {pname}\n")
            for row in par_list[:10]:
                sum_f.write(f"{row[0]}, {int(row[1])}")
                sum_f.write("\n")

            print("N boreholes:", bh_set.n_boreholes)

            plot_bh_selection(cfg, bh_set, pname, par_list[:10])

if __name__ == '__main__':
    workdir = Path(sys.argv[1]).absolute()
    main(workdir)