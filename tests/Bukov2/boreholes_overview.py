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


def get_bh_subset_group( bh_set, g_name):
    indices = [i_bh for i_bh, bh in enumerate(bh_set.boreholes.values()) if bh.group == g_name]
    return bh_set.subset(indices)

def sub_packer_set(packers, bh_subset):
    return packers[bh_subset.keys()]

def print_sensitivity_bh(f, bh_set, value, i_bh):
    i_bh = int(i_bh)
    f.write(f"{value}, {i_bh}, g: {bh_set.boreholes[i_bh].group}")
    f.write("\n")

def select_by_group_rank(bh_set, bh_par_dict):
    """
    In each BH group sort boreholes by sum of their ranks in the first 8 parameters
    (omitting conductivity a,b,c)
    Select first 20
    :param bh_set:
    :param bh_param_selected:
    :return:
    """
    for par_dict in bh_par_dict.values():
        par_names = list(par_dict.keys())
        par_dict['sum'] = sum(rank for rank, sobol in par_dict.values())
    par_shorts = ['ym', 'sx', 'sy', 'sz', 'K0', 'Ke', 'Kd', 'Kg']
    groups = {}
    for i_bh, par_dict in bh_par_dict.items():
        bh = bh_set.boreholes[i_bh]
        group = bh.group
        bh_list = groups.setdefault(group, list())
        bh_list.append([i_bh, bh, par_dict])



    with open(workdir / "group_ranks.txt", "w") as f:
        f.write(f"index | ID | sum rank |{'| |'.join(par_shorts)}\n")

        for group, bh_list in groups.items():
            bh_list.sort(key=lambda x : x[2]['sum'])
            f.write(f"{group}:\n")
            for i_bh, bh, par_dict in bh_list:
                par_dict_list = [f"#{par_dict[par][0]:03d}| {par_dict[par][1]:8.6f}" for par in par_names[:-3]]
                par_dict_str = " | ".join(par_dict_list)
                f.write(f"  {i_bh:03d} | {bh.id:12s} | sum#{par_dict['sum']} | {par_dict_str} |\n")

def main(workdir):
    bh_packers = bcommon.pkl_read(workdir, "all_bh_configs.pkl")
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    bh_set_orig = boreholes.make_borehole_set(workdir, cfg)

    # Select boreholes subset
    bh_set = bh_set_orig
    print("N boreholes:", bh_set.n_boreholes)
    #bh_set = get_bh_subset_group(bh_set_orig, "left_far_up")
    #bh_packers = sub_packer_set(bh_packers, bh_set)


    # find best boreholes on the subset


    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    n_params = problem['num_vars']

    # For every param get list of ST indices together with index of boreholes
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

    bh_par_dict = {}
    # Write down a given number of boreholes with best evaluations
    with open(workdir / "summary.txt", "w") as sum_f:
        for pname, par_list in zip(problem['names'], bh_param_selected):
            bh_max_sobol = {}
            for sobol, i_bh in par_list:
                val = bh_max_sobol.setdefault(i_bh, 0)
                bh_max_sobol[i_bh] = max(val, sobol)
            par_list = list(zip(bh_max_sobol.values(), bh_max_sobol.keys()))
            par_list.sort(reverse=True)  # largest sensitivities first
            for rank, (sobol, i_bh) in enumerate(par_list):
                bh_dict = bh_par_dict.setdefault(i_bh, dict())
                bh_dict[pname] = (rank, sobol)

            with open(workdir / f"param_{pname}_bh.txt", "w") as f:
                for row in par_list:
                    print_sensitivity_bh(f, bh_set, *row)

            sum_f.write(f"parameter {pname}\n")
            for row in par_list[:10]:
                print_sensitivity_bh(sum_f, bh_set, *row)

            print("N boreholes:", bh_set.n_boreholes)

            plot_bh_selection(cfg, bh_set, pname, par_list[:10])

    select_by_group_rank(bh_set, bh_par_dict)


if __name__ == '__main__':
    workdir = Path(sys.argv[1]).absolute()
    main(workdir)