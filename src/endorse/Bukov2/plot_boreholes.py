import numpy as np
import pyvista as pv


def create_scene(cfg_geometry):
    cfg = cfg_geometry.main_tunnel
    # Create a plotting object
    plotter = pv.Plotter()
    # L5
    x_half = cfg.width / 2
    y_half = cfg.length / 2

    box = pv.Box(bounds=(-x_half, +x_half, -y_half -10, +y_half -10, 0, cfg.height))
    plotter.add_mesh(box, color='grey' , opacity=0.7)

    plotter.add_axes()
    plotter.show_bounds(grid='front', all_edges=True)
    return plotter

def plot_bh_set(plotter, bh_set: 'BoreholeSet'):
    # Create a horizontal cylinder
    r, l0, l1 = bh_set.avoid_cylinder
    avoid_cylinder = pv.Cylinder(center=bh_set.transform([0.5 * l0 + 0.5 * l1, 0, 0]), direction=(1, 0, 0), radius=r, height=l1-l0)
    plotter.add_mesh(avoid_cylinder, color='red')

    r, l0, l1 = bh_set.active_cylinder
    active_cylinder = pv.Cylinder(center=bh_set.transform([0.5 * l0 + 0.5 * l1, 0, 0]), direction=(1, 0, 0), radius=r, height=l1-l0)
    plotter.add_mesh(active_cylinder, color='grey', opacity=0.1)

    for i in range(bh_set.n_y_angles):
        for j in range(bh_set.n_z_angles):
            iangle_norm = (i / bh_set.n_y_angles, j / bh_set.n_z_angles)
            for i_bh in bh_set.angles_table[i][j]:
                add_bh(plotter, iangle_norm, bh_set, i_bh)
    return plotter

def add_bh(plotter, angle_norm, bh_set, i_bh):
    p_w, dir, p_tr = bh_set.bh_list[i_bh]
    p_w = bh_set.transform(p_w)
    p_tr = bh_set.transform(p_tr)
    points, bounds = bh_set.point_lines
    p_begin = points[i_bh, bounds[i_bh][0], :]
    p_end = points[i_bh, bounds[i_bh][1], :]

    color = (0.8 * angle_norm[0] + 0.1, 0.2, 0.8 * angle_norm[1] + 0.1)
    #print(f"Adding: {bh} col: {color}")
    line = pv.Line(p_w, p_tr)
    plotter.add_mesh(line, color='grey', line_width=1)

    line = pv.Line(p_begin, p_end)
    plotter.add_mesh(line, color=color, line_width=2)

    # Transversal point
    sphere = pv.Sphere(0.5, p_tr)
    plotter.add_mesh(sphere, color=color)

    # for pt in points[i_bh, : bounds[i_bh][1]]:
    #     sphere = pv.Sphere(0.3, pt)
    #     plotter.add_mesh(sphere, color=color)

#
# # Example usage
#
#
# plotter = create_scene()
#
# add_line_segment(plotter, [0, 0, 0], [5, 5, 5])
# plotter.show()
