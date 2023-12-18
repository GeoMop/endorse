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
            for bh in bh_set.direction_lookup(i,j):
                add_bh(plotter, iangle_norm, bh_set, bh)
    return plotter

def add_bh(plotter, angle_norm, bh_set, bh):
    p1 = bh[:3]
    d1 = bh[3:]
    p2 = bh_set.transform(p1 + d1)
    p3 = bh_set.transform(p1 + 3 * d1)
    p1 = bh_set.transform(p1)

    color = (0.8 * angle_norm[0] + 0.1, 0.2, 0.8 * angle_norm[1] + 0.1)
    #print(f"Adding: {bh} col: {color}")
    line = pv.Line(p1, p3)
    plotter.add_mesh(line, color=color, line_width=5)

    sphere = pv.Sphere(0.5, p2)
    plotter.add_mesh(sphere, color=color, line_width=2)
#
# # Example usage
#
#
# plotter = create_scene()
#
# add_line_segment(plotter, [0, 0, 0], [5, 5, 5])
# plotter.show()
