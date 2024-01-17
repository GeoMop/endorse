import os
import pytest
from endorse import common
from pathlib import Path
from endorse.Bukov2 import plot_boreholes as pb
script_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skip
def test_borehole_scene():
    workdir = script_dir
    conf_file = os.path.join(workdir, "./Bukov2_mesh.yaml")
    cfg = common.config.load_config(conf_file)
    mesh_file = pb.create_scene(cfg.geometry)

@pytest.mark.skip
def test_plot_boreholes():
    bh_list = [155, 51, 49, 149, 14, 98, 96, 108, 140, 119, 139]

    def plot_bh_selection(cfg, bh_set, param_name, bh_tuples):
        # pv.start_xvfb()
        # plotter = pv.Plotter(off_screen=True)
        plotter = pv.Plotter()

        plotter = plot_boreholes.create_scene(plotter, cfg.geometry)
        plot_boreholes.add_cylinders(plotter, bh_set)
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

        # plotter.add_scalar_bar(title="Scalar Data", n_labels=5, title_font_size=20, label_font_size=15)
        plotter.add_text(param_name)
        plotter.show()


def test_pvd():
    times = [0,
             *list(range(100, 120, 1)),
             *list(range(120, 140, 2)),
             *list(range(140, 160, 4)),
             *list(range(160, 200, 8)),
             200]
    print(len(times))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
    sens_dir = script_dir / "3d_model" / "PE_D03/sensitivity"
    files = list(sorted(sens_dir.glob("*.vtu")))
    rel_files = [f.relative_to(sens_dir.parent)for f in files]
    file_zip = list(zip(rel_files, times))
    print(file_zip)
    pb.write_pvd_file(file_zip, "sensitivity.pvd")