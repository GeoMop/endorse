from typing import *
import numpy as np
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import matplotlib as mpl
import vtk
import os
from xml.etree import ElementTree as ET
import attrs
from pathlib import Path
from endorse.Bukov2 import bukov_common as bcommon


def add_fields(mesh, value, label):
    """
    :param mesh: pyvista mesh
    :param value: sequance of values
    :param label: sequance of labels
    :return: mesh with added constant PointData.
    """
    for val, lab in zip(value, label):
        field = np.full(mesh.n_points, val)
        mesh.point_data[lab] = field
    return mesh


def meshes_bh_vtk(i_bh: int, bh: 'Borehole', chamber_data = None):
    if chamber_data is None:
        bounds = [0]
        values =  []
        value_names = []
    else:
        bounds, values, value_names = chamber_data

    #p_w, dir, p_tr = bh_set.bh_list[i_bh]
    p_w = bh.lateral.transform(bh.start)
    p_tr = bh.lateral.transform(bh.transversal)

    default_values = [i_bh, *bh.yz_angles]
    default_labels = ['index', 'l5_angle', 'inclination']
    meshes = []

    line = pv.Line(p_w, p_tr)
    line1 = add_fields(line, default_values, default_labels )
    if len(value_names) > 0:
        line1 = add_fields(line1, np.mean(values, axis=0), value_names)
    meshes.append(line1)       # borehole line

    # text = pv.Text3D(str(i_bh), height=0.5, center=p_tr, normal=bh.unit_direction)

    # Chamber cylinders
    for begin, end, value in zip(bounds[:-1], bounds[1:], values):
        p_begin = bh.lateral.transform(bh.line_point(begin))
        p_end = bh.lateral.transform(bh.line_point(end-1))
        cylinder = pv.Cylinder(
            center=(p_begin +p_end)/2,
            direction=p_end - p_begin,
            radius = 0.056/2*10,
            height=np.linalg.norm(p_end - p_begin),
            resolution=8,
            capping=False
        )
        cyl1 = add_fields(cylinder, default_values, default_labels)
        meshes.append(add_fields(cyl1, value, value_names))
    return meshes


def make_mesh_bh_set(bh_set: 'BoreholeSet', chamber_data = None):
    meshes = []
    for (i_bh,(ind_bh, bh)) in enumerate(zip(bh_set.orig_indices, bh_set.boreholes)):
        if chamber_data is None:
            ch_d = None
        else:
            bounds, data, labels = chamber_data
            ch_d = bounds[i_bh], data[i_bh], labels
        meshes.extend(meshes_bh_vtk(ind_bh, bh, chamber_data=ch_d))
    return pv.merge(meshes, merge_points=False)


def export_vtk_bh_set(workdir, bh_set, chamber_data = None, fname="boreholes.vtk"):

    # Merge the primitives
    combined_mesh = make_mesh_bh_set(bh_set, chamber_data)        # Try to get distinguise values on interfaces.

    # Visualize the combined mesh with different colors for each ID
    # plotter = pv.Plotter()
    # plotter.add_mesh(combined_mesh, scalars=['ID', *value_names])
    # plotter.show()

    combined_mesh.save(workdir / fname)

def _make_main_tunnel(cfg):
    # L5
    x_half = cfg.width / 2
    y_half = cfg.length / 2

    return pv.Box(bounds=(-x_half, +x_half, -y_half - 10, +y_half - 10, 0, cfg.height))

def create_scene(plotter, cfg_geometry):
    cfg = cfg_geometry.main_tunnel
    # Create a plotting object

    # L5
    box = _make_main_tunnel(cfg)
    plotter.add_mesh(box, color='grey' , opacity=0.7)

    plotter.add_axes()
    plotter.show_grid(font_size=20, xtitle="X", ytitle="Y", ztitle="Z")
    plotter.camera.position = (-30, 0, 0)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.parallel_projection = True

    #plotter.show_bounds(grid='front', all_edges=True)
    return plotter


def _make_cylinders(lateral: 'Lateral'):
    corner_min = lateral.transform([2, -2, -1.8])
    corner_max = lateral.transform([12, 2, 2.2])
    box = pv.Box(bounds=(corner_min[0], corner_max[0], corner_min[1], corner_max[1], corner_min[2], corner_max[2]))

    # Create a horizontal cylinder
    r, l0, l1 = lateral.avoid_cylinder
    avoid_cylinder = pv.Cylinder(center=lateral.transform([0.5 * l0 + 0.5 * l1, 0, 0]), direction=(1, 0, 0), radius=r,
                                 height=l1 - l0)

    r, l0, l1 = lateral.active_cylinder
    active_cylinder = pv.Cylinder(center=lateral.transform([0.5 * l0 + 0.5 * l1, 0, 0]), direction=(1, 0, 0), radius=r,
                                  height=l1 - l0)

    return box, avoid_cylinder, active_cylinder

def add_cylinders(plotter, lateral: 'Lateral'):
    box, avoid_cylinder, active_cylinder = _make_cylinders(lateral)
    plotter.add_mesh(box, color='grey' , opacity=0.7)
    plotter.add_mesh(avoid_cylinder, color='red', opacity=0.3)
    plotter.add_mesh(active_cylinder, color='grey', opacity=0.1)


def plot_bh_set(plotter, bh_set: 'BoreholeSet'):
    for i_bh, bh in enumerate(bh_set.boreholes):
        add_bh(plotter, bh)
    return plotter


def plot_bh_subset(plotter, bh_set: 'BoreholeSet', bh_tuples):
    values, ids = zip(*bh_tuples)
    bh_indices = [int(id) for id in ids]
    values = np.array(values)
    normalized_values = (values - values.min()) / (values.max() - values.min())
    cmap = plt.cm.get_cmap("viridis")
    colors = cmap(normalized_values)
    for bh_id, col in zip(bh_indices, colors):
        add_bh(plotter, bh_set.boreholes[bh_id], color=col)



def add_bh(plotter, bh: 'Borehole', color=None, label=False):
    #p_w, dir, p_tr = bh_set.bh_list[i_bh]
    p_w = bh.lateral.transform(bh.start)
    p_tr = bh.lateral.transform(bh.transversal)
    bounds = bh.bounds
    p_begin = bh.lateral.transform(bh.line_point(bounds[0]))
    p_end = bh.lateral.transform(bh.line_point(bounds[1]))

    angle_norm = (np.array(bh.yz_angles) + 90) / 180

    if color is None:
        color = (0.8 * angle_norm[0] + 0.1, 0.2, 0.8 * angle_norm[1] + 0.1)
    if label:
        # Somehow doesn't work, prevents plot of several boreholes.
        plotter.add_point_labels([p_end], [bh.id], text_color=color,
            font_size = 50, point_size = 50,
            render_points_as_spheres = True, always_visible = True, shadow = True
        )
    #print(f"Adding: {bh} col: {color}")
    line = pv.Line(p_w, p_tr)
    plotter.add_mesh(line, color='grey', line_width=1)

    line = pv.Line(p_begin, p_end)
    plotter.add_mesh(line, color=color, line_width=2)

    # Transversal point
    sphere = pv.Sphere(0.2, p_tr)
    plotter.add_mesh(sphere, color=color)

    # end point
    sphere = pv.Sphere(0.3, bh.lateral.transform(bh.end_point))
    plotter.add_mesh(sphere, color='black')

    # for pt in points[i_bh, : bounds[i_bh][1]]:
    #     sphere = pv.Sphere(0.3, pt)
    #     plotter.add_mesh(sphere, color=color)


def _add_cone(plotter, start, direction, length, angle, color):
    cone = pv.Cone(center=start+length/2*direction/np.linalg.norm(direction), direction=-direction, height=length, angle=angle, resolution=angle)
    plotter.add_mesh(cone, color=color, line_width=1, opacity=0.1)
    plotter.add_mesh(cone, color=color, line_width=1, style='wireframe')
    cone = pv.Cone(center=start-length/2*direction/np.linalg.norm(direction), direction=direction, height=length, angle=angle, resolution=angle)
    plotter.add_mesh(cone, color=color, line_width=1, opacity=0.1)
    plotter.add_mesh(cone, color=color, line_width=1, style='wireframe')
    line = pv.Line(start-5*length*direction, start+5*length*direction)
    plotter.add_mesh(line, color=color, line_width=2)

def add_foliation_cylinders(plotter, lateral):
    if lateral.foliation_angle_tolerance < 90:
        fol_dir = bcommon.direction_vector(-lateral.foliation_longitude+lateral.l5_azimuth+90, lateral.foliation_latitude)
        r, l0, l1 = lateral.avoid_cylinder
        fol_start = lateral.transform([1.0 * l0 - 0.0 * l1, 0, 0])
        _add_cone(plotter, fol_start, fol_dir, 10, lateral.foliation_angle_tolerance, 'green')
    return plotter

#######################################################################


from xml.dom import minidom

def write_pvd_file(files, output_name):
    """
    Write a PVD file for a given sequence of files with line breaks, indentation, and XML declaration.

    :param files: List of tuples (file_path, time_step).
    :param output_name: Full path for the output PVD file.
    """
    # Prepare the root of the PVD file
    pvd_root = ET.Element("VTKFile")
    pvd_root.set("type", "Collection")
    pvd_root.set("version", "0.1")
    pvd_root.set("byte_order", "LittleEndian")
    collection = ET.SubElement(pvd_root, "Collection")

    for file_path, time_step in files:
        # Add dataset entry to PVD file
        dataset = ET.SubElement(collection, "DataSet")
        dataset.set("timestep", str(time_step))
        dataset.set("part", "0")
        dataset.set("file", str(file_path))

    # Convert to string with indentation
    rough_string = ET.tostring(pvd_root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_string = reparsed.toprettyxml(indent="    ")

    # Ensure XML declaration is included
    #xml_declaration = '<?xml version="1.0"?>\n'
    formatted_xml = pretty_string

    # Write the formatted XML to file
    with open(output_name, 'w') as output_file:
        output_file.write(formatted_xml)


def PVD_point_fields(workdir, output_name, times, points, fields_dict):
    """
    Create a sequence of VTU files from 3D points and multiple scalar fields over time, and write a PVD file.

    :param workdir: Directory to save the VTU and PVD files.
    :param output_name: Base name for output files.
    :param times: Array of time labels.
    :param points: Array of 3D coordinates for N points (size N x 3).
    :param fields_dict: Dictionary of scalar fields, each field is a NumPy array of size N x K.
    """
    # Ensure the output directory exists
    os.makedirs(workdir / output_name, exist_ok=True)

    # Convert numpy array of points to VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points))

    # Create a VTK PolyData object
    #polydata = vtk.vtkPolyData()
    polydata = vtk.vtkUnstructuredGrid()
    polydata.SetPoints(vtk_points)

    # List to store file paths and time steps for PVD file
    files = []

    for k, time_step in enumerate(times):
        # Update polydata for the current time step
        for field_name, field_data in fields_dict.items():
            vtk_array = numpy_to_vtk(field_data[:, k])
            vtk_array.SetName(field_name)
            polydata.GetPointData().AddArray(vtk_array)

        # Set up the VTU file writer
        writer = vtk.vtkXMLUnstructuredGridWriter()
        vtu_file_name = f"{output_name}/{output_name}_{k}.vtu"
        writer.SetFileName(str(workdir / vtu_file_name))

        # Set input data and write the file
        writer.SetInputData(polydata)
        success = writer.Write()
        assert success == 1
        # Append file info for PVD file
        files.append((vtu_file_name, time_step))

    # Write the PVD file
    pvd_output_path = workdir / f"{output_name}.pvd"
    write_pvd_file(files, pvd_output_path)
#
#
# def PVD_point_fields(workdir, output_name, times, points, fields_dict):
#     """
#     Create a sequence of VTU files and a PVD file from 3D points and scalar fields over time.
#
#     :param points: Array of 3D coordinates for N points (size N x 3).
#     :param x_field: NumPy array representing the X scalar field (size N x K).
#     :param y_field: NumPy array representing the Y scalar field (size N x K).
#     :param time_labels: Array of K time labels.
#     :param output_dir: Directory to save the VTU and PVD files.
#     """
#     # Ensure the output directory exists
#     os.makedirs(workdir / output_name, exist_ok=True)
#
#     # Create a list to store file paths and time steps
#     files = []
#
#     for it, time  in enumerate(times):
#         # Create a PolyData object
#         mesh = pv.PolyData(points)
#         # Add scalar fields for the current time step
#         for name, data in fields_dict.items():
#             mesh.point_data[name] = data[:, it]
#
#         # Save as VTU file
#         file_path = workdir / output_name / f"step_{it}.vtk"
#         mesh.save(file_path)
#         files.append((file_path, time))
#
#     # Create a PVD file
#     write_pvd_file(files, workdir / f"{output_name}.pvd")


def PVD_eval_field(workdir, times, points, field):
    field_dict = dict(
        pressure_mean=field.mean(axis=-1),
        pressure_std=field.std(axis=-1),
        pressure_min=field.min(axis=-1),
        pressure_max=field.max(axis=-1),
        pressure_q1=np.quantile(field, 0.25, axis=-1),
        pressure_med=np.quantile(field, 0.5, axis=-1),
        pressure_q2=np.quantile(field, 0.75, axis=-1),
    )
    PVD_point_fields(workdir, 'eval_field', times, points, field_dict)

def PVD_data_on_bhset(workdir, bh_set):
    times, points, field = bh_set.projected_data
    points = points.reshape(-1, 3)
    field = field.reshape(-1, *field.shape[2:])
    assert field.shape[0] == points.shape[0]
    assert field.shape[1] == len(times)
    PVD_eval_field(workdir, times, points, field)



def plot_borehole_position(cfg, bh):
    pv.start_xvfb()
    font_size = 20
    pv.global_theme.font.size = font_size
    plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
    #plotter.set_font(font_size=font_size)
    plotter = create_scene(plotter, cfg.geometry)
    lateral = bh.lateral
    add_cylinders(plotter, lateral)
    add_bh(plotter, bh)
    pv.global_theme.font.size = font_size
    for i, part in enumerate(bh.bh_description.split("\n")):
        plotter.add_text(part, position=(500, 60 - 20*(i+1)), font_size=10)
    #plotter.renderer.axes_actor.label_text_property.font_size = font_size

    return plotter


def save_projections(plotter, workdir, fname):

    camera_positions = [
        ([-30, 0, 0], [0, 0, 0], [0, 0, 1]),
        ([0, -30, 0], [0, 0, 0], [0, 0, 1]),
        ([0, 0, 30], [0, 0, 0], [1, 0, 0])
    ]

    out_files = []
    for axis in [0, 1, 2]:
        plotter.camera.position, plotter.camera.focal_point, plotter.camera.up = camera_positions[axis]
        plotter.camera.parallel_projection = True
        plotter.render()

        #f_name = workdir / f"bh_shot_{axis}.svg"
        f_name = workdir / fname
        f_name = f_name.parent / f"{f_name.stem}_{axis}{f_name.suffix}"
        if f_name.suffix == ".png":
              plotter.screenshot(f_name)
        elif f_name.suffix == ".svg":  # or f_name.suffix == ".pdf":      # PDF too large
            plotter.save_graphic(f_name, raster=False)
        out_files.append(f_name)

    return out_files

@attrs.define
class PlotCfg:
    workdir : Path
    cfg : 'dotdict'
    bh_set : 'BoreholeSet'
    chambers : 'Chambers'
    i_bh: int
    opt_packers: List[List['PackerConfig']]
    param_names : List[str]
    show: bool = False


    def plot_borehole(self):
        plotter = plot_borehole_position(self.cfg, self.bh_set.boreholes[self.i_bh])
        return save_projections(plotter, self.workdir, f"bh_{self.i_bh}.png")

    def plot_chamber_data(self, sensitivities, f_name, vmin):
        """
        Plot some precomputed chamber data for selected chamber sizes.
        :return:
        """
        n_params = len(self.param_names)
        n_points = self.chambers.n_points
        sizes = [4]
        range_sens = lambda b, s: sensitivities[self.chambers.index[b, min(b + s, n_points)]]
        chamber_data = [[ range_sens(i_begin, size)
                         for i_begin in range(0, n_points - size)]
                            for i_size, size in enumerate(sizes)]

        # flatten
        chamber_array = np.array([i  for l in chamber_data for i in l])

        vmax = np.max(chamber_array)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                                   norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        sm.set_array([])

        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4), sharex=True)
        color_values = []
        for i_param, label in enumerate(self.param_names):
            ax = axes[i_param]
            for i_size, size in enumerate(sizes):
                point_values = [l[i_param] for l in chamber_data[i_size]]
                point_values = np.maximum(np.array(point_values), vmin)
                x_pos = size // 2 + np.arange(0, n_points - size)
                colors = sm.to_rgba(point_values)
                assert len(x_pos) == len(colors)
                width = np.ones(len(x_pos))
                ax.broken_barh(list(zip(x_pos, width)), (i_size+0.1, 0.8), facecolors=colors)

            #ax.set_ylabel(label, rotation=0, horizontalalignment='right', verticalalignment='center')
            #ax.set_yticks([0.5, 1.5, 2.5])
            #ax.set_yticklabels(['2', '4', '8'])
            #ax.set_xticks
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        xticks = list(range(0, 80, 10))
        ax.set_xticks(xticks)
        ax.set_xticklabels(list(range(0, 40, 5)))
        axes[0].set_title("Size", pad=-20)  # Title above the first axis

        # Create a common colorbar for all subplots
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # x-position, y-position, width, height
        #ticks = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0, vmaxmax_sobol]
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', cax = cbar_ax)

        #plt.tight_layout()
        fname = self.workdir / f_name
        fig.savefig(fname)
        if self.show:
            plt.show()
        return fname

    def plot_mean_pressure(self):
        chamber_pressures = self.chambers.cumul_bh_data[4:, :, :] - self.chambers.cumul_bh_data[:-4, :, :]
        # (n_points - 1, n_times, n_samples)
        # Calculate the mean over the last dimension (samples)
        mean_pressures = np.mean(chamber_pressures, axis=2)   # mean over samples
        selected_pressures = mean_pressures[::4, :]
        fig = self._plot_time_fun(selected_pressures)
        fname = self.workdir / "mean_time_fun.pdf"
        fig.savefig(fname)
        return fname

    def plot_q90_pressure(self):
        chamber_pressures = self.chambers.cumul_bh_data[4:, :, :] - self.chambers.cumul_bh_data[:-4, :, :]
        # (n_points - 1, n_times, n_samples)
        # Calculate the mean over the last dimension (samples)
        mean_pressures = np.quantile(chamber_pressures, q=0.9, axis=2)   # mean over samples
        selected_pressures = mean_pressures[::4, :]
        fig = self._plot_time_fun(selected_pressures)
        fname = self.workdir / "q90_time_fun.pdf"
        fig.savefig(fname)
        return fname

    def _plot_time_fun(self, selected_pressures):
        """
        Goal: check projected pressures on the borehole
        :return:
        """

        # Create a colormap
        cmap = plt.cm.gist_rainbow   # better discriminatino of points then viridis
        colors = cmap(np.linspace(0, 1, selected_pressures.shape[0]))

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, pressure in enumerate(selected_pressures):
            ax.plot(pressure, color=colors[i], label=f'Point {4*i}')

        # Create a ScalarMappable for the color bar
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, selected_pressures.shape[0]))
        sm.set_array([])  # This line is necessary for ScalarMappable to work with colorbar

        # Add a color bar associated with the axis
        cbar = fig.colorbar(sm, ax=ax, ticks=range(0, selected_pressures.shape[0]))
        cbar.ax.set_yticklabels([f'Point {4*i}' for i in range(selected_pressures.shape[0])])

        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Pressure')
        ax.set_title('Mean Pressure Over Time for Every 4th Point')
        return fig

    def plot_relative_residual(self):
        arr_copy = self.chambers.bh_data
        orig_arr = self.chambers.orig_bh_data
        #outlier_mask = self.chambers.outlier_mask

        arr_copy = np.maximum(arr_copy, 0.1)
        # Compute the time mean on arr_copy
        time_mean_copy = np.nanmean(arr_copy, axis=1, keepdims=True)

        # Normalize orig_arr and arr_copy by dividing by the time mean of arr_copy
        normalized_orig_arr = orig_arr / time_mean_copy
        normalized_arr_copy = arr_copy / time_mean_copy

        # Calculating medians, quartiles, min, and max over times for the normalized copy
        min_values, Q1, median, Q3,  max_values = [
            np.nanpercentile(normalized_arr_copy, q, axis=(1,2))
            for q in [0, 25, 50, 75, 100]
            ]

        n_points, n_times, _ = orig_arr.shape
        colors = plt.cm.rainbow(np.linspace(0, 1, n_times))  # Different colors for each time

        # Create plots
        fig, ax1  = plt.subplots(1,1, figsize=(12, 12))
        for i in range(n_points):
            #for j in range(n_times):

            # Plot min and max values as thin vertical lines
            ax1.bar(i, max_values[i] - min_values[i], width=0.1, bottom=min_values[i], color='black', align='center')
            ax1.bar(i, Q3[i] - Q1[i], width=0.5, bottom=Q1[i], color='blue', align='center')
            ax1.bar(i, 0.01, width=0.6, bottom=median[i]-0.005, color='red', align='center')

        ax = ax1
        #ax.set_yscale('log')
        ax.set_xlabel('Point')
        ax.set_ylabel('Log of (p / p.mean)')
        #ax.legend(title='Time', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticks(range(n_points))
        #ax.grid(True)
        ax.set_title('Relative Residuals - box plot over times and samples')
        #ax2.set_title('Relative Residuals - box plot over samples of time sequence variance')
        #fig.tight_layout()
        fname = self.workdir / "residual_range.pdf"
        fig.savefig(fname)
        return fname

    # Example usage
    # plot_boxplot_style_changes(orig_arr, arr_copy, outlier_mask)

    # def plot_relative_residual(self):
    #     """
    #     Goal: check projected pressures on the borehole
    #     :return:
    #     """
    #     pressures = self.chambers.bh_data
    #     # Calculate the relative residual with respect to mean over samples and points
    #     #
    #     mean_pressures = pressures.mean(axis=(0,2))
    #     #std_pressures = pressures.std(axis=2)
    #     relative_pressure = pressures / mean_pressures[np.newaxis, :, np.newaxis]
    #     rel_p_time_mean = relative_pressure.mean(axis=1)
    #     rel_p_time_std = relative_pressure.mean(axis=1)
    #     rel_p_time_mean_QX = [rel_p_time_mean.percentile(p, axis=2)
    #
    #     # Number of points and times
    #     n_points, n_times, _ = pressures.shape
    #
    #     # Set up the color map for different times
    #     colors = plt.cm.rainbow(np.linspace(0, 1, n_times))
    #
    #     # Create the figure object
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #
    #     # Plotting
    #     for i in range(n_points):
    #         for j in range(n_times):
    #             # Deterministic jitter in x-coordinates based on time index
    #             y = relative_residual[i, j, :]
    #             x = np.full_like(y, i + 0.1 * (j - n_times / 2))
    #
    #             ax.scatter(x, y, color=colors[j], label=f'Time {j}' if i == 0 else "")
    #
    #     ax.set_xlabel('Point')
    #     ax.set_ylabel('Relative Residual')
    #     ax.set_title('Grouped Scatter Plot of Relative Residuals')
    #     ax.legend(title='Time', bbox_to_anchor=(1.05, 1), loc='upper left')
    #     ax.set_xticks(range(n_points))
    #     ax.grid(True)
    #     fig.tight_layout()
    #     fname = self.workdir / "residual.pdf"
    #     fig.savefig(fname)
    #     return fname

    def plot_stacked_barplot(self, ax, data_array, packers, cmap, col, vec_objective, sm_obj):
        """
        Plot a stacked bar plot for the given data.

        Args:
        ax (matplotlib.axes.Axes): Axis object to plot on.
        data_array (numpy.array): Array of shape (n_chambers, n_params).
        packers (numpy.array): Array of shape (n_chambers + 1,) providing boundaries.
        """
        # Invert the y-axis
        max_x = np.max(packers)
        m_size = 5

        n_chambers, n_params = data_array.shape
        for i in range(n_params):
            # plot objective, only for parameter that match the column i.e. against which the column is optimized)
            if i == col:
                colors = sm_obj.to_rgba(vec_objective[i, :])
                ax.scatter([max_x + 2, max_x + 3], [i, i], m_size, color=colors)

            # plot chambers
            for j in range(n_chambers):
                width = packers[j + 1] - packers[j]
                color = cmap.to_rgba(data_array[j, i])
                ax.barh(i, width, left=packers[j], color=color)
                ax.axvline(x=packers[j], color='black', linewidth=0.5, ymin=i / n_params,
                           ymax=(i + 1) / n_params)
                if col == 0:
                    ax.set_yticks(range(n_params))
                    ax.set_yticklabels(self.param_names)

        #ax.set_yticks([])

        #ax.set_xlabel('Chamber Length')

    def plot_best_packers_st(self):
        data_array = [[p.st_values[:,:]  for p in l] for l in self.opt_packers]
        packers = [[p.packers for p in l] for l in self.opt_packers]
        vec_objective = [[p.opt_values for p in l] for l in self.opt_packers]
        fig = self._plot_best_packers(data_array, packers, vec_objective)
        fname = self.workdir / "optim_packers_st.pdf"
        fig.savefig(fname)
        return fname

    # def plot_best_packers_full(self):
    #     data_array = [[p.sobol_indices[:,:,0]  for p in l] for l in self.opt_packers]
    #     packers = [[p.packers for p in l] for l in self.opt_packers]
    #     vec_objective = [[p.opt_values for p in l] for l in self.opt_packers]
    #     fig = self._plot_best_packers(data_array, packers, vec_objective)
    #     fname = self.workdir / "optim_packers_full.pdf"
    #     fig.savefig(fname)
    #     return fname

    def _plot_best_packers(self, data_array, packers, vec_objective):
        """
        Goal: show the packer positions of the bes variants, compare indices
        """
        n_params = self.chambers.n_params
        n_variants = len(data_array[0])
        threshold = 1e-5
        data_array = np.maximum(data_array, threshold)
        max_sobol = np.max(data_array)
        print("Max sobol: ", max_sobol)

        cmap = plt.cm.jet   # better discriminatino of points then viridis
        sm = ScalarMappable(cmap=cmap, norm=mcolors.LogNorm(threshold, max_sobol))
        sm.set_array([])  # This line is necessary for ScalarMappable to work with colorbar

        cmap = plt.cm.gist_rainbow
        sm_objective = ScalarMappable(cmap=cmap, norm=mcolors.LogNorm(0.001, 1.0))
        sm_objective.set_array([])  # This line is necessary for ScalarMappable to work with colorbar

        #sm = combined_mapper(max_sobol, threshold)

        fig, axs = plt.subplots(n_variants, n_params, figsize=(20, 15),  sharex='col', sharey='row')
                                #constrained_layout=True,)

        for i in range(n_variants):
            for j in range(n_params):
        # for i in range(1):
        #     for j in range(1):
                ax = axs[i, j] #if n_variants > 1 else axs[j]
                objective = vec_objective[j][i]
                self.plot_stacked_barplot(ax, data_array[j][i], packers[j][i], sm, j, objective, sm_objective)

                if i == 0:
                    # Set column labels (parameter names) for the top row
                    ax.set_title(self.param_names[j])

                if j == 0:
                    ax.invert_yaxis()
                    # Set row labels (variant indices) for the first column
                    ax.set_ylabel(f'Variant {i + 1}')

        ticks = [1e-5,  3e-5, 1e-4,  3e-4, 1e-3,  3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, max_sobol]
        cbar_ax = fig.add_axes([0.92, 0.05, 0.01, 0.9])  # x-position, y-position, width, height
        # ticks = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0, vmaxmax_sobol]
        cbar = fig.colorbar(sm, ax=axs, orientation='vertical', #fraction=0.015, pad=0.05,
                            ticks=ticks, cax=cbar_ax)
        cbar_ax = fig.add_axes([0.96, 0.05, 0.01, 0.9])  # x-position, y-position, width, height
        cbar = fig.colorbar(sm_objective, ax=axs, orientation='vertical', #fraction=0.015, pad=0.05,
                            ticks=ticks, cax=cbar_ax)

        #fig.tight_layout()
        return  fig

    def all(self):
        plots = [
            *self.plot_borehole(),
            self.plot_chamber_data(self.chambers.chambers_sensitivities,
                                   f_name="chambers_sa.pdf", vmin=1e-5),
            self.plot_chamber_data(self.chambers.chambers_norm_sensitivities,
                                   f_name="chambers_norm_sa.pdf", vmin=1e-2),
            self.plot_mean_pressure(),
            self.plot_q90_pressure(),
            self.plot_relative_residual(),
            self.plot_best_packers_st(),
            # self.plot_best_packers_full()
        ]
        bcommon.create_combined_pdf(plots, self.workdir / "summary.pdf")


# def create_custom_colormap(max_value=2.0, threshold=1e-4, n_points=20):
#     """
#     Create a custom colormap with a logarithmic transition from threshold to 1.0.
#
#     Args:
#     threshold (float): Threshold value up to which the color is white.
#     max_value (float): Maximum value for the transition to green.
#     n_points (int): Number of points to approximate the logarithmic transition.
#
#     Returns:
#     LinearSegmentedColormap: The custom colormap.
#     """
#     # Ensure the threshold and max_value are within a valid range
#     threshold = min(max(threshold, 1e-10), max_value)  # Avoid division by zero in log
#
#     # Generate points for logarithmic transition
#     log_min = np.log(threshold)
#     log_max = np.log(1)
#     log_points = np.exp(np.linspace(log_min, log_max, n_points))
#
#     # Create the colormap data
#     cdict = {'red': [], 'green': [], 'blue': []}
#     last_val = 0
#     for val in log_points:
#         norm_val = val / max_value
#         r = min(val, 1)
#         g = min(max(val - 1, 0), 1)
#         cdict['red'].append((norm_val, r, r))
#         cdict['green'].append((norm_val, g, g))
#         cdict['blue'].append((norm_val, 0, 0))
#         last_val = norm_val
#
#     # Add final point for max_value
#     cdict['red'].append((1.0, 0.0, 0.0))
#     cdict['green'].append((1.0, 1.0, 1.0))
#     cdict['blue'].append((1.0, 0.0, 0.0))
#
#     # Create the colormap
#     custom_cmap = LinearSegmentedColormap('custom_colormap', cdict)
#     return custom_cmap


def create_value_mapper(threshold=1e-3, max_value=1.0):
    """
    Create a value mapping function for the custom colormap.

    Args:
    threshold (float): Threshold value for the logarithmic transition.
    max_value (float): Maximum value of the data.

    Returns:
    function: A function that maps input values to the normalized range (0, 1).
    """
    T = threshold
    M = max_value
    # Calculate constants a, b, and c
    b = (T + M - 2) / (1 - T)
    a = 1 / np.log((T + M - 1) ** 2)
    c = 1 - b * T

    # Define the function
    def func(x):
        log_arg = b * x + c
        return a * np.log(log_arg)

    def inv(y):
        return (np.exp(y / a) - c) / b

    return func, inv

def create_custom_colormap():
    """
    Create a custom colormap for the transformed value range.

    Returns:
    LinearSegmentedColormap: The custom colormap.
    """
    cdict = {
        'red':   ((0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)),
        'green': ((0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
        'blue':  ((0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    }
    return mcolors.LinearSegmentedColormap('custom_colormap', cdict)

def combined_mapper(max_value, threshold=1e-3):
    _forward, _inverse = create_value_mapper(threshold, max_value)
    value_mapper = mcolors.FuncNorm((_forward, _inverse), vmin=0, vmax=max_value)

    custom_cmap = create_custom_colormap()
    #norm = mcolors.Normalize(vmin=0, vmax=max_value)
    #cmapper = lambda x : custom_cmap(value_mapper(norm.inverse(x)))
    sm = plt.cm.ScalarMappable(norm=value_mapper, cmap=custom_cmap)
    return sm

def plot_sensitivity_histograms(sensitivities, param_names):
    import scipy
    """
      Plots histograms for an array `sensitivities` based on the derivative of a smoothed empirical CDF
      (using Gaussian kernel density estimation) for each parameter, with logarithmic scale and fixed x range.

      :param sensitivities: numpy array of shape (n_samples, n_params)
      :param param_names: list of parameter names of length n_params
      """
    n_params = sensitivities.shape[1]

    # Using a discrete color map (tab10)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_params)]

    fig, ax = plt.subplots(figsize=(12, 6))

    x_grid = np.geomspace(1e-10, 2, 1000)  # Fixed x range

    for i in range(n_params):
        data = sensitivities[:, i]

        # Gaussian Kernel Density Estimation
        kde = scipy.stats.gaussian_kde(data, bw_method='silverman')
        pdf = kde.evaluate(x_grid)
        ax.plot(x_grid, pdf, color=colors[i % len(colors)], label=param_names[i])

    ax.set_xscale('log')  # Setting logarithmic scale
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e6)
    ax.legend()
    ax.set_xlabel('Parameter Value (log scale)')
    ax.set_ylabel('Density')
    ax.set_title('Smoothed Histograms of Sensitivities on Log Scale')

    plt.show()