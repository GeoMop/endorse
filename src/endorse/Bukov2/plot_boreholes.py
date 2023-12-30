from typing import *
import numpy as np
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import vtk
import os
from xml.etree import ElementTree as ET
import attrs
from pathlib import Path
from endorse.Bukov2 import bukov_common as bcommon
from endorse.Bukov2 import bh_chambers


def create_scene(plotter, cfg_geometry):
    cfg = cfg_geometry.main_tunnel
    # Create a plotting object

    # L5
    x_half = cfg.width / 2
    y_half = cfg.length / 2

    box = pv.Box(bounds=(-x_half, +x_half, -y_half -10, +y_half -10, 0, cfg.height))
    plotter.add_mesh(box, color='grey' , opacity=0.7)

    plotter.add_axes()
    plotter.show_bounds(grid='front', all_edges=True)
    return plotter

def add_cylinders(plotter, bh_set: 'BoreholeSet'):
    # Create a horizontal cylinder
    r, l0, l1 = bh_set.avoid_cylinder
    avoid_cylinder = pv.Cylinder(center=bh_set.transform([0.5 * l0 + 0.5 * l1, 0, 0]), direction=(1, 0, 0), radius=r, height=l1-l0)
    plotter.add_mesh(avoid_cylinder, color='red')

    r, l0, l1 = bh_set.active_cylinder
    active_cylinder = pv.Cylinder(center=bh_set.transform([0.5 * l0 + 0.5 * l1, 0, 0]), direction=(1, 0, 0), radius=r, height=l1-l0)
    plotter.add_mesh(active_cylinder, color='grey', opacity=0.1)


def plot_bh_set(plotter, bh_set: 'BoreholeSet'):
    for i in range(bh_set.n_y_angles):
        for j in range(bh_set.n_z_angles):
            for i_bh in bh_set.angles_table[i][j]:
                add_bh(plotter, bh_set, i_bh)

    # i,j = 1, 2
    # iangle_norm = (i / bh_set.n_y_angles, j / bh_set.n_z_angles)
    # for i_bh in bh_set.angles_table[i][j]:
    #     add_bh(plotter, iangle_norm, bh_set, i_bh)

    return plotter



def add_bh(plotter, bh_set, i_bh):
    p_w, dir, p_tr = bh_set.bh_list[i_bh]
    p_w = bh_set.transform(p_w)
    p_tr = bh_set.transform(p_tr)
    points, bounds = bh_set.point_lines
    p_begin = points[i_bh, bounds[i_bh][0], :]
    p_end = points[i_bh, bounds[i_bh][1] - 1, :]

    i, j, k = bh_set.angle_ijk(i_bh)
    angle_norm = (i / bh_set.n_y_angles, j / bh_set.n_z_angles)

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


@attrs.define
class PlotCfg:
    workdir : Path
    cfg : 'dotdict'
    bh_set : 'BoreholeSet'
    chambers : 'Chambers'
    i_bh: int
    opt_packers: List[List[bh_chambers.PackerConfig]]
    param_names : List[str]
    show: bool


    def plot_borehole_position(self):
        pv.start_xvfb()
        plotter = pv.Plotter(off_screen=True)
        plotter = create_scene(plotter, self.cfg.geometry)
        add_cylinders(plotter, self.bh_set)
        add_bh(plotter, self.bh_set, self.i_bh)
        plotter.add_text(self.bh_set.bh_description(self.i_bh))

        camera_positions = [
            ([-50, 0, 0], [0, 0, 0], [0, 0, 1]),
            ([0, -50, 0], [0, 0, 0], [0, 0, 1]),
            ([0, 0, 50], [0, 0, 0], [0, 1, 0])
        ]
        resolution = (1920, 1080)

        out_files = []
        for axis in range(3):
            plotter.camera.position, plotter.camera.focal_point, plotter.camera.up = camera_positions[axis]
            plotter.camera.parallel_projection = True
            f_name = self.workdir / f"bh_shot_{axis}.png"
            plotter.screenshot(f_name, window_size=resolution)
            out_files.append(f_name)

        if self.show:
            plotter.show()

        return out_files


    def plot_chamber_data(self):
        """
        Plot some precomputed chamber data for selected chamber sizes.
        :return:
        """
        n_params = len(self.param_names)
        n_points = self.chambers.n_points
        sizes = [2, 4, 8]

        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3     * n_params), sharex=True)
        color_values = []
        for i_param, label in enumerate(self.param_names):
            ax = axes[i_param]

            for i_size, size in enumerate(sizes):
                for i_begin in range(0, n_points - size):
                    # Calculate the color based on the chamber values
                    i_end = min(i_begin + size, n_points)
                    i_pos = (i_begin + i_end) // 2
                    chamber_data = self.chambers.chamber(i_begin, i_end)
                    #if chamber_data is None:
                    #    continue

                    value = chamber_data[i_param]
                    color_values.append(value)
                    # Add a horizontal stripe to the plot
                    ax.broken_barh([(i_pos, 1)], (i_size+0.1, 0.8), facecolors=plt.cm.viridis(value))


            ax.set_ylabel(label, rotation=0, horizontalalignment='right', verticalalignment='center')
            ax.set_yticks([0.5, 1.5, 2.5])
            ax.set_yticklabels(['2', '4', '8'])


        axes[0].set_title("Size", pad=-20)  # Title above the first axis

        # Create a common colorbar for all subplots
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=min(color_values), vmax=max(color_values)))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # x-position, y-position, width, height
        fig.colorbar(sm, cax=cbar_ax)

        #plt.tight_layout()
        fname = self.workdir / "chambers_sa.pdf"
        fig.savefig(fname)
        if self.show:
            plt.show()
        return fname

    def plot_mean_time_fun(self):
        """
        Goal: check projected pressures on the borehole
        :return:
        """
        pressures = self.chambers.bh_data
        # (n_points - 1, n_times, n_samples)
        # Calculate the mean over the last dimension (samples)
        mean_pressures = np.mean(pressures, axis=2)
        selected_pressures = mean_pressures[::4]

        # Create a colormap
        cmap = plt.cm.jet   # better discriminatino of points then viridis
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

        fname = self.workdir / "mean_time_fun.pdf"
        fig.savefig(fname)
        if self.show:
            plt.show()
        return fname

    def plot_relative_residual(self):
        arr_copy = self.chambers.bh_data
        orig_arr = self.chambers.orig_bh_data
        outlier_mask = self.chambers.outlier_mask
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
        fig, (ax1, ax2)  = plt.subplots(2,1, figsize=(12, 12))
        for i in range(n_points):
            for j in range(n_times):
                # Scatter plot of outliers
                outlier_data = normalized_orig_arr[i, j, outlier_mask[i, j]]
                ax2.scatter([i + 0.028 * j] * len(outlier_data), outlier_data, color=colors[j], alpha=0.5, s=1)

            # Plot min and max values as thin vertical lines
            ax1.bar(i, max_values[i] - min_values[i], width=0.1, bottom=min_values[i], color='black', align='center')
            ax1.bar(i, Q3[i] - Q1[i], width=0.5, bottom=Q1[i], color='blue', align='center')
            ax1.bar(i, 0.01, width=0.6, bottom=median[i]-0.005, color='red', align='center')

        for ax in [ax1, ax2]:
            ax.set_yscale('log')
            ax.set_xlabel('Point')
            ax.set_ylabel('Log of (p / p.mean)')
            #ax.legend(title='Time', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xticks(range(n_points))
            #ax.grid(True)
        ax1.set_title('Relative Residuals - data range')
        ax2.set_title('Relative Residuals - outliers')
        fig.tight_layout()
        fname = self.workdir / "residual.pdf"
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

    def plot_stacked_barplot(self, ax, data_array, packers, cmap):
        """
        Plot a stacked bar plot for the given data.

        Args:
        ax (matplotlib.axes.Axes): Axis object to plot on.
        data_array (numpy.array): Array of shape (n_chambers, n_params).
        packers (numpy.array): Array of shape (n_chambers + 1,) providing boundaries.
        """
        n_chambers, n_params = data_array.shape
        for i in range(n_params):
            for j in range(n_chambers):
                width = packers[j + 1] - packers[j]
                color = cmap.to_rgba(data_array[j, i])
                ax.barh(i, width, left=packers[j], color=color)

        #ax.set_yticks([])

        #ax.set_xlabel('Chamber Length')

    def plot_best_packers(self):
        """
        Goal: show the packer positions of the bes variants, compare indices
        """
        data_array = [[p.sobol_indices[:,:,0]  for p in l] for l in self.opt_packers]
        packers = [[p.packers for p in l] for l in self.opt_packers]
        n_params = self.chambers.n_params
        n_variants = len(data_array[0])
        max_sobol = np.max(data_array)
        print("Max sobol: ", max_sobol)
        sm = combined_mapper(max_sobol)

        fig, axs = plt.subplots(n_variants, n_params, figsize=(20, 15),  sharex='col', sharey='row')
                                #constrained_layout=True,)

        for i in range(n_variants):
            for j in range(n_params):
        # for i in range(1):
        #     for j in range(1):
                ax = axs[i, j] #if n_variants > 1 else axs[j]
                self.plot_stacked_barplot(ax, data_array[j][i], packers[j][i], sm)

                if i == 0:
                    # Set column labels (parameter names) for the top row
                    ax.set_title(self.param_names[j])

                if j == 0:
                    # Set row labels (variant indices) for the first column
                    ax.set_ylabel(f'Variant {i + 1}')
                    ax.set_yticks(range(n_params))
                    ax.set_yticklabels(self.param_names)

        ticks = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0, max_sobol]
        cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.03, pad=0.05,
                            ticks=ticks)

        #fig.tight_layout()
        fname = self.workdir / "optim_packers.pdf"
        fig.savefig(fname)
        return fname

    def all(self):
        plots = [
            *self.plot_borehole_position(),
            self.plot_chamber_data(),
            self.plot_mean_time_fun(),
            self.plot_relative_residual(),
            self.plot_best_packers()
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
        return a * np.log(b * x + c)

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