import numpy as np
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk
import matplotlib.pyplot as plt
import vtk
import os
from xml.etree import ElementTree as ET


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

    # i,j = 1, 2
    # iangle_norm = (i / bh_set.n_y_angles, j / bh_set.n_z_angles)
    # for i_bh in bh_set.angles_table[i][j]:
    #     add_bh(plotter, iangle_norm, bh_set, i_bh)

    return plotter

def add_bh(plotter, angle_norm, bh_set, i_bh):
    p_w, dir, p_tr = bh_set.bh_list[i_bh]
    p_w = bh_set.transform(p_w)
    p_tr = bh_set.transform(p_tr)
    points, bounds = bh_set.point_lines
    p_begin = points[i_bh, bounds[i_bh][0], :]
    p_end = points[i_bh, bounds[i_bh][1] - 1, :]

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






def plot_chamber_data(chambers, param_labels):
    n_params = len(param_labels)
    n_points = chambers.n_points
    sizes = [2, 4, 8]

    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3     * n_params), sharex=True)
    color_values = []
    for i_param, label in enumerate(param_labels):
        ax = axes[i_param]

        for i_size, size in enumerate(sizes):
            for i_begin in range(0, n_points - size):
                # Calculate the color based on the chamber values
                i_end = min(i_begin + size, n_points)
                i_pos = (i_begin + i_end) // 2
                chamber_data = chambers.chamber(i_begin, i_end)
                if chamber_data is None:
                    continue

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
    plt.show()