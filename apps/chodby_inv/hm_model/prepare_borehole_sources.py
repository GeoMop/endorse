import numpy as np
import meshio
import pyvista as pv
import boreholes
import pathlib

import chodby_inv.input_data as input_data
work_dir = input_data.work_dir
module_dir = pathlib.Path(__file__).parent


def process_gmsh_tetrahedral_mesh(input_file, output_file, points, values_dict):
    """
    Loads a GMSH tetrahedral mesh, finds elements crossing the lines (p1, p2) from points array,
    and writes a new VTU file with an additional element data fields in values_dict.
    """
    # Load the mesh
    mesh = meshio.read(input_file)

    # print(mesh.field_data)
    # print(mesh.cell_data['gmsh:physical'][1].shape)

    # Ensure the mesh contains tetrahedral elements
    if "tetra" not in mesh.cells_dict:
        raise ValueError("Mesh must contain tetrahedral elements.")

    # Convert to PyVista UnstructuredGrid
    pv_mesh = pv.UnstructuredGrid({pv.CellType.TETRA: mesh.cells_dict["tetra"]}, mesh.points)

    # Get elements
    tetrahedra = mesh.cells_dict["tetra"]

    # Find elements that intersect the given line
    cell_values = dict()
    for name in values_dict.keys():
        cell_values[name] = np.zeros(len(tetrahedra), dtype=int)
    for i,(p1,p2) in enumerate(points):
        intersecting_cells = pv_mesh.find_cells_intersecting_line(p1, p2)
        for name,value in values_dict.items():
            cell_values[name][intersecting_cells] = value[i]

    # Create element data
    element_data = {name: [values] for name,values in cell_values.items()}

    # Write new mesh in VTU format
    meshio.write(output_file, meshio.Mesh(points=mesh.points, cells=[("tetra", tetrahedra)], cell_data=element_data), binary=True)




# Example usage - create a VTU file that contains two constant fields:
# sigma=1 and p_ref=40 in elements intersecting borehole chambers.

input_mesh_file = module_dir / "L5_mesh_6_healed.msh"
output_mesh_file = work_dir / "flow_sigma.vtu"

# Initialize point and field lists
bhs = boreholes.Boreholes()
points = []
sigmas = []
pressures = []
for bi in range(bhs.n_boreholes):
    for ci in range(bhs.n_chambers(bi)):
        points.append((bhs.chamber_start(bi, ci), bhs.chamber_end(bi, ci)))
        sigmas.append(1)
        pressures.append(40)

# Create the VTU file
process_gmsh_tetrahedral_mesh(input_mesh_file, output_mesh_file, points, {'sigma':sigmas, 'p_ref':pressures})
