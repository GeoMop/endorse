import numpy as np
import meshio
import pyvista as pv
import boreholes
from ..input_data import bh_cfg_file


def process_gmsh_tetrahedral_mesh(input_file, output_file, points, values_dict):
    """
    Loads a GMSH tetrahedral mesh, finds elements crossing the line (p1, p2),
    and writes a new VTU file with an additional element data field 'sigma'.
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
    meshio.write(output_file, meshio.Mesh(points=mesh.points, cells=[("tetra", tetrahedra)], cell_data=element_data), binary=False)




# Example usage
input_mesh_file = "L5_mesh_6_healed.msh"
output_mesh_file = "flow_sigma.vtu"

bhs = boreholes.Boreholes(bh_cfg_file)
points = []
sigmas = []
pressures = []
for bi in range(bhs.n_boreholes):
    for ci in range(bhs.n_chambers(bi)):
        points.append((bhs.chamber_start(bi, ci), bhs.chamber_end(bi, ci)))
        sigmas.append(1)
        pressures.append(40)

process_gmsh_tetrahedral_mesh(input_mesh_file, output_mesh_file, points, {'sigma':sigmas, 'p_ref':pressures})
