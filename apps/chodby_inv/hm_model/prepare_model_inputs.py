import numpy as np
import meshio
import pyvista as pv
import pathlib
import csv

from endorse import common
import chodby_inv.input_data as input_data
import boreholes
from chodby_inv.piezo.piezo_canonic import to_datetime, linear_time

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




def prepare_borehole_sources():
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


def prepare_excavation_functions():
    # Create strings defining the FieldTimeFunction data of excavation progress
    # in the North and South test chambers to be used in Flow123d simulation.

    events_cfg = common.config.load_config(input_data.events_yaml)
    blasts = events_cfg['blasts']
    excavation = events_cfg['excavation']
    hm_model = events_cfg['hm_model']
    days_shift = boreholes.excavation_days_shift(events_cfg)
    delta = 0.001
    final_stationing = 10
    final_stationing_new = 10.1

    values_n = [ {"t":0, "value":0} ]
    values_s = [ {"t":0, "value":0} ]

    last_stationing_n = 0
    last_stationing_s = 0

    for blast in blasts:
        t = float( linear_time([blast['datetime']], excavation)[0] + days_shift )
        if blast.side == "N":
            values_n.append( { "t":t-delta, "value":last_stationing_n } )
            if blast.face_stationing == final_stationing:
                last_stationing_n = -final_stationing_new
            else:
                last_stationing_n = -blast.face_stationing
            values_n.append( { "t":t, "value":last_stationing_n } )
        elif blast.side == "S":
            values_s.append( { "t":t-delta, "value":last_stationing_s } )
            if blast.face_stationing == final_stationing:
                last_stationing_s = final_stationing_new
            else:
                last_stationing_s = blast.face_stationing
            values_s.append( { "t":t, "value":last_stationing_s } )

    values_n.append( { "t":hm_model.days_simulation, "value":last_stationing_n } )
    values_s.append( { "t":hm_model.days_simulation, "value":last_stationing_s } )

    return values_n, values_s

