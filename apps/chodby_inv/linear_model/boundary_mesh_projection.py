import meshio
import pyvista as pv
import numpy as np


def gmsh_to_pyvista(gmsh_filename):
    """
    Reads a GMSH file using meshio and converts it to a PyVista PolyData mesh.
    
    Assumes that the mesh contains triangle cells.
    
    Parameters:
        gmsh_filename (str): Path to the GMSH file (e.g., .msh).
        
    Returns:
        pv.PolyData: The mesh as a PyVista PolyData object.
    """
    # Read the mesh with meshio.
    mesh = meshio.read(gmsh_filename)
    points = mesh.points

    # Look for triangle cells.
    triangle_cells = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangle_cells = cell_block.data
            break

    if triangle_cells is None:
        raise ValueError("No triangle cells found in the GMSH file.")

    # PyVista expects a faces array formatted as:
    # [n_points, p0, p1, p2, n_points, p0, p1, p2, ...]
    # For triangles, n_points is 3.
    faces = np.hstack([np.full((triangle_cells.shape[0], 1), 3, dtype=np.int64),
                       triangle_cells]).flatten()

    return pv.PolyData(points, faces)


def project_gmsh_to_smooth(gmsh_mesh, smooth_mesh, max_distance=1.0):
    """
    Projects the nodes of the detailed GMSH mesh onto a smooth reference surface
    by moving each node along its normal until it intersects the smooth mesh.
    
    Uses PyVista's multi_ray_trace to compute all intersections at once.
    
    Parameters:
        gmsh_mesh (pv.PolyData): The detailed mesh (from GMSH) as a PyVista mesh.
        smooth_mesh (pv.PolyData): The smooth reference surface (e.g., from a VTK file).
        max_distance (float): The maximum distance along the normal to search for an intersection.
        
    Returns:
        pv.PolyData: The updated gmsh_mesh with projected node positions.
    """
    # Compute point normals if not already present.
    if 'Normals' not in gmsh_mesh.point_data:
        gmsh_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    normals = gmsh_mesh.point_data['Normals']

    origins = gmsh_mesh.points
    # Scale normals to the desired ray length.
    directions = normals * max_distance

    # Use multi_ray_trace to compute intersections for all rays at once.
    # It returns a tuple (intersections, ray_ids).
    intersections, ray_ids = smooth_mesh.multi_ray_trace(origins, directions)
    
    new_points = gmsh_mesh.points.copy()
    # Update nodes that found an intersection.
    for pt, rid in zip(intersections, ray_ids):
        new_points[rid] = pt

    gmsh_mesh.points = new_points
    return gmsh_mesh


def pyvista_to_gmsh(pv_mesh, gmsh_filename):
    """
    Converts a PyVista mesh (PolyData) to the GMSH file format using meshio.
    
    Assumes that the mesh is made of triangles.
    
    Parameters:
        pv_mesh (pv.PolyData): The PyVista mesh to be converted.
        gmsh_filename (str): Output file name for the GMSH mesh (e.g., "output.msh").
    """
    # Get the points.
    points = pv_mesh.points

    # PyVista stores faces as a 1D array: [n, p0, p1, p2, n, p0, p1, p2, ...]
    # For a triangle mesh, each face should start with a 3.
    faces_flat = pv_mesh.faces
    if faces_flat.size % 4 != 0:
        raise ValueError("Unexpected face data format; expected triangles with 4 integers per face.")

    faces = faces_flat.reshape((-1, 4))
    # Check that all faces are triangles.
    if not np.all(faces[:, 0] == 3):
        raise ValueError("The provided PyVista mesh is not made up solely of triangles.")

    # Extract only the connectivity (vertex indices) for each triangle.
    triangle_cells = faces[:, 1:]
    
    # Create a meshio Mesh object.
    mesh = meshio.Mesh(points, cells=[("triangle", triangle_cells)])
    
    # Write out the mesh in GMSH format.
    meshio.write(gmsh_filename, mesh, file_format="gmsh")
    print(f"Mesh saved in GMSH format to: {gmsh_filename}")


if __name__ == "__main__":
    # Input filenames.
    gmsh_input_filename = "gmsh_mesh.msh"      # Input GMSH mesh file.
    smooth_filename = "smooth_mesh.vtk"        # Smooth reference mesh in VTK format.
    
    # Output filename.
    gmsh_output_filename = "projected_gmsh_mesh.msh"
    
    # Convert the GMSH file to a PyVista mesh.
    gmsh_mesh = gmsh_to_pyvista(gmsh_input_filename)
    print(f"Loaded GMSH mesh with {gmsh_mesh.n_points} points and {gmsh_mesh.n_faces} faces.")

    # Load the smooth mesh.
    smooth_mesh = pv.read(smooth_filename)
    print(f"Loaded smooth mesh with {smooth_mesh.n_points} points.")

    # Project the nodes of the GMSH mesh onto the smooth surface.
    projected_mesh = project_gmsh_to_smooth(gmsh_mesh, smooth_mesh, max_distance=2.0)
    
    # Optionally, visualize the projected mesh.
    projected_mesh.plot(show_edges=True, line_width=1, title="Projected GMSH Mesh")
    
    # Convert the resulting PyVista mesh back to a GMSH file.
    pyvista_to_gmsh(projected_mesh, gmsh_output_filename)
