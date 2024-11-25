"""
Procesing of a raw laserscan data to multiresolution surface mesh and subsequend 3d surrounding mesh.
"""
import logging

import numpy as np
import pymeshlab
import pyvista as pv
import meshio
from joblib import Memory
from pathlib import Path
import attrs
import cfg
import logging


# Set up joblib cache directory
memory = Memory(location='cache_dir', verbose=1)




@attrs.define
class MeshData:
    """
    A wrapper class to store mesh data and reconstruct MeshSet when needed.
    """
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray = None  # Optional normals
    meshset: pymeshlab.MeshSet = attrs.field(default=None, repr=False)  # Cached MeshSet object

    def to_meshset(self, force_copy=False):
        """
        Reconstructs a MeshSet object from the stored data.
        """
        if self.meshset is None or force_copy:
            ms = pymeshlab.MeshSet()
            if self.normals is None:
                mesh = pymeshlab.Mesh(vertex_matrix=self.vertices,
                                  face_matrix=self.faces)
            else:
                mesh = pymeshlab.Mesh(vertex_matrix=self.vertices,
                                  face_matrix=self.faces,
                                  v_normals_matrix=self.normals)
            ms.add_mesh(mesh, "mesh_from_data")
            if force_copy:
                return ms
            self.meshset = ms
        return self.meshset

    def __getstate__(self):
        """
        Exclude 'meshset' from pickling.
        """
        state = self.__dict__.copy()
        # Remove 'meshset' from the state
        if 'meshset' in state:
            del state['meshset']
        return state

    def __setstate__(self, state):
        """
        Restore the object's state and reinitialize 'meshset' as None.
        """
        self.__dict__.update(state)
        # Reinitialize 'meshset' after unpickling
        self.meshset = None

    def save_to_vtk(self, filename:Path):
        """
        Save the mesh to a VTK binary file using PyVista.
        """
        vertices = self.vertices
        faces = self.faces

        # PyVista expects faces to start with the number of points per face
        faces_pyvista = np.hstack([
            np.full((faces.shape[0], 1), faces.shape[1], dtype=np.int64),
            faces
        ]).flatten()

        # Create a PyVista mesh
        mesh = pv.PolyData(vertices, faces_pyvista)

        # Save the mesh to a VTK file in binary format
        mesh.save(str(filename), binary=True)

    def save_gmsh(self, filename: Path):
        nodes = self.vertices
        faces = self.faces

        # Ensure that nodes is of shape (n_nodes, 3)
        assert nodes.shape[1] == 3, "Nodes array must be of shape (n_nodes, 3)."

        # Ensure that faces is of shape (n_faces, 3) for triangular elements
        assert faces.shape[1] == 3, "Faces array must be of shape (n_faces, 3) for triangular elements"

        # Create the mesh object
        cells = [("triangle", faces)]
        mesh = meshio.Mesh(
            points=nodes,
            cells=cells
        )

        # Write the mesh to Gmsh format
        meshio.write(str(filename), mesh, file_format="gmsh22")

    def apply_filter(self, filter_name, **kwargs):
        """
        Apply a filter to the mesh using pymeshlab.
        """
        ms = self.to_meshset(force_copy=True)
        ms.apply_filter(filter_name, **kwargs)

        # Update mesh_data with the simplified mesh
        simplified_mesh = ms.current_mesh()
        vertices = np.array(simplified_mesh.vertex_matrix())
        faces = np.array(simplified_mesh.face_matrix())
        normals = np.array(simplified_mesh.vertex_normal_matrix())
        return MeshData(vertices, faces, normals, meshset=ms)


@memory.cache
def load_mesh(filename:Path):
    """
    Load a mesh from an OBJ file using pymeshlab and cache the data.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(filename))
    mesh = ms.current_mesh()
    vertices = np.array(mesh.vertex_matrix())
    faces = np.array(mesh.face_matrix())
    normals = np.array(mesh.vertex_normal_matrix())
    logging.info(f"Loaded mesh from '{filename}'.")
    return MeshData(vertices, faces, normals, meshset=ms)

def load_vtk_mesh(filename: Path) -> MeshData:
    """
    Load a VTK file into a MeshData object.

    Parameters:
    - filename (Path): Path to the VTK file.

    Returns:
    - MeshData: An instance of MeshData containing the mesh information.
    """
    # Read the VTK file using PyVista
    mesh = pv.read(str(filename))

    # Extract vertices (points)
    vertices = mesh.points  # numpy array of shape (n_points, 3)

    # Check if the mesh has faces
    if hasattr(mesh, 'faces'):
        faces = mesh.faces
    else:
        faces = mesh.cells
    assert faces.size > 0, "Mesh has no faces."

    # PyVista stores faces in a flat array where the first number of each face
    # indicates the number of points in the face.
    # For example: [3, v0, v1, v2, 3, v3, v4, v5, ...]
    face_array = faces.reshape(-1, 4)
    assert np.all(face_array[:, 0] == 3), "Only triangular faces are supported."
    face_array= face_array[:, 1:]  # Remove the first column
    #assert face_array[1] == 3, "Only triangular faces are supported."

    # Extract normals if available
    # if "Normals" in mesh.point_data:
    #     normals = mesh.point_data["Normals"]
    # else:
    #     normals = None

    # Create MeshData object
    mesh_data = MeshData(vertices=vertices, faces=face_array)
    mesh_data.to_meshset()
    return mesh_data


# Step 2: Transform Points to Local Coordinates
def transform_to_local_coordinates(mesh_data, origin_global):
    """
    Transform vertices to local coordinates based on a global origin.
    """
    # Subtract the origin from all vertices
    origin_global = np.array(origin_global)
    vertices_local = mesh_data.vertices - origin_global
    logging.info(f"Transformed mesh to local coordinates based on origin: {origin_global}.")
    return MeshData(vertices=vertices_local, faces=mesh_data.faces, normals=mesh_data.normals)

# Step 3: Estimate Resolution and Extent
# def estimate_resolution_and_extent(mesh_data):
#     """
#     Estimate the resolution and extent of a mesh.
#     """
#     # Reconstruct MeshSet to use pymeshlab filters
#     ms = mesh_data.to_meshset()
#     ms.print_filter_list()
#     # Compute edge lengths using pymeshlab
#     ms.apply_filter('compute_geometric_measures')
#     edge_lengths = ms.current_mesh().edge_length_array()
#     resolution = np.mean(edge_lengths)
#
#     # Compute extent
#     bbox_min, bbox_max = ms.current_mesh().bounding_box()
#     extent = bbox_max - bbox_min
#     print(f"Estimated resolution (average edge length): {resolution:.4f} units.")
#     print(f"Extent of the dataset (X, Y, Z): {extent} units.")
#     return resolution, extent

def estimate_resolution_and_extent(mesh_data):
    """
    Estimate the resolution and extent of a mesh.
    """
    #ms = mesh_data.to_meshset()
    #pymeshlab.pmeshlab.print_filter_list()

    vertices = mesh_data.vertices
    faces = mesh_data.faces

    # Step 1: Generate all edges from faces without Python loops
    # Edges are defined between vertices in faces: (v0, v1), (v1, v2), (v2, v0)
    edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=0)  # Shape: (n_faces * 3, 2)

    # Step 2: Sort the edge vertex indices to ensure undirected edges
    edges = np.sort(edges, axis=1)  # Shape: (n_edges, 2)

    # Step 3: Remove duplicate edges
    edges = np.unique(edges, axis=0)  # Shape: (n_unique_edges, 2)

    # Step 4: Compute edge vectors and lengths using vectorized operations
    edge_vectors = vertices[edges[:, 0]] - vertices[edges[:, 1]]  # Shape: (n_unique_edges, 3)
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)           # Shape: (n_unique_edges,)

    # Step 5: Compute the average edge length (resolution)
    resolution = np.mean(edge_lengths)
    # Compute extent
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extent = bbox_max - bbox_min

    print(f"Estimated resolution (average edge length): {resolution:.4f} units.")
    print(f"Extent of the dataset (X, Y, Z): {extent}")
    print(f"BB: {bbox_min} , {bbox_max}")
    return resolution, extent

#
# # Step 4: Display Mesh with PyVista
# def display_mesh_with_pyvista(ms):
#     """
#     Visualize the mesh interactively using PyVista.
#     """
#     # Extract vertices and faces from pymeshlab
#     vertices = np.array(ms.current_mesh().vertex_matrix())
#     faces = np.array(ms.current_mesh().face_matrix())
#
#     # PyVista expects face array to start with the number of vertices per face
#     faces_pyvista = np.hstack(
#         [np.full((faces.shape[0], 1), faces.shape[1]), faces]
#     ).flatten()
#
#     # Create a PyVista mesh
#     mesh = pv.PolyData(vertices, faces_pyvista)
#
#     # Initialize the plotter
#     plotter = pv.Plotter()
#     plotter.add_mesh(mesh, color='lightgray', show_edges=True)
#     plotter.add_axes()
#     plotter.show()

def simplify_mesh(mesh_data, target_face_num):
    """
    Simplify the mesh using face quality to preserve important features.
    """
    return mesh_data.apply_filter(
        'meshing_decimation_quadric_edge_collapse',
        targetfacenum=target_face_num,
        preservetopology=True,
        #preserveboundary=True,
        #boundaryweight=1.0,
        #preservenormal=True,
        #planarquadric=True,
        #qualitythr=0.5,  # Prevent collapsing faces with quality above 0.5
        #qualityweight=True,  # Use face quality in simplification
        autoclean=True
    )

def remesh(mesh_data, target_edge_length):
    """
    Remesh the mesh using face quality to preserve important features.
    """

    return mesh_data.apply_filter(
        'meshing_isotropic_explicit_remeshing',
        adaptive=True,
        targetlen=pymeshlab.PureValue(target_edge_length),  # Set desired edge length
        iterations=3  # Number of iterations
    )

# Step 4: Display Mesh with PyVista
def display_mesh_with_pyvista(mesh_data):
    """
    Visualize the mesh interactively using PyVista.
    """
    vertices = mesh_data.vertices
    faces = mesh_data.faces

    # PyVista expects faces to start with the number of points per face
    faces_pyvista = np.hstack(
        [np.full((faces.shape[0], 1), faces.shape[1]), faces]
    ).flatten()

    # Create a PyVista mesh
    mesh = pv.PolyData(vertices, faces_pyvista)

    # Initialize the plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightgray', show_edges=True)
    plotter.add_axes()
    plotter.show()



# Main Function
def main():
    # Load and cache the mesh
    ms = load_mesh(cfg.input_obj_file)

    # Transform to local coordinates
    ms = transform_to_local_coordinates(ms, cfg.origin_global_coords)

    # Estimate resolution and extent
    resolution, extent = estimate_resolution_and_extent(ms)
    ms = ms.apply_filter('meshing_remove_duplicate_vertices')
    ms = ms.apply_filter('meshing_remove_duplicate_faces')

    ms.save_to_vtk(cfg.workdir / "L5_local.vtk")
    ms.save_gmsh(cfg.workdir / "L5_local.msh")

    # Simplify the mesh with face quality
    for i in range(1, 8):
        print(f"Compute Level {i}.")
        #target_face_num = len(ms.faces) // (4 ** i)   # half in each surface direction
        #reduced_ms = simplify_mesh_with_face_quality(ms, target_face_num)
        target_edge_length = resolution * 2 ** i
        #reduced_ms = remesh(ms, target_edge_length)

        reduced_ms = ms.apply_filter('generate_surface_reconstruction_vcg',
                             voxsize=pymeshlab.PureValue(target_edge_length),
                             geodesic=2,
                             smoothnum=1,  # Laplace iteration to smooth borders
                             widenum=10,  # How many voxels is size of holes to fill.
                             simplification=True,
                             normalsmooth=3
                             )

        print(f"Save Level {i}: ", f"L5_local_{i}.vtk")
        reduced_ms.save_to_vtk(cfg.workdir / f"L5_local_{i}.vtk")
        reduced_ms.save_gmsh(cfg.workdir / f"L5_local_{i}.msh")

    # Display the mesh
    #display_mesh_with_pyvista(ms)


if __name__ == "__main__":
    main()
