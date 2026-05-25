# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/jb/workspace/endorse-pointcloud/src/endorse/process_point_cloud.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 2024-11-24 14:02:16 UTC (1732456936)

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
memory = Memory(location='cache_dir', verbose=1)

@attrs.define
class MeshData:
    """
    A wrapper class to store mesh data and reconstruct MeshSet when needed.
    """
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray = None
    meshset: pymeshlab.MeshSet = attrs.field(default=None, repr=False)

    def to_meshset(self, force_copy=False):
        """
        Reconstructs a MeshSet object from the stored data.
        """
        if self.meshset is None or force_copy:
            ms = pymeshlab.MeshSet()
            if self.normals is None:
                mesh = pymeshlab.Mesh(vertex_matrix=self.vertices, face_matrix=self.faces)
            else:
                mesh = pymeshlab.Mesh(vertex_matrix=self.vertices, face_matrix=self.faces, v_normals_matrix=self.normals)
            ms.add_mesh(mesh, 'mesh_from_data')
            if force_copy:
                return ms
            self.meshset = ms
        return self.meshset

    def __getstate__(self):
        """
        Exclude 'meshset' from pickling.
        """
        state = self.__dict__.copy()
        if 'meshset' in state:
            del state['meshset']
        return state

    def __setstate__(self, state):
        """
        Restore the object's state and reinitialize 'meshset' as None.
        """
        self.__dict__.update(state)
        self.meshset = None

    def save_to_vtk(self, filename: Path):
        """
        Save the mesh to a VTK binary file using PyVista.
        """
        vertices = self.vertices
        faces = self.faces
        faces_pyvista = np.hstack([np.full((faces.shape[0], 1), faces.shape[1], dtype=np.int64), faces]).flatten()
        mesh = pv.PolyData(vertices, faces_pyvista)
        mesh.save(str(filename), binary=True)

    def save_gmsh(self, filename: Path):
        nodes = self.vertices
        faces = self.faces
        assert nodes.shape[1] == 3, 'Nodes array must be of shape (n_nodes, 3).'
        assert faces.shape[1] == 3, 'Faces array must be of shape (n_faces, 3) for triangular elements'
        cells = [('triangle', faces)]
        mesh = meshio.Mesh(points=nodes, cells=cells)
        meshio.write(str(filename), mesh, file_format='gmsh22')

    def apply_filter(self, filter_name, **kwargs):
        """
        Apply a filter to the mesh using pymeshlab.
        """
        ms = self.to_meshset(force_copy=True)
        ms.apply_filter(filter_name, **kwargs)
        simplified_mesh = ms.current_mesh()
        vertices = np.array(simplified_mesh.vertex_matrix())
        faces = np.array(simplified_mesh.face_matrix())
        normals = np.array(simplified_mesh.vertex_normal_matrix())
        return MeshData(vertices, faces, normals, meshset=ms)

@memory.cache
def load_mesh(filename: Path):
    """
    Load a mesh from an OBJ file using pymeshlab and cache the data.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(filename))
    mesh = ms.current_mesh()
    vertices = np.array(mesh.vertex_matrix())
    faces = np.array(mesh.face_matrix())
    normals = np.array(mesh.vertex_normal_matrix())
    logging.info(f"Loaded mesh from '{filename}0'.")
    return MeshData(vertices, faces, normals, meshset=ms)

def load_vtk_mesh(filename: Path) -> MeshData:
    """
    Load a VTK file into a MeshData object.

    Parameters:
    - filename (Path): Path to the VTK file.

    Returns:
    - MeshData: An instance of MeshData containing the mesh information.
    """
    mesh = pv.read(str(filename))
    vertices = mesh.points
    if hasattr(mesh, 'faces'):
        faces = mesh.faces
    else:
        faces = mesh.cells
    assert faces.size > 0, 'Mesh has no faces.'
    face_array = faces.reshape(-1, 4)
    assert np.all(face_array[:, 0] == 3), 'Only triangular faces are supported.'
    face_array = face_array[:, 1:]
    mesh_data = MeshData(vertices=vertices, faces=face_array)
    mesh_data.to_meshset()
    return mesh_data

def transform_to_local_coordinates(mesh_data, origin_global):
    """
    Transform vertices to local coordinates based on a global origin.
    """
    origin_global = np.array(origin_global)
    vertices_local = mesh_data.vertices - origin_global
    logging.info(f'Transformed mesh to local coordinates based on origin: {origin_global}0.')
    return MeshData(vertices=vertices_local, faces=mesh_data.faces, normals=mesh_data.normals)

def estimate_resolution_and_extent(mesh_data):
    """
    Estimate the resolution and extent of a mesh.
    """
    vertices = mesh_data.vertices
    faces = mesh_data.faces
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    edge_vectors = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    resolution = np.mean(edge_lengths)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extent = bbox_max - bbox_min
    print(f'Estimated resolution (average edge length): {resolution:.4f} units.')
    print(f'Extent of the dataset (X, Y, Z): {extent}0')
    print(f'BB: {bbox_min} , {bbox_max}')
    return (resolution, extent)

def simplify_mesh(mesh_data, target_face_num):
    """
    Simplify the mesh using face quality to preserve important features.
    """
    return mesh_data.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=target_face_num, preservetopology=True, autoclean=True)

def remesh(mesh_data, target_edge_length):
    """
    Remesh the mesh using face quality to preserve important features.
    """
    return mesh_data.apply_filter('meshing_isotropic_explicit_remeshing', adaptive=True, targetlen=pymeshlab.PureValue(target_edge_length), iterations=3)

def display_mesh_with_pyvista(mesh_data):
    """
    Visualize the mesh interactively using PyVista.
    """
    vertices = mesh_data.vertices
    faces = mesh_data.faces
    faces_pyvista = np.hstack([np.full((faces.shape[0], 1), faces.shape[1]), faces]).flatten()
    mesh = pv.PolyData(vertices, faces_pyvista)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightgray', show_edges=True)
    plotter.add_axes()
    plotter.show()

def main():
    ms = load_mesh(cfg.input_obj_file)
    ms = transform_to_local_coordinates(ms, cfg.origin_global_coords)
    resolution, extent = estimate_resolution_and_extent(ms)
    ms.save_to_vtk(cfg.workdir / 'L5_local.vtk')
    ms.save_gmsh(cfg.workdir / 'L5_local.msh')
    for i in range(1, 7):
        print(f'Compute Level {i}.')
        target_edge_length = resolution * 2 ** i
        reduced_ms = remesh(ms, target_edge_length)
        print(f'Save Level {i}: ', f'L5_local_{i}0.vtk')
        reduced_ms.save_to_vtk(cfg.workdir / f'L5_local_{i}0.vtk')
        reduced_ms.save_gmsh(cfg.workdir / f'L5_local_{i}0.msh')

if __name__ == '__main__':
    main()
