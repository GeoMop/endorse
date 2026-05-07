import numpy as np
import meshio
import os

# Assuming the corrected save_gmsh function is defined above
def save_gmsh(mesh_data, filename):
    nodes = mesh_data.vertices
    faces = mesh_data.faces

    # Ensure that nodes is of shape (n_nodes, 3)
    assert nodes.shape[1] == 3, "Nodes array must be of shape (n_nodes, 3)."

    # Ensure that faces is of shape (n_faces, 3) for triangular elements
    assert faces.shape[1] == 3, "Faces array must be of shape (n_faces, 3) for triangular elements"

    # # Example: Assign physical group ID 1 to all faces
    # face_tags = np.ones(faces.shape[0], dtype=int)
    #
    # # Include the cell data
    # cell_data = {
    #     "gmsh:physical": [face_tags],
    #     "gmsh:geometrical": [face_tags]
    # }
    #
    # # Recreate the mesh object with cell data
    # mesh = meshio.Mesh(
    #     points=nodes,
    #     cells=faces,
    #     cell_data=cell_data
    # )

    # # Create the mesh object
    cells = [("triangle", faces)]
    mesh = meshio.Mesh(
        points=nodes,
        cells=cells
    )


    meshio.write(filename, mesh, file_format="gmsh22")

class MeshData:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

def test_save_gmsh():
    # Sample vertices (nodes)
    vertices = np.array([
        [0.0, 0.0, 0.0],  # Node 0
        [1.0, 0.0, 0.0],  # Node 1
        [0.0, 1.0, 0.0],  # Node 2
        [0.0, 0.0, 1.0]   # Node 3
    ])

    # Sample faces (triangles)
    faces = np.array([
        [0, 1, 2],  # Face 0
        [0, 1, 3]   # Face 1
    ])

    # Create mesh_data object
    mesh_data = MeshData(vertices, faces)

    # Output filename
    test_filename = "test_mesh_output.msh"

    # Call the function
    save_gmsh(mesh_data, test_filename)

    # Verify the file was created
    assert os.path.isfile(test_filename), f"File {test_filename} was not created."

    # Read the mesh back in
    mesh = meshio.read(test_filename)

    # Verify the points match
    np.testing.assert_array_almost_equal(mesh.points, vertices, decimal=6)

    # Verify the cells match
    found_triangle_cells = False
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            found_triangle_cells = True
            np.testing.assert_array_equal(cell_block.data, faces)
            break
    assert found_triangle_cells, "No triangle cells found in the mesh."

    print("Test passed successfully.")

    # Clean up
    os.remove(test_filename)

# Run the test
if __name__ == "__main__":
    test_save_gmsh()
