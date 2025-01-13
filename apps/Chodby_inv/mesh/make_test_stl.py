import numpy as np

# Define vertices of a simple triangular prism for STL
vertices = np.array([
    [0.0, 0.0, 0.0],  # Vertex 0
    [1.0, 0.0, 0.0],  # Vertex 1
    [0.5, 0.866, 0.0],  # Vertex 2
    [0.0, 0.0, 1.0],  # Vertex 3 (top of prism)
    [1.0, 0.0, 1.0],  # Vertex 4
    [0.5, 0.866, 1.0]  # Vertex 5
])

# Define triangular faces of the prism using vertex indices
# Each face is defined as a triplet of vertex indices
faces = np.array([
    [0, 1, 2],  # Bottom triangle
    [3, 4, 5],  # Top triangle
    [0, 1, 4],  # Side 1
    [0, 4, 3],
    [1, 2, 5],  # Side 2
    [1, 5, 4],
    [2, 0, 3],  # Side 3
    [2, 3, 5]
])

# Write to an STL file
stl_filename = "simple_prism.stl"
with open(stl_filename, 'w') as f:
    f.write("solid prism\n")
    for face in faces:
        # Calculate normal vector for the face
        v1, v2, v3 = vertices[face]
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else normal
        f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
        f.write("    outer loop\n")
        for vertex_idx in face:
            vertex = vertices[vertex_idx]
            f.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")
    f.write("endsolid prism\n")

stl_filename
