import gmsh
import os
import numpy as np

def shift(source_file, shift_vec, out_file):
    gmsh.initialize()
    source_file = os.path.abspath(source_file)
    if os.path.isfile(source_file):
        # gmsh API opens a new file for a a missing one
        gmsh.open(source_file)
    else:
        raise FileNotFoundError(f"{source_file}")
    transform_mat = np.array([1.0, 0, 0, shift_vec[0],
                              0, 1.0, 0, shift_vec[1],
                              0, 0, 1.0, shift_vec[2],
                              0, 0, 0,   1.0])
    gmsh.model.mesh.affineTransform(transform_mat)
    gmsh.write(out_file)

