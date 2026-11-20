from types import SimpleNamespace

import numpy as np

from mesh_output import write_msh22
from bgem.gmsh.gmsh_io import GmshIO


def test_direct_writer_preserves_sparse_ids_tags_names_and_float_precision(tmp_path):
    mesh = SimpleNamespace(
        nodes={10: [1e-17, 0., 0.], 21: [1., 0., 0.], 35: [0., 1., 0.], 77: [0., 0., 1.]},
        elements={99: (4, [1, 17], [10, 21, 35, 77]),
                  101: (2, [4, 8], [10, 21, 35]),
                  107: (1, [7, 12], [10, 21])},
        physical={'bulk rock': (1, 3), 'EW': (4, 2), 'trace': (7, 1)},
    )
    path = tmp_path/'direct.msh2'
    write_msh22(mesh, path)
    assert path.read_text().splitlines()[:2] == ['$MeshFormat', '2.2 0 8']
    restored = GmshIO(str(path))
    assert restored.physical == mesh.physical
    assert restored.elements == mesh.elements
    for node, xyz in mesh.nodes.items():
        np.testing.assert_array_equal(restored.nodes[node], xyz)
