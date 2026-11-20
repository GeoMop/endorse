import numpy as np
import pytest

from mesh_verify import tetra_face_index, face_keys


def test_face_incidence_and_exterior_across_chunk_boundaries():
    tetra = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.uint64)
    with tetra_face_index(tetra, chunk_size=1) as index:
        query = np.array([[3, 2, 1], [0, 1, 3], [0, 3, 4]], dtype=np.uint64)
        np.testing.assert_array_equal(index.incidence(query), [2, 1, 0])
        exterior = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3],
                             [1, 2, 4], [1, 3, 4], [2, 3, 4]], dtype=np.uint64)
        np.testing.assert_array_equal(index.boundary, np.sort(face_keys(exterior)))
        filename = index.keys.filename
    from pathlib import Path
    assert not Path(filename).exists()


def test_nonmanifold_faces_fail_before_region_verification():
    tetra = np.array([[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]], dtype=np.uint64)
    with pytest.raises(ValueError, match='Nonmanifold'):
        with tetra_face_index(tetra, chunk_size=2):
            pass
