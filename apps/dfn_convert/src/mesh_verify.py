"""Disk-backed tetrahedron face incidence for large mixed-dimensional meshes."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


FACE_DTYPE = np.dtype([('a', np.uint64), ('b', np.uint64), ('c', np.uint64)])


def face_keys(faces):
    return np.ascontiguousarray(np.sort(faces, axis=1), dtype=np.uint64).view(FACE_DTYPE).ravel()


def _boundary_faces(keys, chunk_size):
    """Scan sorted runs, carrying the last run across disk-read chunks."""
    boundary = []
    previous, previous_count = None, 0
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start:start+chunk_size]
        starts = np.r_[0, np.flatnonzero(chunk[1:] != chunk[:-1])+1]
        unique = chunk[starts].copy()
        counts = np.diff(np.r_[starts, len(chunk)])
        if previous is not None:
            if previous == unique[0]:
                counts[0] += previous_count
            elif previous_count == 1:
                boundary.append(np.array([previous], dtype=FACE_DTYPE))
        if np.any(counts > 2):
            raise ValueError('Nonmanifold tetrahedron faces')
        boundary.append(unique[:-1][counts[:-1] == 1])
        previous, previous_count = unique[-1].copy(), int(counts[-1])
    if previous_count == 1:
        boundary.append(np.array([previous], dtype=FACE_DTYPE))
    return np.concatenate(boundary) if boundary else np.empty(0, dtype=FACE_DTYPE)


@dataclass(frozen=True)
class FaceIndex:
    keys: np.memmap
    boundary: np.ndarray

    def incidence(self, triangles):
        query = face_keys(triangles)
        return np.searchsorted(self.keys, query, side='right') - np.searchsorted(self.keys, query, side='left')


@contextmanager
def tetra_face_index(tetra, directory=Path('.'), chunk_size=200_000):
    """Index all four faces per tetrahedron without copying the full face table.

    Put directory on a disk filesystem: /tmp can be RAM-backed on Linux.
    Temporary storage is 96 bytes per tetrahedron and is removed on exit.
    """
    with TemporaryDirectory(prefix='.dfn-verify-', dir=directory) as temporary:
        faces = np.memmap(Path(temporary)/'faces.bin', mode='w+', dtype=np.uint64,
                          shape=(4*len(tetra), 3))
        try:
            for start in range(0, len(tetra), chunk_size):
                block = tetra[start:start+chunk_size]
                offset = 4*start
                for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                    part = np.sort(block[:, indices], axis=1)
                    faces[offset:offset+len(block)] = part
                    offset += len(block)
            keys = faces.view(FACE_DTYPE).reshape(-1)
            keys.sort()
            boundary = _boundary_faces(keys, 4*chunk_size)
            yield FaceIndex(keys, boundary)
        finally:
            faces._mmap.close()
