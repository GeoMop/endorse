"""
Flow123d mechanics .msh output -> ParaView .vtu, keeping displacement and stress
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv

from endorse.common import File
from endorse.mesh_class import _load_mesh

_TAG = "[export_vtu]"


def element_node_field(mesh, name: str, n_comp: int = 3) -> np.ndarray:
    data = mesh.gmsh_io.element_node_data[name]
    item = data[sorted(data.keys())[-1]]
    out = np.zeros((len(mesh.nodes), n_comp))
    for eid, row in zip(item.tags, item.values):
        node_idx = np.asarray(mesh.elements[mesh.el_indices[eid]].node_indices, dtype=int)
        by_id = node_idx[np.argsort([mesh.node_ids[n] for n in node_idx])]
        out[by_id] = np.asarray(row, dtype=float).reshape(len(node_idx), n_comp)
    return out


def export_case(msh_path: str, out_path: str = None) -> Path:
    """
    One solved case -> .vtu, a plain format conversion of the Flow123d output: the whole mesh
    (SUBC support tetrahedra included), stress and region_id as cell data, displacement as point
    data. Any derived field (point stress, strain, ...) is left to the postprocessing scripts.
    """
    mesh = _load_mesh(File(str(msh_path)), None)
    out = Path(out_path) if out_path else Path(msh_path).with_suffix(".vtu")

    cells = [np.concatenate(([len(el.node_indices)],
                             np.asarray(el.node_indices, dtype=np.int64)))
             for el in mesh.elements]
    grid = pv.UnstructuredGrid(np.concatenate(cells), mesh._pv_celltypes(), mesh.nodes)

    for name in ("stress", "region_id"):
        if name in mesh.gmsh_io.element_data:
            # Mesh.get_p0_values maps $ElementData on the ordering of the elements
            values = np.asarray(mesh.get_p0_values(name, time=1e30), dtype=float)
            assert len(values) == len(mesh.elements), \
                f"pole {name}: {len(values)} hodnot, sit ma {len(mesh.elements)} elementu"
            grid.cell_data[name] = values

    if "displacement" in mesh.gmsh_io.element_node_data:
        grid.point_data["displacement"] = element_node_field(mesh, "displacement")

    grid.save(out)
    return out


def export_case_isolated(msh_path: str, out_path: str) -> Path:
    cmd = [sys.executable, str(Path(__file__).resolve()), str(msh_path), str(out_path)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        return None
    for line in (done.stdout or "").splitlines():
        if line.startswith(_TAG):
            print(line, flush=True)
    return Path(out_path)


def export_dns_case(cfg, spatial_file: str, case_dir: str) -> Optional[Path]:
    """
    One solved DNS case -> case_dir/dns_full.vtu, or None if cfg.loads.export_vtu is false
    """
    if not bool(cfg.loads.get("export_vtu", True)):
        return None
    return export_case_isolated(spatial_file, os.path.join(case_dir, "dns_full.vtu"))


def export_macro_case(cfg, spatial_file: str, case_dir: str) -> Optional[Path]:
    """
    One solved macro case -> case_dir/macro_fields.vtu, or None if cfg.loads.export_vtu is false
    """
    if not bool(cfg.loads.get("export_vtu", True)):
        return None
    return export_case_isolated(spatial_file, os.path.join(case_dir, "macro_fields.vtu"))


if __name__ == "__main__":
    export_case(*sys.argv[1:])
