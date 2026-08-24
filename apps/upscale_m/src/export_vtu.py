"""
Flow123d mechanics .msh output -> ParaView .vtu, keeping displacement and stress
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

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


def export_case(msh_path: str, out_path: str = None, drop_region_id=None,
                suffix: str = "") -> Path:
    """
    One solved case -> .vtu, by default without the SUBC support tetrahedra
    """
    mesh = _load_mesh(File(str(msh_path)), None)
    out = Path(out_path) if out_path else Path(msh_path).with_suffix(".vtu")

    keep = np.ones(len(mesh.elements), dtype=bool)

    if drop_region_id is not None and str(drop_region_id) not in ("", "None"):
        drop_id = int(drop_region_id)
        keep = np.array([el.tags[0] != drop_id for el in mesh.elements])

    cells = [np.concatenate(([len(el.node_indices)],
                             np.asarray(el.node_indices, dtype=np.int64)))
             for el in mesh.elements]
    grid = pv.UnstructuredGrid(np.concatenate(cells), mesh._pv_celltypes(), mesh.nodes)

    cell_fields: Dict[str, np.ndarray] = {}
    for name in ("stress", "region_id"):
        if name in mesh.gmsh_io.element_data:
            # Mesh.get_p0_values maps $ElementData on the ordering of the elements
            cell_fields[name] = np.asarray(mesh.get_p0_values(name, time=1e30), dtype=float)
            assert len(cell_fields[name]) == len(mesh.elements), \
                f"pole {name}: {len(cell_fields[name])} hodnot, sit ma {len(mesh.elements)} elementu"
            grid.cell_data[name + suffix] = cell_fields[name]

    if "displacement" in mesh.gmsh_io.element_node_data:
        grid.point_data["displacement" + suffix] = element_node_field(mesh, "displacement")

    if "stress" in cell_fields:
        stress_pt = np.zeros((len(mesh.nodes), cell_fields["stress"].shape[1]))
        for iel, el in enumerate(mesh.elements):
            stress_pt[el.node_indices] = cell_fields["stress"][iel]
        grid.point_data["stress_pt" + suffix] = stress_pt

    if not keep.all():
        grid = grid.extract_cells(np.flatnonzero(keep))
        for junk in ("vtkOriginalCellIds", "vtkOriginalPointIds"):
            grid.cell_data.pop(junk, None)
            grid.point_data.pop(junk, None)

    grid.save(out)
    return out


def export_case_isolated(msh_path: str, out_path: str, drop_region_id=None,
                         suffix: str = "") -> Path:
    cmd = [sys.executable, str(Path(__file__).resolve()), str(msh_path), str(out_path),
           "None" if drop_region_id is None else str(int(drop_region_id)), suffix]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        return None
    for line in (done.stdout or "").splitlines():
        if line.startswith(_TAG):
            print(line, flush=True)
    return Path(out_path)


def export_dns_case(cfg, micro, spatial_file: str, case_dir: str) -> Optional[Path]:
    """
    One solved DNS case -> case_dir/dns_full.vtu, or None if cfg.loads.export_vtu is false
    """
    if not bool(cfg.loads.get("export_vtu", True)):
        return None
    support_id = micro.regions.get(str(cfg.geometry.support_region), (None,))[0]
    return export_case_isolated(spatial_file, os.path.join(case_dir, "dns_full.vtu"), support_id)


def export_macro_case(cfg, macro_mesh, spatial_file: str, case_dir: str) -> Optional[Path]:
    """
    One solved macro case -> case_dir/macro_fields.vtu, or None if cfg.loads.export_vtu is false
    """
    if not bool(cfg.loads.get("export_vtu", True)):
        return None
    support_id = macro_mesh.gmsh_io.physical.get(str(cfg.geometry.support_region), (None,))[0]
    return export_case_isolated(spatial_file, os.path.join(case_dir, "macro_fields.vtu"),
                                support_id, "_macro")


if __name__ == "__main__":
    export_case(*sys.argv[1:])
