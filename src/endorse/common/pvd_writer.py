from typing import *
from pathlib import Path

import xml.etree.ElementTree as ET
from xml.dom import minidom

import pyvista as pv


class PVDWriter:
    """
    Minimal ParaView time-series writer.

    Usage:
        with PVDWriter("transport_statistics.pvd") as pvd:
            pvd.write(mesh0, 0.0)
            pvd.write(mesh1, 1.0)
            ...

    This creates:
      - transport_statistics/transport_statistics_0000.vt[p/u/s/r/i/m]
      - transport_statistics/transport_statistics_0001.vt[p/u/s/r/i/m]
      - transport_statistics.pvd   (references files above with timestep values)
    """

    _EXT_MAP = {
        pv.PolyData: ".vtp",
        pv.UnstructuredGrid: ".vtu",
        pv.StructuredGrid: ".vts",
        pv.RectilinearGrid: ".vtr",
        pv.ImageData: ".vti",
        pv.MultiBlock: ".vtm",
    }

    def __init__(self, fout: Union[str, Path]):
        self.pvd_path = Path(fout)
        self.series_dir = self.pvd_path.with_suffix("")  # subdir next to the .pvd
        self.series_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[Tuple[str, float]] = []
        self._counter = 0
        self._stem = self.pvd_path.stem

    def __enter__(self) -> "PVDWriter":
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always try to write the PVD with entries collected so far.
        try:
            self._write_pvd(self._entries, self.pvd_path)
        except Exception:
            # Never mask the original exception; ignore PVD write failure here.
            pass
        # Re-raise any exception from the context block.
        return False

    def write(self, mesh: pv.DataSet, t_val: float) -> None:
        """Save a single timestep and record it in the PVD."""
        ext = self._choose_ext(mesh)
        fname = f"{self._stem}_{self._counter:04d}{ext}"
        out_path = self.series_dir / fname

        # Let PyVista infer the correct writer from the extension
        mesh.save(out_path)  # PyVista chooses the writer based on extension. :contentReference[oaicite:2]{index=2}

        # Store relative path (from PVD location) + timestep
        rel = f"{self.series_dir.name}/{fname}"
        self._entries.append((rel, float(t_val)))
        self._counter += 1

    # ---- helpers -------------------------------------------------------------

    def _choose_ext(self, mesh: pv.DataSet) -> str:
        # Pick a sensible XML VTK extension from the mesh type; fallback to legacy .vtk
        for cls, ext in self._EXT_MAP.items():
            if isinstance(mesh, cls):
                return ext
        return ".vtk"

    @staticmethod
    def _write_pvd(files: List[Tuple[str, float]], output_name: Path) -> None:
        # Simple, readable PVD header
        root = ET.Element("VTKFile", attrib={
            "type": "Collection",
            "version": "0.1",
            "byte_order": "LittleEndian",
        })
        coll = ET.SubElement(root, "Collection")
        for rel, t in files:
            ET.SubElement(coll, "DataSet", attrib={
                "timestep": f"{float(t)}",
                "part": "0",
                "file": rel,
            })
        xml_bytes = ET.tostring(root, encoding="utf-8")
        pretty = minidom.parseString(xml_bytes).toprettyxml(indent="    ")
        output_name.write_text(pretty, encoding="utf-8")