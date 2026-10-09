"""Check the local environment needed by the single-fracture study."""

from __future__ import annotations

import argparse
import importlib
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from endorse import common

try:
    from .setup import DEFAULT_CONFIG, ENV_CHECK_DIR, StudyConfig
except ImportError:  # pragma: no cover - allows direct script execution.
    from setup import DEFAULT_CONFIG, ENV_CHECK_DIR, StudyConfig


PACKAGE_NAMES = ["numpy", "yaml", "xarray", "zarr", "pyvista", "bgem"]


def main() -> None:
    """Print package and Flow123d availability checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-flow123d", action="store_true")
    parser.add_argument("--skip-zarr-write", action="store_true")
    args = parser.parse_args()

    for package_name in PACKAGE_NAMES:
        check_import(package_name)

    cfg = StudyConfig.from_yaml(args.config)
    if not args.skip_zarr_write:
        check_xarray_zarr_write(cfg.zarr_write_timeout_seconds)
    if not args.skip_flow123d:
        check_flow123d(cfg.flow_call_config, cfg.flow_probe_timeout_seconds)


def check_import(package_name: str) -> None:
    """Import one package and report its version."""
    module = importlib.import_module(package_name)
    version = getattr(module, "__version__", "unknown")
    print(f"OK import {package_name}: {version}")


def check_flow123d(flow_call_config: dict[str, object], timeout_seconds: int) -> None:
    """Run Flow123d through the endorse ``call_flow`` wrapper."""
    template_text = """\
flow123d_version: 3.9.0
problem: !Coupling_Sequential
  description: Environment probe for endorse.common.call_flow.
  mesh:
    mesh_file: flow_probe.msh
  flow_equation: !Flow_Darcy_LMH
    input_fields:
      - region: boundary
        bc_type: dirichlet
        bc_piezo_head: !FieldFormula
          "x + y + z"
      - region: BULK
        conductivity: 1.0
    n_schurs: 2
    output:
      fields:
        - piezo_head_p0
        - velocity_p0
        - region_id
    balance: {}
"""
    mesh_text = """\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
2 2 "boundary"
3 1 "BULK"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
$EndNodes
$Elements
5
1 2 2 2 0 1 3 2
2 2 2 2 0 1 2 4
3 2 2 2 0 1 4 3
4 2 2 2 0 2 3 4
5 4 2 1 0 1 2 3 4
$EndElements
"""

    def timeout_handler(signum: int, frame: object) -> None:
        raise TimeoutError("Flow123d probe timed out")

    flow_dir = ENV_CHECK_DIR / "flow123d"
    if flow_dir.exists():
        shutil.rmtree(flow_dir)
    flow_dir.mkdir(parents=True)
    template = flow_dir / "flow_probe_tmpl.yaml"
    mesh = flow_dir / "flow_probe.msh"
    template.write_text(template_text, encoding="utf-8")
    mesh.write_text(mesh_text, encoding="utf-8")
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        with common.workdir(flow_dir, clean=False):
            result = common.call_flow(common.dotdict.create(flow_call_config), str(template), {})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    print(f"OK flow123d call_flow probe: {flow_call_config.get('flow_executable')}")
    print(f"OK flow123d return code: {result.process.returncode}")


def check_xarray_zarr_write(timeout_seconds: int) -> None:
    """Verify that a minimal xarray-to-Zarr write completes."""
    probe_path = ENV_CHECK_DIR / "xarray_zarr_probe.zarr"
    if probe_path.exists():
        shutil.rmtree(probe_path)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "import numpy as np, xarray as xr;"
        "from pathlib import Path;"
        f"p=Path({str(probe_path)!r});"
        "ds=xr.Dataset({'a': ('x', np.arange(3.0))});"
        "ds.to_zarr(p, mode='w', consolidated=True, zarr_format=2)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=timeout_seconds)
    print(f"OK xarray.to_zarr probe: {probe_path}")


if __name__ == "__main__":
    sys.exit(main())
