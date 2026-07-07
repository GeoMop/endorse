"""Prepare and run the local hydro-mechanical Flow123d model."""

import argparse
import shutil
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import input_data
from endorse import common


module_dir = Path(__file__).resolve().parent
work_dir = input_data.work_dir


DEFAULT_REPLACEMENTS = {
    "rock_conductivity": "1e-13",
    "rock_storativity": "1",
    "packer_conductivity": "1e-13",
    "packer_storativity": "1",
    "water_conductivity": "1e-6",
    "watyer_storativity": "1",
    "fracture_conductivity": "1e-6",
    "fracture_storativity": "1",
    "fracture_cross_section": "1e-3",
    "rock_young": "60e9",
    "rock_poisson": "0.25",
    "fracture_young": "1e7",
    "fracture_poisson": "0.25",
}


def machine_config(config_path: Path | None, flow_executable: str) -> common.dotdict:
    """Return Flow123d machine configuration."""
    if config_path is not None and config_path.exists():
        return common.load_config(config_path).machine_config
    return common.dotdict({"flow_executable": [flow_executable]})


def prepare_mesh_file() -> None:
    """Make the mesh filename expected by the YAML template available."""
    expected_mesh = work_dir / "wpt_section.msh"
    generated_mesh = work_dir / "wpt_section.msh2"
    if not expected_mesh.exists() and generated_mesh.exists():
        shutil.copy2(generated_mesh, expected_mesh)


def run_model(cfg: common.dotdict, replacements: dict[str, str] | None = None) -> common.FlowOutput:
    """Substitute YAML template placeholders and run Flow123d."""
    yaml_replacements = DEFAULT_REPLACEMENTS.copy()
    if replacements is not None:
        yaml_replacements.update(replacements)

    prepare_mesh_file()
    with common.workdir(work_dir):
        return common.call_flow(cfg, input_data.hm_sim_tmpl_yaml, yaml_replacements)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=module_dir / "config.yaml",
        help="Optional config file with machine_config.",
    )
    parser.add_argument(
        "--flow-executable",
        default="flow123d",
        help="Flow123d executable used when --config is not present.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = machine_config(args.config, args.flow_executable)
    run_model(cfg)
