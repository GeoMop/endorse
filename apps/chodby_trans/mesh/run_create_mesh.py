import shutil
from pathlib import Path
import logging

from endorse import common
from chodby_trans import job
from chodby_trans.mesh.create_mesh import main

script_path = Path(__file__).absolute()

if __name__ == '__main__':

    app_dir = script_path.parents[1]
    work_dir = app_dir / "workdir_msh"
    job.set_workdir(work_dir)
    shutil.copytree(app_dir / job.input.dir_path.name, job.input.dir_path, dirs_exist_ok=True)
    logging.info(job.to_str())

    if not job.input.dir_path.exists():
        raise Exception(f"Input data '{job.input.dir_path}' not found in workdir '{work_dir}'")
    cfg_path = job.input.transport_cfg_path
    cfg = common.config.load_config(str(cfg_path))

    main(cfg, work_dir, dfn_seed=1, mesh_seed=1)