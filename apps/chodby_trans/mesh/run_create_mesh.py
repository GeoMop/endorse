import shutil
from pathlib import Path
import logging

from endorse import common
from endorse.mesh import fracture_tools
from chodby_trans import job
from chodby_trans.mlmc_levels import coarsest_level_id, finest_level_id, update_mesh_cfg
from chodby_trans.mesh.create_mesh import make_fractures, make_mesh

script_path = Path(__file__).absolute()


def main(cfg, workdir, dfn_seed, mesh_seed):
    with common.workdir(workdir, clean=False):
        fr_pop, fracture_set, n_large = make_fractures(cfg.mesh, dfn_seed)
        fr_stats = fracture_tools.fracture_set_stats(fracture_set)
        print(f"N fracture set: {len(fracture_set)}")
        print(f"Fracture stats:\n"
              f"min: {fr_stats['min_radius']},\n"
              f"max: {fr_stats['max_radius']},\n"
              f"avg: {fr_stats['avg_radius']},\n"
              f"med: {fr_stats['med_radius']}")

        fine_level_id = finest_level_id(cfg)
        coarse_level_id = coarsest_level_id(cfg)

        # finest
        level = cfg.mlmc.levels[fine_level_id]
        cfg_mesh = update_mesh_cfg(cfg.mesh, fine_level_id, level)
        cfg_mesh.mesh_name += "_fine"
        make_mesh(cfg_mesh, fracture_set, mesh_seed)

        # finest with buffer
        level = cfg.mlmc.levels[fine_level_id]
        cfg_mesh = update_mesh_cfg(cfg.mesh, fine_level_id, level)
        cfg_mesh.geometry.box_dimensions = [v + 2*level.buffer_width for v in cfg_mesh.geometry.box_dimensions]
        cfg_mesh.geometry.main_tunnel.length += 2*level.buffer_width
        cfg_mesh.mesh_name += "_fine_buffer"
        make_mesh(cfg_mesh, fracture_set, mesh_seed)

        # coarsest
        level = cfg.mlmc.levels[coarse_level_id]
        cfg_mesh = update_mesh_cfg(cfg.mesh, coarse_level_id, level)
        cfg_mesh.mesh_name += "_coarse"
        coarse_fracture_set = [fr for fr in fracture_set if fr.r > level.fr_min_limit]
        print(f"N coarse fracture set: {len(coarse_fracture_set)}")
        make_mesh(cfg_mesh, coarse_fracture_set, mesh_seed)


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
