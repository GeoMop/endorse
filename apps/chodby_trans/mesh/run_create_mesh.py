import shutil
from pathlib import Path
import logging

from endorse import common
from endorse.mesh import fracture_tools
from chodby_trans import job
from chodby_trans.mesh.create_mesh import make_fractures, make_mesh

script_path = Path(__file__).absolute()

# def update_mesh_cfg(cfg, level):
#     mesh_scale = level.mesh_step_scale
#
#     mcfg = common.dotdict.create(cfg)
#
#     mcfg.mesh.boundary_mesh_step = mesh_scale * mcfg.mesh.boundary_mesh_step
#     mcfg.mesh.fracture_mesh_step = mesh_scale * mcfg.mesh.fracture_mesh_step
#     mcfg.mesh.main_tunnel_mesh_step = mesh_scale * mcfg.mesh.main_tunnel_mesh_step
#     mcfg.mesh.boreholes_mesh_step = mesh_scale * mcfg.mesh.boreholes_mesh_step
#
#     mcfg.mesh.mesh_name = mcfg.mesh.mesh_name + f"_L{level.id}"
#     return mcfg


def update_mesh_cfg(cfg_mesh, level_dict):

    mcfg = common.apply_variant(cfg_mesh, level_dict.params)
    # assert mcfg == cfg_mesh
    mcfg.mesh_name = mcfg.mesh_name + f"_L{level_dict.id}"
    return mcfg


def main(cfg, workdir, dfn_seed, mesh_seed):
    with common.workdir(workdir, clean=False):
        fr_pop, fracture_set, n_large = make_fractures(cfg.mesh, dfn_seed)
        fr_stats = fracture_tools.fracture_set_stats(fracture_set)
        print(f"N fracture set: {len(fracture_set)}")
        print(f"Fracture radius min: {fr_stats['min_radius']}, max: {fr_stats['max_radius']}")

        # L0 fine
        level = cfg.mlmc.levels[0]
        cfg_mesh = update_mesh_cfg(cfg.mesh, level)
        cfg_mesh.mesh_name += "_fine"
        make_mesh(cfg_mesh, fracture_set, mesh_seed)

        # L0 fine with buffer
        level = cfg.mlmc.levels[0]
        cfg_mesh = update_mesh_cfg(cfg.mesh, level)
        cfg_mesh.geometry.box_dimensions = [v + 2*level.buffer_width for v in cfg_mesh.geometry.box_dimensions]
        cfg_mesh.geometry.main_tunnel.length += 2*level.buffer_width
        cfg_mesh.mesh_name += "_fine_buffer"
        make_mesh(cfg_mesh, fracture_set, mesh_seed)

        # L0 coarse
        level = cfg.mlmc.levels[1]
        cfg_mesh = update_mesh_cfg(cfg.mesh, level)
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
