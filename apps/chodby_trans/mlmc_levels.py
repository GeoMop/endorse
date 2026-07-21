from __future__ import annotations

from endorse import common


def coarsest_level_id(cfg) -> int:
    return 0


def finest_level_id(cfg) -> int:
    return len(cfg.mlmc.levels) - 1


def update_mesh_cfg(cfg_mesh, level_id: int, level_cfg):
    mcfg = common.apply_variant(cfg_mesh, level_cfg.params)
    mcfg.mesh_name = mcfg.mesh_name + f"_L{level_id}"
    return mcfg
