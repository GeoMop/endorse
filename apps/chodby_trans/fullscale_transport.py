import logging
import os
import shutil
from typing import *

import numpy as np

from endorse import common

from endorse.common import dotdict, File, report, memoize
from endorse.mesh_class import Mesh
# from .indicator import indicator_set, indicators, IndicatorFn
from bgem.stochastic.fracture import Fracture, Population
# from endorse import hm_simulation

from endorse.fullscale_transport import compute_fields, fracture_map, apply_fields, output_times

from apps.Chodby_trans.mesh.create_mesh import make_mesh

script_dir = os.path.dirname(os.path.realpath(__file__))

# @attrs.define
# class ResultSpec:
#     name: str
#     # Quantile name
#     quantile_exp: float
#     # Quantile parameter, quantile_prob = 1 - quantile_param; 0 = maximum
#     times: List[float]
#     # Output times (years)
#     unit: str
#     # Unit of quantity


# def transport_result_format(cfg:dotdict) -> List[ResultSpec]:
#     q_times = quantity_times(output_times(cfg.transport_fullscale))
#     unit = "g/m3"
#     results = [ResultSpec(ind.indicator_label_short, ind.q_exp, q_times, unit) for ind in indicator_set()]
#     return results


def fullscale_transport(cfg_path, seed):
    cfg = common.load_config(cfg_path)
    return transport_run(cfg, seed)

@memoize
def transport_run(cfg, seed):
    # large_model = File(os.path.join(cfg._config_root_dir, cfg_fine.piezo_head_input_file))
    large_model = None

    fr_pop = Population.initialize_3d( cfg.fractures.population, cfg.geometry.box_dimensions)
    mesh_file, fractures, n_large = make_mesh(cfg, fr_pop, seed)


    # full_mesh = Mesh.load_mesh(mesh_file, heal_tol=1e-4)
    full_mesh = Mesh.load_mesh(mesh_file, heal_tol=None) # already healed

    el_to_ifr = None
    if "fractures" in cfg.geometry.include and fractures is not None:
        # modifies the regions: fr_large, fr_small
        el_to_ifr = fracture_map(full_mesh, fractures, n_large, dim=3)
        mesh_modified_filepath = os.path.join(os.path.dirname(mesh_file.path),
                                              os.path.splitext(os.path.basename(mesh_file.path))[0] + "_modified.msh2")
        mesh_modified_file = full_mesh.write_fields(mesh_modified_filepath)
    # mesh_modified = Mesh.load_mesh(mesh_modified_file)
    input_fields_file, est_velocity = compute_fields(cfg, full_mesh, apply_fields.bulk_fields_mockup_tunnel,
                                                     el_to_ifr, fractures, dim=3)

    # input_fields_file = File("input_fields.msh2")
    input_msh_filepath = os.path.splitext(input_fields_file.path)[0] + ".msh"
    shutil.copy2(input_fields_file.path, input_msh_filepath)
    input_msh = File(input_msh_filepath)

    # input_msh = File("input_fields.msh")

    return parametrized_run(cfg, large_model, input_msh)

def parametrized_run(cfg, large_model, input_fields_file):
    cfg_fine = cfg.transport_fullscale
    params = cfg_fine.copy()

    new_params = dict(
        mesh_file=input_fields_file,
        # piezo_head_input_file=large_model,
        input_fields_file = input_fields_file,
        dg_penalty = cfg_fine.dg_penalty,
        end_time_years = cfg_fine.end_time,
        trans_solver__a_tol= cfg_fine.trans_solver__a_tol,
        trans_solver__r_tol= cfg_fine.trans_solver__r_tol,
        output_times = [[t, 'y'] for t in output_times(cfg_fine)]
        #max_time_step = dt,
        #output_step = 10 * dt
    )
    params.update(new_params)
    params.update(set_source_term(cfg))
    template = os.path.join(cfg._config_root_dir, cfg_fine.input_template)
    fo = common.call_flow(cfg.machine_config, template, params)

    # return get_indicator(cfg, fo)


# def quantity_times(o_times):
#     """
#     Denser times set.
#     """
#     times = []
#     for a, b in zip(o_times[:-1], o_times[1:]):
#         step = max((b - a) / 5.0 ,  1000)
#         times.extend(np.arange(a, b, step))
#     return times
#
# @report
# def get_indicator(cfg, fo):
#     cfg_fine = cfg.transport_fullscale
#     z_dim = 0.9 * 0.5 * cfg.geometry.box_dimensions[2]
#     z_shift = cfg.geometry.borehole.z_pos
#     z_cuts = (z_shift - z_dim, z_shift + z_dim)
#     inds = indicators(fo.solute.spatial_file, f"{cfg_fine.conc_name}_conc", z_cuts)
#     plots.plot_indicators(inds)
#     #itime = IndicatorFn.common_max_time(inds)  # not splined version, need slice data
#     #plots.plot_slices(fo.solute.spatial_file, f"{cfg_fine.conc_name}_conc", z_cuts, [itime-1, itime, itime+1])
#     q_times = quantity_times(output_times(cfg_fine))
#
#     ind_value = [ind.time_max()[1] for ind in inds]
#     ind_time = [ind.time_max()[0] for ind in inds]
#     ind_series = np.array([ind.spline(q_times) for ind in inds])
#     return np.concatenate((ind_time, ind_value, ind_series.flatten()))
#


def set_source_term(cfg):
    # borehole radius
    cfg_fine = cfg.transport_fullscale
    cfg_src = cfg_fine.sources_params
    cfg_bh = cfg.geometry.storage_borehole

    source_params = dict(
        # UOS surface: S = pi * du * hu [m2]
        sources_uos_surface=np.pi * cfg_src.diameter * cfg_src.length,
        # container region volume: V = pi * dc^2/4 * hc [m3]
        sources_container_vol=np.pi * 0.25 * cfg_bh.diameter ** 2 * (cfg_bh.length - cfg_bh.plug),
        sources_buffer_thickness=cfg_src.buffer_thickness,
        conc_flux_file=os.path.join(cfg._config_root_dir, cfg_fine.conc_flux_file)
    )
    return source_params


# def compute_hm_bulk_fields(cfg, cfg_basedir, points):
#     cfg_geom = cfg.geometry
#
#     # TEST
#     # bulk_cond, bulk_por = apply_fields.bulk_fields_mockup(cfg_geom, cfg.transport_fullscale.bulk_field_params, points)
#
#     # RUN HM model
#     fo = hm_simulation.run_single_sample(cfg, cfg_basedir)
#     mesh_interp = hm_simulation.TunnelInterpolator(cfg_geom, flow123d_output=fo)
#     bulk_cond, bulk_por = apply_fields.bulk_fields_mockup_from_hm(cfg, mesh_interp, points)
#
#     # bulk_cond = apply_fields.rescale_along_xaxis(cfg_geom, bulk_cond, points)
#     # bulk_por = apply_fields.rescale_along_xaxis(cfg_geom, bulk_por, points)
#     return bulk_cond, bulk_por


if __name__ == '__main__':
    # output_dir = None
    # len_argv = len(sys.argv)
    # assert len_argv > 1, "Specify input yaml file and output dir!"
    # if len_argv == 2:
    #     output_dir = os.path.abspath(sys.argv[1])
    # output_dir = script_dir
    output_dir = os.path.join(script_dir, "output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    conf_file = os.path.join(script_dir, "config/trans_mesh_config.yaml")
    cfg = common.config.load_config(conf_file)

    common.CallCache.instance(expire_all=True)

    seed = 101
    with common.workdir(output_dir, clean=False):
        transport_run(cfg, seed)
