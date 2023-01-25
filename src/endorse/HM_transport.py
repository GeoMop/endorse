import logging
import os
import shutil
from typing import *

import numpy as np

from . import common
from .common import dotdict, memoize, File, call_flow, workdir, report
from .mesh import repository_mesh as repo_mesh, mesh_tools
from .homogenisation import  subdomains_mesh, Homogenisation, Subdomain, MacroSphere, make_subproblems, Subproblems
from .mesh.repository_mesh import one_borehole
from .mesh_class import Mesh
from . import apply_fields
from . import plots
from . import flow123d_inputs_path
from .indicator import indicators, IndicatorFn
from bgem.stochastic.fracture import Fracture
from endorse import hm_simulation

def input_files(cfg):
    return [
        cfg.transport_fullscale.piezo_head_input_file,
        cfg.transport_fullscale.conc_flux_file,
        "test_data/accepted_parameters.csv"
    ]

#
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx, array[idx]

def fullscale_transport(cfg_path, source_params, seed):
    """
    1. apply conouctivity to given mesh:
       - on borehole neighbourhood, select elements
       - calculate barycenters
       - apply conductivity
       - write the field
    2. substitute source term space distribution
    3. return necessary files
    """
    cfg = common.load_config(cfg_path)
    cfg_basedir = os.path.dirname(cfg_path)
    #files = input_files(cfg.transport_fullscale)

    cfg_fine = cfg.transport_fullscale
    large_model = File(os.path.join(cfg_basedir, cfg_fine.piezo_head_input_file))
    conc_flux = File(os.path.join(cfg_basedir, cfg_fine.conc_flux_file))
    plots.plot_source(conc_flux)

    full_mesh_file, fractures, n_large = fullscale_transport_mesh(cfg_fine, seed)

    full_mesh = Mesh.load_mesh(full_mesh_file, heal_tol=1e-4)
    el_to_ifr = fracture_map(full_mesh, fractures, n_large)
    # mesh_modified_file = full_mesh.write_fields("mesh_modified.msh2")
    # mesh_modified = Mesh.load_mesh(mesh_modified_file)

    input_fields_file, est_velocity = compute_fields(cfg, cfg_basedir, full_mesh, el_to_ifr, fractures)

    # input_fields_file = compute_fields(cfg, full_mesh, el_to_fr)
    params = cfg_fine.copy()

    # estimate times
    bulk_vel_est, fr_vel_est = est_velocity
    end_time = (50 / bulk_vel_est + 50 / fr_vel_est)
    dt = 0.5 / bulk_vel_est
    # convert to years

    end_time = end_time / common.year
    dt = dt / common.year

    #end_time = 10 * dt
    new_params = dict(
        mesh_file=input_fields_file,
        piezo_head_input_file=large_model,
        conc_flux_file=conc_flux,
        input_fields_file = input_fields_file,
        end_time = end_time,
        max_time_step = dt,
        output_step = 10 * dt
    )
    params.update(new_params)
    params.update(set_source_limits(cfg))
    template = flow123d_inputs_path.joinpath(cfg_fine.input_template)
    fo = common.call_flow(cfg.flow_env, template, params)
    z_dim = 0.9 * 0.5 * cfg.geometry.box_dimensions[2]
    z_shift = cfg.geometry.borehole.z_pos
    z_cuts = (z_shift - z_dim, z_shift + z_dim)
    inds = indicators(fo.solute.spatial_file, f"{cfg_fine.conc_name}_conc", z_cuts)
    plots.plot_indicators(inds)
    itime = IndicatorFn.common_max_time(inds)  # not splined version, need slice data
    plots.plot_slices(fo.solute.spatial_file, f"{cfg_fine.conc_name}_conc", z_cuts, [itime-1, itime, itime+1])
    ind_time_max = [ind.time_max()[1] for ind in inds]
    return ind_time_max

def fracture_map(mesh, fractures, n_large) -> Dict[int, Fracture]:
    """
    - join all fracture regions into single "fractures" region
    - return dictionary mapping element idx to fracture
    :param mesh:
    :param fractures:
    :return:
    """
    own_name_to_id = {fr.region.name: fr.region.id for fr in fractures}
    own_to_gmsh_id = {own_name_to_id[name]: gmsh_id for name,(gmsh_id, dim) in mesh.gmsh_io.physical.items() if name in own_name_to_id}

    max_reg = max( [gmsh_id for gmsh_id, dim in mesh.gmsh_io.physical.values()] )
    small_reg_id = max_reg + 1
    large_reg_id = max_reg + 2
    large_reg_map = {(own_to_gmsh_id[fr.region.id], 2): (large_reg_id, 2, "fr_large")  for fr in fractures[:n_large]}
    small_reg_map = {(own_to_gmsh_id[fr.region.id], 2): (small_reg_id, 2, "fr_small")  for fr in fractures[n_large:]}
    new_reg_map = large_reg_map
    new_reg_map.update(small_reg_map)

    # if do_heal:
    #     hm.heal_mesh(gamma_tol=0.01)
    #     hm.move_all(geom_dict["shift_vec"])
    #     elm_to_orig_reg = hm.map_regions(new_reg_map)
    #     hm.stats_to_yaml(mesh_name + "_heal_stats.yaml")
    #     assert hm.healed_mesh_name == mesh_healed
    #     hm.write()
    # else:
    iel_to_orig_reg = mesh.map_regions(new_reg_map)

    reg_to_ifr = {own_to_gmsh_id[fr.region.id]: ifr for ifr, fr in enumerate(fractures)}
    elm_to_ifr = {el_idx: reg_to_ifr[reg_id] for el_idx, (reg_id, dim) in iel_to_orig_reg.items()}
    return elm_to_ifr

def set_source_limits(cfg):
    geom = cfg.geometry
    br = geom.borehole.radius

    cfg_trans = cfg.transport_fullscale
    cfg_source = cfg_trans.source_params
    x_pos = cfg_source.source_ipos * (cfg_source.source_length + cfg_source.source_space)
    source_params = dict(
        source_y0=-2 * br,
        source_y1=2 * br,
        source_x0=x_pos,
        source_x1=x_pos + cfg_source.source_length,
    )
    return source_params


def compute_fields(cfg:dotdict, cfg_basedir, mesh:Mesh, fr_map: Dict[int, int], fractures:List[Fracture]):
    """
    :param params: transport parameters dictionary
    :param mesh: GmshIO of the computational mesh (read only)
    :param fr_map: map ele id to the fracture (only for fracture 2d elements
    :return: el_ids:List[int], cond:List[float], cross:List[float]
    """
    cfg_trans = cfg.transport_fullscale
    cfg_bulk_fields = cfg_trans.bulk_field_params

    conductivity = np.full( (len(mesh.elements),), float(cfg_bulk_fields.cond_min))
    cross_section = np.full( (len(mesh.elements),), float(1.0))
    porosity = np.full((len(mesh.elements),), 1.0)
    el_slice_3d = mesh.el_dim_slice(3)
    # Bulk fields
    bulk_cond, bulk_por = compute_hm_bulk_fields(cfg, cfg_basedir, mesh.el_barycenters()[el_slice_3d])
    conductivity[el_slice_3d] = bulk_cond
    porosity[el_slice_3d] = bulk_por
    logging.info(f"3D slice: {el_slice_3d}")
    c_min, c_max = np.min(conductivity), np.max(conductivity)
    logging.info(f"cond range: {c_min}, {c_max}")
    plots.plot_field(mesh.el_barycenters()[el_slice_3d], bulk_cond, file="conductivity_yz.pdf")
    plots.plot_field(mesh.el_barycenters()[el_slice_3d], bulk_por, file="porosity_yz.pdf")

    # Fracture
    cfg_fr = cfg_trans.fractures
    cfg_fr_fields = cfg_trans.fr_field_params
    el_slice_2d = mesh.el_dim_slice(2)
    logging.info(f"2D slice: {el_slice_2d}")
    i_default = len(fractures)
    fr_map_slice = [fr_map.get(i, i_default) for i in range(el_slice_2d.start, el_slice_2d.stop)]
    fr_cond, fr_cross, fr_por = apply_fields.fr_fields_repo(cfg_fr, cfg_fr_fields,
                                                            mesh.elements[el_slice_2d], fr_map_slice, fractures)
    conductivity[el_slice_2d] = fr_cond
    cross_section[el_slice_2d] = fr_cross
    porosity[el_slice_2d] = fr_por
    fields = dict(
        conductivity=conductivity,
        cross_section=cross_section,
        porosity=porosity
    )
    cond_file = mesh.write_fields("input_fields.msh2", fields)

    # estimate velocities on bulk and fracture
    # for cond range 1e-13 - 1e-9 and porosity about 1, we have velocity 1e-16  to 5.5e-10
    # i.e velocity about the order of conductivity or one order less
    # for fracture, cond range:

    pos_fr = fr_cond > 0
    est_velocity = (np.quantile(bulk_cond, 0.4)/10, np.quantile(fr_cond[pos_fr],  0.4))
    return cond_file, est_velocity

def compute_hm_bulk_fields(cfg, cfg_basedir, points):
    cfg_geom = cfg.geometry

    # TEST
    # bulk_cond, bulk_por = apply_fields.bulk_fields_mockup(cfg_geom, cfg.transport_fullscale.bulk_field_params, points)

    # RUN HM model
    fo = hm_simulation.run_single_sample(cfg, cfg_basedir)
    mesh_interp = hm_simulation.TunnelInterpolator(cfg_geom, flow123d_output=fo)
    bulk_cond, bulk_por = apply_fields.bulk_fields_mockup_from_hm(cfg, mesh_interp, points)

    # bulk_cond = apply_fields.rescale_along_xaxis(cfg_geom, bulk_cond, points)
    # bulk_por = apply_fields.rescale_along_xaxis(cfg_geom, bulk_por, points)
    return bulk_cond, bulk_por


@report
@memoize
def fullscale_transport_mesh(cfg, seed):
    main_box_dimensions = cfg.geometry.box_dimensions

    # Fixed large fractures
    fix_seed = cfg.fractures.fixed_seed
    large_min_r = cfg.fractures.large_min_r
    large_box_dimensions = cfg.fractures.large_box
    fractures = mesh_tools.generate_fractures(cfg.fractures, (large_min_r, None), large_box_dimensions, fix_seed)
    n_large = len(fractures)
    # random small scale fractures
    small_fr = mesh_tools.generate_fractures(cfg.fractures, (None, large_min_r), main_box_dimensions, seed)
    fractures.extend(small_fr)
    logging.info(f"Generated fractures: {n_large} large, {len(small_fr)} small.")
    return one_borehole(cfg.geometry, fractures, cfg.mesh), fractures, n_large


# def transport_observe_points(cfg):
#     cfg_geom = cfg.geometry
#     lx, ly, lz = cfg_geom.box_dimensions
#     np.
#     with "observe_points.csv":
#
# observe_points:
# - [0, 0.1, 0]
# - {point: [0.55, 0.55, 0], snap_region: 1d_lower}
# - {point: [0.7, 0.8, 0], snap_region: 1d_upper}
