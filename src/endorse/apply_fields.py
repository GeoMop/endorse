import numpy as np

from .common import File
from .mesh_class import Mesh
from endorse import hm_simulation

def conductivity_mockup(cfg_geom, cfg_fields, output_mesh:Mesh):
    X, Y, Z = output_mesh.el_barycenters().T
    cond_file = "fine_conductivity.msh2"
    cond_max = float(cfg_fields.cond_max)
    cond_min = float(cfg_fields.cond_min)

    edz_r = cfg_geom.edz_radius # 2.5
    in_r = cfg_fields.inner_radius
    Z = Z - cfg_geom.borehole.z_pos
    # axis wit respect to EDZ radius
    Y_rel = Y / cfg_fields.h_axis
    Z_rel = Z / cfg_fields.v_axis

    # distance from center, 1== edz_radius
    distance = np.sqrt((Y_rel * Y_rel + Z_rel * Z_rel)) / (edz_r)

    theta = (1 - distance)/(1 - in_r)
    cond_field = np.minimum(cond_max, np.maximum(cond_min, np.exp(theta * np.log(cond_max) + (1-theta) * np.log(cond_min))))
    abs_dist = np.sqrt(Y * Y + Z * Z)
    cond_field[abs_dist < cfg_geom.borehole.radius] = 1e-18
    #print({(i+1):cond for i,cond in enumerate(cond_field)})
    output_mesh.write_fields(cond_file,
                            dict(conductivity=cond_field))
    return File(cond_file)


def bulk_fields_mockup_from_hm(cfg, interp: hm_simulation.TunnelInterpolator, XYZ):
    # use Tunnel Interpolator
    # in X axis it is constant
    # X, Y, Z = XYZ.T
    points = XYZ.T
    cfg_hm = cfg.tsx_hm_model.hm_params
    cfg_geom = cfg.geometry
    selected_time = cfg_hm.end_time * 3600 * 24  # end_time of hm simulation

    cond_field = interp.interpolate_field("conductivity", points, time=selected_time)
    init_por, por_field = interp.compute_porosity(cfg_hm, points, time=selected_time)

    # x_scaling = np.where(np.logical_and(points[0, :] > 0, points[0, :] < cfg_geom.borehole.length), 1.0, 0.0)
    # cond_min = float(cfg.transport_fullscale.bulk_field_params.cond_min)
    # cond_field = np.where(np.logical_and(points[0, :] > 0, points[0, :] < cfg_geom.borehole.length), 500*cond_field, cond_min)
    # TODO: remove HACK
    cond_min = 10*np.min(cond_field)
    cond_max = np.max(cond_field)
    print(f"conductivity (min,max): ({cond_min},{cond_max})")
    bulk_cond, bulk_por = bulk_fields_mockup(cfg_geom, cfg.transport_fullscale.bulk_field_params, XYZ,
                                             cond = (cond_min, cond_max))
    cond_field = bulk_cond

    por_field = np.where(np.logical_and(points[0, :] > 0, points[0, :] < cfg_geom.borehole.length), por_field, init_por)

    return cond_field, por_field


def clip_along_xaxis(cfg_geom, field_vals, XYZ):
    X, Y, Z = XYZ.T
    x_scaling = np.where(np.logical_and(X > 0, X < cfg_geom.borehole.length), 1.0, 0.0)
    field_vals = field_vals * x_scaling
    return field_vals


def bulk_fields_mockup(cfg_geom, cfg_bulk_fields, XYZ, cond=None):
    X, Y, Z = XYZ.T

    edz_r = cfg_geom.edz_radius # 2.5
    in_r = cfg_bulk_fields.inner_radius
    Z = Z - cfg_geom.borehole.z_pos
    # axis wit respect to EDZ radius
    Y_rel = Y / cfg_bulk_fields.h_axis
    Z_rel = Z / cfg_bulk_fields.v_axis

    # distance from center, 1== edz_radius
    distance = np.sqrt((Y_rel * Y_rel + Z_rel * Z_rel)) - in_r
    theta = distance / (edz_r - in_r)
    x_scaling = np.where(np.logical_and(X > 0, X < cfg_geom.borehole.length), 1.0, 0.0)

    if cond is None:
        cond_max = float(cfg_bulk_fields.cond_max)
        cond_min = float(cfg_bulk_fields.cond_min)
    else:
        cond_min, cond_max = cond

    cond_field = np.exp((1-theta) * np.log(cond_max) + theta * np.log(cond_min)) * x_scaling
    cond_field = np.clip(cond_field, cond_min, cond_max)
    #abs_dist = np.sqrt(Y * Y + Z * Z)
    #cond_field[abs_dist < cfg_geom.borehole.radius] = 1e-18

    por_max = float(cfg_bulk_fields.por_max)
    por_min = float(cfg_bulk_fields.por_min)
    por_field = np.exp((1-theta) * np.log(por_max) + theta * np.log(por_min)) * x_scaling
    por_field = np.clip(por_field, por_min, por_max)
    #cond_field[abs_dist < cfg_geom.borehole.radius] = 1e-18
    #print({(i+1):cond for i,cond in enumerate(cond_field)})

    return cond_field, por_field

viscosity = 1e-3
gravity_accel = 10
density = 1000
permeability_to_conductivity = gravity_accel * density / viscosity

def fr_fields_repo(cfg_fr, cfg_fr_fields, fr_elements,  fr_map, fractures):
    """
    :param cfg_fr_fields:
    :param fr_elements: list of all 2d elements including the boundary elements
    :param i_begin:
    :param fr_map:
    :return:
    """
    #apperture_per_r = float(cfg_fr_fields.apperture_per_size)
    permeability_factor = permeability_to_conductivity * float(cfg_fr_fields.permeability_factor)

    families = cfg_fr.population
    fr_r = [fr.r for fr in fractures]
    fr_r.append(0.0)    # default for boundary elements
    fr_r = np.array(fr_r)
    fr_a = np.zeros_like(fr_r)
    fr_b = np.zeros_like(fr_r)
    fr_ifamily = [fr.i_family for fr in fractures]

    #fr_r = np.zeros(len(fr_elements))
    #fr_a = np.zeros_like(fr_r)
    #fr_b = np.zeros_like(fr_r)
    tr_alpha = np.array([(12 * float(f.tr_a)/permeability_factor) ** (1./3.) for f in families])
    tr_beta = np.array([float(f.tr_b) / 3. for f in families])
    fr_a[:-1] = tr_alpha[fr_ifamily]
    fr_b[:-1] = tr_beta[fr_ifamily]


    # for iel, el in enumerate(fr_elements):
    #     try:
    #         i_fr = fr_map[iel + i_begin]
    #         fr_alpha[iel] = tr_alpha[fr.i_family]
    #         fr_b[iel] = tr_beta[fr.i_family]
    #         fr_r[iel] = fr.r
    #
    #     except KeyError:
    #         pass
    # cross = 0.001

    #cross = float(cfg_fr_fields.apperture_per_size) * fr_r
    fr_cross = fr_a * fr_r ** fr_b
    # cond = 0.01
    #cond = float(cfg_fr_fields.permeability_factor) * permeability_to_conductivity/12 * cross * cross
    fr_cond =  permeability_factor / 12 * fr_cross * fr_cross
    fr_porosity = np.full_like(fr_cond, 1.0)

    return fr_cond[fr_map], fr_cross[fr_map], fr_porosity[fr_map]


def fr_fields_prametric(cfg_fr, cfg_fr_fields, fr_elements,  fr_map, fractures):
    """
    :param cfg_fr_fields:
    :param fr_elements: list of all 2d elements including the boundary elements
    :param i_begin:
    :param fr_map:
    :return:
    """
    apperture_per_r = float(cfg_fr_fields.apperture_per_size)
    permeability_factor = float(cfg_fr_fields.permeability_factor)

    families = cfg_fr.population
    fr_r = [fr.r for fr in fractures]
    fr_r.append(0.0)    # default for boundary elements
    fr_r = np.array(fr_r)
    fr_a = np.zeros_like(fr_r)
    fr_b = np.zeros_like(fr_r)
    #fr_ifamily = [fr.i_family for fr in fractures]

    #fr_r = np.zeros(len(fr_elements))
    #fr_a = np.zeros_like(fr_r)
    #fr_b = np.zeros_like(fr_r)
    #tr_alpha = np.array([(12 * float(f.tr_a)/(permeability_to_conductivity * ) ** (1./3.) for f in families])
    #tr_beta = np.array([float(f.tr_b) / 3. for f in families])
    fr_a[:-1] = apperture_per_r
    fr_b[:-1] = 1


    fr_cross = fr_a * fr_r ** fr_b
    # cond = 0.01
    #cond = float(cfg_fr_fields.permeability_factor) * permeability_to_conductivity/12 * cross * cross
    fr_cond =  permeability_factor * permeability_to_conductivity / 12 * fr_cross * fr_cross
    fr_porosity = np.full_like(fr_cond, 1.0)

    return fr_cond[fr_map], fr_cross[fr_map], fr_porosity[fr_map]
