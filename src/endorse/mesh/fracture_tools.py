import logging
import yaml

from bgem.stochastic.fracture import Population
from endorse.mesh import mesh_tools

def create_fractures_rectangles(gmsh_geom, fractures, shift, base_shape: 'ObjectSet'):
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    if len(fractures) == 0:
        return []


    shapes = []
    for i, fr in enumerate(fractures):
        shape = base_shape.copy()
        print("fr: ", i, "tag: ", shape.dim_tags)
        shape = shape.scale([fr.rx, fr.ry, 1]) \
            .rotate(axis=[0,0,1], angle=fr.shape_angle) \
            .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
            .translate(fr.center + shift).set_region(fr.region)

        shapes.append(shape)

    fracture_fragments = gmsh_geom.fragment(*shapes)
    return fracture_fragments


def fr_dict_repr(fr):
    return dict(r=float(fr.r), normal=fr.normal.tolist(), center=fr.center.tolist(),
                aspect=float(fr.aspect), shape_angle=float(fr.shape_angle), region=fr.region.name)


def fracture_set(cfg, fr_population:Population, seed):
    main_box_dimensions = cfg.geometry.box_dimensions

    # Fixed large fractures
    fix_seed = cfg.fractures.fixed_seed
    large_min_r = cfg.fractures.large_min_r
    large_box_dimensions = cfg.fractures.large_box
    fr_limit = cfg.fractures.n_frac_limit
    logging.info(f"Large fracture seed: {fix_seed}")
    max_large_size = max([fam.size.diam_range[1] for fam in fr_population.families])
    fractures = mesh_tools.generate_fractures(fr_population, (large_min_r, max_large_size), fr_limit, large_box_dimensions, fix_seed)

    large_fr_dict=dict(seed=fix_seed, fr_set=[fr_dict_repr(fr) for fr in fractures])
    with open(f"large_Fr_set.yaml", "w") as f:
        yaml.dump(large_fr_dict, f, sort_keys=False)
    n_large = len(fractures)
    #if n_large == 0:
    #    raise ValueError()
    # random small scale fractures
    small_fr = mesh_tools.generate_fractures(fr_population, (None, large_min_r), fr_limit, main_box_dimensions, seed)
    fractures.extend(small_fr)
    logging.info(f"Generated fractures: {n_large} large, {len(small_fr)} small.")
    return fractures, n_large
