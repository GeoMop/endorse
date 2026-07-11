"""
Stochastic DFN sampling for upscale_m — port of R. Siddall's GenerateFractures3D_Universal to the
current bgem API (bgem@JB_homo).

API changes handled here (vs the original vendored bgem):
- DiscShape is gone; the disc reference shape is EllipseShape, which has UNIT AREA, so the bgem
  fracture scaling `Fracture.r` equals sqrt(fracture area), NOT the geometric radius:
      r_bgem = rho_geometric * sqrt(pi)
  This module's config and outputs use GEOMETRIC RADII rho (a disc r: 0.2 in config.yaml means a
  disc of radius 0.2), converted to/from bgem units internally.
- Population is built from a family config dict (Population.from_cfg: Fisher orientation +
  truncated power law + p_32 area intensity) instead of add_family();
  set_sample_range returns a new Population (functional style).

Placement logic is ported verbatim from the original: bgem samples the count, sizes and
orientations; centers are then (re)placed uniformly with safety pads so that fractures either may
intersect the domain boundary (allow_boundary_intersections: yes -> clipped later by micro_mesh)
or keep a prescribed distance from it. The special case of a fixed horizontal normal relaxes the
z-pad exactly as in the original code.

Config section (fractures.stochastic):
    seed: 1234                       # np.random seed (bgem samples via the global generator)
    p_32: 0.3                        # mean fracture area per unit volume in the population range
    power: 2.5                       # power-law exponent k_r
    population_radius_range: [0.02, 100.0]   # geometric radii of the natural population
    sample_radius_range: [0.15, 0.3]         # geometric radii actually generated
    fisher: {trend: 0.0, plunge: 90.0, concentration: 7.0}
    fixed_normal: null               # or e.g. [0, 0, 1] -> all normals overridden
    allow_boundary_intersections: 'no'
    boundary_distance: 0.05          # used when intersections are not allowed
"""
from typing import List

import numpy as np

from bgem.stochastic import Population, UniformBoxPosition, EllipseShape

from micro_mesh import MicroFracture

SQRT_PI = float(np.sqrt(np.pi))


def _to_bgem_size(rho: float) -> float:
    return rho * SQRT_PI


def _to_geometric_radius(r_bgem: float) -> float:
    return r_bgem / SQRT_PI


def stochastic_fracture_set(cfg_st, L_domain: float) -> List[MicroFracture]:
    """
    Sample a fracture set for the cubic domain [0, L_domain]^3. Returns MicroFracture list
    (geometric radii, centers in domain coordinates, unit normals).
    """
    seed = int(cfg_st.get("seed", 0))
    fixed_normal = cfg_st.get("fixed_normal", None)
    allow_intersect = str(cfg_st.get("allow_boundary_intersections", "no")).lower() in (
        "yes", "y", "true")
    boundary_distance = float(cfg_st.get("boundary_distance", 0.0))
    eps_geo = 1e-4 * L_domain

    fisher = cfg_st.fisher
    # fixed normals: near-degenerate Fisher concentration, normals overridden below (original trick)
    concentration = 1e9 if fixed_normal is not None else float(fisher.concentration)
    r_pop = cfg_st.population_radius_range
    family = dict(
        trend=float(fisher.trend), plunge=float(fisher.plunge), concentration=concentration,
        power=float(cfg_st.power),
        r_min=_to_bgem_size(float(r_pop[0])), r_max=_to_bgem_size(float(r_pop[1])),
        p_32=float(cfg_st.p_32),
    )
    box = [L_domain, L_domain, L_domain]
    population = Population.from_cfg({"fractures_upscale_m": family}, box, shape=EllipseShape())
    sample_range = [_to_bgem_size(float(r)) for r in cfg_st.sample_radius_range]
    population = population.set_sample_range(sample_range)

    np.random.seed(seed)
    fracture_set = population.sample(pos_distr=UniformBoxPosition(box), keep_nonempty=True)

    fractures = []
    n_skipped = 0
    for fr in fracture_set:
        rho = _to_geometric_radius(float(fr.r))
        if fixed_normal is not None:
            normal = np.asarray(fixed_normal, dtype=float)
        else:
            normal = np.asarray(fr.normal, dtype=float).flatten()
        normal = normal / np.linalg.norm(normal)

        # safety pads for the center placement (ported: fracture must fit incl. buffer distance)
        if not allow_intersect and boundary_distance > 0.0:
            pad_xy = rho + boundary_distance
            if fixed_normal is not None and np.allclose(normal, [0.0, 0.0, 1.0], atol=1e-5):
                pad_z = boundary_distance  # horizontal disc: z-extent is only the aperture
            else:
                pad_z = rho + boundary_distance
        else:
            pad_xy = pad_z = 0.0

        lo_xy, hi_xy = pad_xy + eps_geo, L_domain - pad_xy - eps_geo
        lo_z, hi_z = pad_z + eps_geo, L_domain - pad_z - eps_geo
        if hi_xy <= lo_xy or hi_z <= lo_z:
            n_skipped += 1
            continue
        center = np.array([np.random.uniform(lo_xy, hi_xy),
                           np.random.uniform(lo_xy, hi_xy),
                           np.random.uniform(lo_z, hi_z)])
        fractures.append(MicroFracture(r=rho, center=center, normal=normal))

    if n_skipped:
        print(f"[upscale_m dfn] {n_skipped} fracture(s) skipped: too large for the domain "
              f"with boundary_distance={boundary_distance}")
    print(f"[upscale_m dfn] sampled {len(fractures)} fracture(s), seed={seed}, "
          f"geometric radii in [{min((f.r for f in fractures), default=0):.4g}, "
          f"{max((f.r for f in fractures), default=0):.4g}]")
    return fractures
