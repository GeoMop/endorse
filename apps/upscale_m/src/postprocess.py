"""
Volume integration of stress and strain over endorse-side averaging windows.

Port of R. Siddall's GeneralComputationClass restricted to the micro mesh of micro_mesh.py:

  <sigma> = (1/V) [ sum_bulk w_e sigma_e V_e  +  sum_frac w_f sigma_f A_f delta ]
  <eps>   = (1/V) [ sum_bulk w_e eps_e   V_e  +  sum_frac w_f eps_f   A_f delta ],   eps = M : sigma

Built on the ORIGINAL endorse/bgem machinery, mirroring macro_flow_model.micro_postprocess:
  - the window grid is a bgem `upscale.fem.Grid` (created by the caller; C-style numbering with
    the z index running fastest = the subdomain_{ix}_{iy}_{iz} report convention),
  - the windows enter endorse as a synthetic macro mesh (`windows_macro_mesh`: one regular
    tetrahedron per window, consumed by MacroCube via barycenter/vertex distances only),
  - BULK (matrix) elements are averaged by endorse `homogenisation.Subproblems` /
    `SubMeshSubproblem.assembly_average_matrix` / `Subproblems.subdomains_average` with the
    MacroCube window shape (each element enters with its volume-fraction weight,
    MacroCube.interact). The endorse average is normalized by the captured volume, so it is
    rescaled by V_captured back to the un-normalized sum of the validated formula
    (denominator = window volume),
  - FRACTURE 2D elements are added by the analogous `FractureSubdomain` (endorse Subdomain is
    bulk-only: dim=3 slice), with area-fraction weights and the half-open interface convention:
    interior window interfaces are [lo, hi), the outer rim inclusive — a fracture lying exactly
    ON an interface is counted in exactly one window,
  - fields are read by endorse mesh_class P0 readers, Voigt conversions by bgem tn_to_voigt.

The only local physics is `hooke_strain` — the isotropic inverse Hooke law
eps = ((1+nu) sigma - nu tr(sigma) I) / E (report sec. 3.4.2). AUDITED: no isotropic
stiffness/compliance builder exists anywhere in endorse or bgem (see PLAN.md log 2026-07-11).
Fracture elements are thin elastic plates weighted by the aperture delta (report sec. 3.1); the
adequacy of this treatment vs the explicit displacement-jump term J is an open question in PLAN.md.

Voigt convention (report): sigma = [11, 22, 33, 23, 13, 12]; strain shear rows carry engineering
shear gamma = 2 eps.
"""
from functools import lru_cache
from typing import Dict, List, Tuple

import attrs
import numpy as np

from bgem.gmsh import gmsh_io
from endorse.homogenisation import Subproblems
from endorse.macro_flow_model import refine_barycenters
from endorse.mesh_class import Mesh

# engineering shear on the Voigt strain shear rows: gamma = 2 eps (report convention)
ENG_SHEAR = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def hooke_strain(sigma: np.ndarray, young: float, poisson: float) -> np.ndarray:
    """
    Isotropic inverse Hooke law, tensor form: eps = ((1+nu) sigma - nu tr(sigma) I) / E.
    Input/output (N, 3, 3). (No such builder exists in endorse/bgem — audited, PLAN.md.)
    """
    trace = np.trace(sigma, axis1=1, axis2=2)
    return ((1.0 + poisson) * sigma - poisson * trace[:, None, None] * np.eye(3)) / young


@lru_cache(maxsize=None)
def _simplex_barycentric_weights(n_vertices: int, level: int) -> np.ndarray:
    """
    Barycentric weights (n_sub, n_vertices) of the endorse refine_barycenters sub-element
    barycenters, computed ONCE on the reference simplex and mapped to any actual element by
    `weights @ element_vertices` (affine invariance of barycentric combinations).
    Needed because endorse refine_element asserts num_vertices == dim + 1 and thus rejects
    TRIANGLES EMBEDDED IN 3D, although it has the triangle refinement tables
    on the (k-1)-dimensional reference simplex the assert holds.
    """
    ref_simplex = np.eye(n_vertices)[:, 1:]  # rows: origin + unit vectors, shape (k, k-1)
    local = refine_barycenters(ref_simplex, level)
    return np.concatenate([1.0 - local.sum(axis=1, keepdims=True), local], axis=1)


def _window_weights(vertices: np.ndarray, lo_eff: np.ndarray, hi_eff: np.ndarray,
                    level: int) -> np.ndarray:
    """
    Fraction of each simplex (shape (n_el, k, 3)) inside the half-open window (bounds carry the
    rim tolerance). Exact 1/0 shortcuts (window is convex); elements cut by the window boundary
    are estimated from the barycenters of the endorse refine_barycenters sub-elements
    (as in MacroCube.interact). Used for the FRACTURE side; bulk goes through endorse Subdomain.
    """
    inside = lambda points: np.all((points > lo_eff) & (points < hi_eff), axis=-1)
    node_in = inside(vertices)
    weights = np.zeros(len(vertices))
    weights[np.all(node_in, axis=1)] = 1.0
    aabb_out = (np.any(np.min(vertices, axis=1) > hi_eff, axis=1)
                | np.any(np.max(vertices, axis=1) < lo_eff, axis=1))
    cut = np.where(~np.all(node_in, axis=1) & ~aabb_out)[0]
    if cut.size:
        w_ref = _simplex_barycentric_weights(vertices.shape[1], level)
        sub_barycenters = np.einsum("sk,nkd->nsd", w_ref, vertices[cut])
        weights[cut] = np.mean(inside(sub_barycenters), axis=1)
    return weights


def windows_macro_mesh(grid: 'bgem.upscale.fem.Grid') -> Mesh:
    """
    Synthetic endorse macro Mesh carrying the averaging windows of the bgem Grid for
    homogenisation.Subproblems: ONE REGULAR TETRAHEDRON per grid cell with barycenter = cell
    center (exact: symmetric vertex directions cancel) and mean vertex distance = the cell
    half-edge, so MacroCube(rel_size=1) reproduces the cubic window exactly. The endorse
    machinery consumes only barycenter()/vertices() of the macro elements.
    """
    assert np.allclose(grid.step, grid.step[0]), \
        f"Non-cubic averaging window {grid.step}: needs a MacroBox shape (see PLAN.md)."
    directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3.0)
    half_edge = grid.step[0] / 2.0
    gio = gmsh_io.GmshIO()
    for i, center in enumerate(grid.barycenters()):
        vertex_ids = list(range(4 * i + 1, 4 * i + 5))
        for node_id, direction in zip(vertex_ids, directions):
            gio.nodes[node_id] = center + half_edge * direction
        gio.elements[i + 1] = (4, (1, 1), vertex_ids)  # gmsh type 4 = tetrahedron
    return Mesh(gio, file=None)


@attrs.define
class FractureSubdomain:
    """
    Aperture-weighted 2D fracture analogue of endorse homogenisation.Subdomain (whose create()
    is bulk-only: it keeps dim=3 elements — upstream question in PLAN.md). Candidates come from
    the endorse BIH (Mesh.candidate_indices) restricted to the fracture regions (one PER
    fracture — bgem geometry_gmsh's fam_<family>_<iii>, never joined into a single region); the
    intersect weight is the AREA fraction inside the HALF-OPEN window (bounds carry the rim
    tolerance, so a fracture lying exactly ON an interior window interface is counted in exactly
    one window). The element measure is area * cross_section[el] — the PER-ELEMENT aperture from
    the solver output (thin-plate overlay, report sec. 3.1).
    """
    mesh: Mesh
    el_indices: np.ndarray
    intersect_weights: np.ndarray
    # full-mesh per-element aperture field (solver output 'cross_section')
    cross_section: np.ndarray

    @staticmethod
    def create(micro_mesh: Mesh, region_id_field: np.ndarray, fracture_region_ids: List[int],
               lo_eff: np.ndarray, hi_eff: np.ndarray, cross_section: np.ndarray,
               level: int = 2) -> 'FractureSubdomain':
        empty = FractureSubdomain(micro_mesh, np.array([], dtype=int), np.array([]),
                                  cross_section)
        if not fracture_region_ids:
            return empty
        fracture_region_id_set = set(fracture_region_ids)
        candidates = micro_mesh.candidate_indices(np.array([lo_eff, hi_eff]))
        frac_els = np.array([ie for ie in candidates
                             if region_id_field[ie] in fracture_region_id_set], dtype=int)
        if frac_els.size == 0:
            return empty
        vertices = np.array([micro_mesh.elements[ie].vertices() for ie in frac_els])
        weights = _window_weights(vertices, lo_eff, hi_eff, level)
        keep = weights > 0.0
        return FractureSubdomain(micro_mesh, frac_els[keep], weights[keep], cross_section)

    @property
    def measures(self) -> np.ndarray:
        """area * per-element aperture (endorse Mesh.el_volumes covers tets only)."""
        if self.el_indices.size == 0:
            return np.array([])
        vertices = np.array([self.mesh.elements[ie].vertices() for ie in self.el_indices])
        cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        return 0.5 * np.linalg.norm(cross, axis=1) * self.cross_section[self.el_indices]

    def weighted_sum(self, element_vec_data: np.ndarray) -> np.ndarray:
        """Un-normalized weighted sum over the fracture elements (counterpart of the bulk
        endorse average * captured volume)."""
        if self.el_indices.size == 0:
            return np.zeros(element_vec_data.shape[1])
        return (self.measures * self.intersect_weights) @ element_vec_data[self.el_indices]


def average_windows(output_mesh: Mesh, aperture_by_region_id: Dict[int, float],
                    subproblems: Subproblems, grid: 'bgem.upscale.fem.Grid',
                    young_rock: float, poisson_rock: float,
                    young_fracture: float, poisson_fracture: float,
                    bulk_region_id: int, fracture_region_ids: List[int] = None,
                    level: int = 2,
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Volume-average strain and stress over the grid windows — one (eps_avg, sigma_avg) 3x3 pair
    per window, ordered as the grid cells:

      <x> = [ endorse_bulk_sum(x) + FractureSubdomain_sum(x) ] / V_ref,  V_ref = window volume

    The bulk side is pure endorse: per subproblem `assembly_average_matrix(output_mesh) @ field`
    (rows are volume-normalized -> rescaled by the captured volume), assembled across
    subproblems by `Subproblems.subdomains_average` — the exact structure of
    macro_flow_model.micro_postprocess/micro_load_response.

    aperture_by_region_id is the UNDEFORMED per-fracture aperture (report sec. 3.1), computed
    ONCE (run_kubc) from the fracture radii — never `cross_section_updated`, which would
    reflect mechanical deformation. Keyed by REGION id, not element id: Flow123d renumbers
    elements in its own output (it does not preserve the pre-solve mesh's element ids — e.g.
    boundary-only faces are dropped from its own element count), so an element-id lookup built
    from a separately-loaded file would silently misalign; region ids are explicit DATA rather
    than an internal numbering scheme and ARE preserved, which is all this needs since aperture
    never varies within one fracture region anyway.
    """
    # endorse readers; time 0.0 = first output time = the steady mechanics solution
    rid = np.asarray(output_mesh.get_p0_values("region_id", 0.0)).astype(int).ravel()
    stress9 = np.asarray(output_mesh.get_p0_values("stress", 0.0), dtype=float)
    assert stress9.shape[1] == 9, f"expected 9-component stress, got {stress9.shape}"
    cross_section = np.array([aperture_by_region_id.get(r, 0.0) for r in rid])

    # full-mesh strain field with the per-region compliance, so the endorse averaging
    # matrix can consume it directly (rows of non-selected regions stay zero)
    eps9 = np.zeros_like(stress9)
    bulk_rows = rid == bulk_region_id
    assert bulk_rows.any(), f"No bulk cells with region id {bulk_region_id} in the output."
    eps9[bulk_rows] = hooke_strain(
        stress9[bulk_rows].reshape(-1, 3, 3), young_rock, poisson_rock).reshape(-1, 9)
    if fracture_region_ids:
        frac_rows = np.isin(rid, fracture_region_ids)
        eps9[frac_rows] = hooke_strain(
            stress9[frac_rows].reshape(-1, 3, 3), young_fracture, poisson_fracture).reshape(-1, 9)

    # --- bulk: ORIGINAL endorse Subproblems machinery (Subdomain.el_indices/intersect_weights,
    # from Subdomain.create's BIH search + MacroCube.interact) ------------------------------
    # subdomains(output_mesh) is expensive (BIH search over the whole mesh) and joblib-memoized
    # per call, so it is called EXACTLY ONCE per subproblem here — NOT also through
    # assembly_average_matrix (which internally calls it again): that second call cannot reuse
    # the first (each load case loads output_mesh fresh, so the memoization key never hits),
    # roughly doubling the wall time per load case for no benefit. The un-normalized weighted
    # sum computed directly below is exactly assembly_average_matrix's row (volumes/V_captured)
    # multiplied back by V_captured, i.e. mathematically identical to the previous route.
    el_volumes = np.asarray(output_mesh.el_volumes)
    per_subproblem = []
    for subproblem in subproblems.subproblems:
        sub_sums = []
        for sd in subproblem.subdomains(output_mesh):
            captured_volumes = el_volumes[sd.el_indices] * np.asarray(sd.intersect_weights)
            sub_sums.append(np.concatenate([
                captured_volumes @ stress9[sd.el_indices],
                captured_volumes @ eps9[sd.el_indices],
            ]))
        per_subproblem.append(np.array(sub_sums))
    sums = subproblems.subdomains_average(per_subproblem)  # (n_windows, 18), grid cell order
    sigma_sums, eps_sums = sums[:, :9], sums[:, 9:]

    # --- fractures: the 2D analogue, half-open window bounds --------------------------------
    # topmost faces of the whole grid stay INCLUSIVE, interior interfaces half-open [lo, hi)
    grid_hi = grid.origin + grid.dimensions
    half = grid.step / 2.0
    V_ref = float(np.prod(grid.step))
    results = []
    for i_window, center in enumerate(grid.barycenters()):
        lo3, hi3 = center - half, center + half
        tol = 1e-8 * max(1.0, float(np.max(grid.step)))
        upper_rim = np.isclose(hi3, grid_hi, rtol=0.0, atol=tol)
        fsub = FractureSubdomain.create(output_mesh, rid, fracture_region_ids,
                                        lo3 - tol, np.where(upper_rim, hi3 + tol, hi3 - tol),
                                        cross_section, level)
        sigma_avg = (sigma_sums[i_window] + fsub.weighted_sum(stress9)) / V_ref
        eps_avg = (eps_sums[i_window] + fsub.weighted_sum(eps9)) / V_ref
        results.append((eps_avg.reshape(3, 3), sigma_avg.reshape(3, 3)))
    return results
