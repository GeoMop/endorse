
"""
Implement static homogenisation.
Setup an object collection subproblems, covering some domain.
Do subdomains calculations for various boundary conditions. These sampling could be done iteratively.
For given set of evaluation points, evaluate the properties in these points:
1. find appropriate subdomain with max. intersection with the negbourhood
   This step provides topological ingormation that is independent of the evaluatied quantities.
2. Compute ekvivalent properties in the points:
   scalar - just volume avarage with kernel weight
   vector - just volume avarage with kernel weight
   tensor - implicitely by 2 vector quantities
   tensor 4 - implicitely by 2 tensor quantities (could be outer product of vectors)

   symmetric tensor from outer product: a2+b1 , a3+c1, b3+c2
"""
import logging
from typing import *
import os
import numpy as np
from bgem.gmsh.gmsh_io import GmshIO
import bih
import attrs
import copy
from scipy import sparse
from .mesh_class import Mesh, Element
from . import common
from .common import dotdict, memoize, File, report

class MacroShapeBase:
    pass

# Macro element shapes, currently just the sphere.
# Shape works as a factory for actual macro elements
@attrs.define
class MacroSphere(MacroShapeBase):
    # radius relative to average distance of vertices form te barycenter
    # More nuances could be done about actual placing of the ball of given size to best match the tetrahedral element.
    rel_radius: float

    def _center_radius(self, macro_el:Element):
        center = macro_el.barycenter()
        distances = np.linalg.norm(macro_el.vertices() - center[None,:], axis=1)
        r = np.mean(distances)
        return center, r

    # could possibly calculate actual center and radius, but in fact we only needs aabb and interaction indicator for micro mesh elements
    def aabb(self, macro_el:Element):
        center, r = self._center_radius(macro_el)
        return np.array([center - r, center + r])

    def interact(self, macro_el:Element, micro_el: Element):
        center, r = self._center_radius(macro_el)
        nodes = micro_el.vertices()
        nc = nodes[:, :] - center[None, :]
        indicate = np.sum(nc * nc, axis=1) < r*r    # shape n - nodes
        return np.any(indicate)

@attrs.define
class SubMeshSubproblem:
    """
    Subproblems extraced from a single fine mesh.
    TODO: alternative, subproblems extracted from fine mesh geometry.
    """
    macro_mesh: Mesh
    micro_mesh: Mesh
    macro_el_shape: MacroShapeBase
    # bounding box of the subproblem
    aabb: np.array
    # indices of macro elements
    macro_elements: np.array
    # indices of subdomains (within list of homogenized macro elements)
    i_subdomains: np.array

    _micro_elements: np.array = None
    _submesh: Mesh = None
    _subdomains: List['Subdomain'] = None
    _average_sparse_matrix: sparse.csr_matrix = None

    @property
    def micro_elements(self):
        if self._micro_elements is None:
            self._micro_elements = self.micro_mesh.candidate_indices(self.aabb)
        return self._micro_elements

    @property
    def submesh(self):
        """Create subproblem mesh."""
        if self._submesh is None:
            self._submesh = self.micro_mesh.submesh(self.micro_elements, "noname")
            print(f"Subproblem AABB: {self.aabb} submesh: {repr_aabb(self._submesh.bih.aabb())}")
        return self._submesh

    @property
    def subdomains(self):
        """Create subproblem mesh."""
        if self._subdomains is None:
            self._subdomains = [Subdomain.create(self.macro_el_shape, self.submesh, self.macro_mesh, iel) for iel in self.macro_elements]

        return self._subdomains

    @report
    def assembly_average_matrix(self):
        rows = []
        cols = []
        vals = []
        for i_sub, sub in enumerate(self.subdomains):
            sub_col = sub.el_indices
            sub_val = sub.weights
            rows.append(np.full_like(sub_col, i_sub))
            cols.append(sub_col)
            vals.append(sub_val)
        return sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(len(self.subdomains), len(self.micro_elements))).tocsr()


    @property
    def average_sparse_matrix(self):
        if self._average_sparse_matrix is None:
            self._average_sparse_matrix = self.assembly_average_matrix()
        return self._average_sparse_matrix


    def average(self, field:np.array):
        assert field.shape[0] == len(self.micro_elements)
        print(f"{self.average_sparse_matrix.shape} @ {field.shape}")
        return self.average_sparse_matrix @ field


def bin_intervals(intervals, n_bins):
    centers = (intervals[:, 0] + intervals[:, 1]) / 2
    bins = np.linspace(np.min(intervals[:, 0]), np.max(intervals[:, 1]), n_bins + 1)
    return np.digitize(centers, bins[1:-1])

def assign_to_subproblems(boxes, subdivision):
    """
    Split the AABB of the macro_mesh macro elements to the subdomains according to the subdivision
    vector providing number of subdomains [n_x, n_y, n_z] in every direction. The n_x * n_y * n_z subdomains will be used.
    Assign every macro element aabb to single subproblem according to the AABB center.
    Return array of subproblem index for every macro element.
    TODO: improve covering for irregular shapes and/or refined meshes, could possibly use a Metis or so.
    """
    i_bin_axis = [ bin_intervals(boxes[:, :, axis], subdivision[axis]) for axis in range(3)]
    i_subproblem = i_bin_axis[0] + subdivision[0] * (i_bin_axis[1] + subdivision[1] * i_bin_axis[2])
    return  i_subproblem


def subproblem_boxes(macro_boxes, subdivision):
    i_subproblems = assign_to_subproblems(macro_boxes, subdivision)
    perm = np.argsort(i_subproblems)
    i_subp_sorted = i_subproblems[perm]
    macro_boxes_sorted = macro_boxes[perm]
    _, idx = np.unique(i_subp_sorted, return_index=True)
    sub_boxes = np.empty((np.prod(subdivision), 2, 3))
    for axis in range(3):
        # estimate overlap
        max_box = np.maximum.reduceat(macro_boxes_sorted[:,1,axis] - macro_boxes_sorted[:,0,axis], idx)
        sub_boxes[:, 0, axis] = np.minimum.reduceat(macro_boxes_sorted[:, 0, axis], idx) - max_box[:]
        sub_boxes[:, 1, axis] = np.maximum.reduceat(macro_boxes_sorted[:, 1, axis], idx) + max_box[:]
        print(max_box)
        # for i_el, i_subp in enumerate(i_subp_sorted):
        #     assert sub_boxes[i_subp, 0, axis] <= macro_boxes[i_el, 0, axis]
        #     assert sub_boxes[i_subp, 1, axis] >= macro_boxes[i_el, 1, axis]

    return sub_boxes, i_subproblems

def make_subproblems(macro_mesh, macro_els, micro_mesh:Mesh, macro_shape:MacroShapeBase, subdivision:np.array) -> List[SubMeshSubproblem]:
    """
    Could be modified
    :param micro_mesh:
    :param macro_mesh:
    :return:
    """
    macro_boxes = np.array([macro_shape.aabb(macro_mesh.elements[iel]) for iel in macro_els])

    sub_boxes, macro_to_subp = subproblem_boxes(macro_boxes, subdivision)
    sub_lists = [[] for _ in sub_boxes]
    for i_subdomain, i_sub in enumerate(macro_to_subp):
        sub_lists[i_sub].append(i_subdomain)

    def make_sub(box, i_subdomains):
        i_macro_els = [macro_els[i] for i in i_subdomains]
        for isub in i_subdomains:
            #print(f"Macro El AABB {isub}: {macro_boxes[isub]} new AABB: {macro_shape.aabb(macro_mesh.elements[macro_els[isub]])}")
            assert np.all(box[0, :] <= macro_boxes[isub][0,:]), f"{box} >? {macro_boxes[isub]}"
            assert np.all(box[1, :] >= macro_boxes[isub][1,:]), f"{box} >? {macro_boxes[isub]}"
        return SubMeshSubproblem(macro_mesh, micro_mesh, macro_shape, box, i_macro_els, i_subdomains)

    subproblems = [
        make_sub(box, subdomains)
        for box, subdomains in zip(sub_boxes, sub_lists)
    ]
    return subproblems

@attrs.define
class Subproblems:
    macro_mesh: Mesh
    homogenized_els: np.array
    subproblems: List[SubMeshSubproblem]

    @staticmethod
    def create(macro_mesh, macro_els, micro_mesh, macro_shape, subdivision):
        subprobs = make_subproblems(macro_mesh, macro_els, micro_mesh, macro_shape, subdivision)
        return Subproblems(macro_mesh, macro_els, subprobs)

    def subdomains_average(self, subprob_avgs):
        subdomain_avg = np.zeros((self.n_subdomains, subprob_avgs[0].shape[1]))
        assert self.n_subdomains == sum((avg.shape[0] for avg in subprob_avgs))
        for subprob, avg in zip(self.subproblems, subprob_avgs):
            assert subdomain_avg.shape[1] == avg.shape[1]
            subdomain_avg[subprob.i_subdomains[:], :] = avg[:, :]
        return subdomain_avg

    @property
    def n_subdomains(self):
        return sum((len(s.i_subdomains) for s in self.subproblems))

    @staticmethod
    def equivalent_tensor_3d(loads, responses):
        # tensor pos. def.  <=> load @ response > 0
        # ... we possibly modify responses to satisfy
        unit_loads = loads / np.linalg.norm(loads, axis=1)[:, None]
        load_components = np.sum(responses * unit_loads, axis=1)
        responses_fixed = responses + (np.maximum(0, load_components) - load_components)[:, None] * unit_loads
        # from LS problem for 6 unknowns in Voigt notation: X, YY, ZZ, YZ, XZ, XY
        # the matrix has three blocks for Vx, Vy, Vz component of the responses
        # each block has different sparsity pattern
        n_loads = loads.shape[0]
        zeros = np.zeros(n_loads)
        ls_mat_vx = np.stack([loads[:, 0], zeros, zeros, zeros, loads[:, 2], loads[:, 1]], axis=1)
        rhs_vx = responses_fixed[:, 0]
        ls_mat_vy = np.stack([zeros, loads[:, 1], zeros, loads[:, 2], zeros, loads[:, 0]], axis=1)
        rhs_vy = responses_fixed[:, 1]
        ls_mat_vz = np.stack([zeros, zeros, loads[:, 2], loads[:, 1], loads[:, 0], zeros], axis=1)
        rhs_vz = responses_fixed[:, 2]
        ls_mat = np.concatenate([ls_mat_vx, ls_mat_vy, ls_mat_vz], axis=0)
        rhs = np.concatenate([rhs_vx, rhs_vy, rhs_vz], axis=0)
        assert ls_mat.shape == (3 * n_loads, 6)
        assert rhs.shape == (3 * n_loads,)
        result = np.linalg.lstsq(ls_mat, rhs, rcond=None)
        cond_tn_voigt, residuals, rank, singulars = result
        condition_number = singulars[0] / singulars[-1]
        if condition_number > 1e3:
            logging.warning(f"Badly conditioned inversion. Residual: {residuals}, max/min sing. : {condition_number}")
        return cond_tn_voigt

    #@report
    def equivalent_tensor_field(self, load_field,  response_field):
        load_field = np.array(load_field)
        response_field = np.array(response_field)
        assert load_field.shape == response_field.shape
        assert load_field.shape[1] == self.n_subdomains
        assert load_field.shape[2] == 3
        tensors = np.empty((self.n_subdomains, 6))
        for isub in range(self.n_subdomains):
            loads = load_field[:, isub, :]
            responses = response_field[:, isub, :]
            tensors[isub, :] = self.equivalent_tensor_3d(loads, responses)
        return tensors

def repr_aabb(aabb):
    return f"AABB({aabb.min()}, {aabb.max()})"


@attrs.define
class Subdomain:
    mesh: Mesh
    macro_el_idx: int
    el_indices: List[int]
    _weights : np.array = None

    @staticmethod
    def create(shape: MacroShapeBase, micro_mesh: Mesh, macro_mesh, i_el):
        """
        Select elements from the micro mesh interacting with a sphere
        approximating the macro element `id_el`.
        """
        macro_el = macro_mesh.elements[i_el]
        #center = macro_el.barycenter()
        #distances = np.linalg.norm(macro_el.vertices() - center[None,:], axis=1)
        #r = np.mean(distances)
        #logging.info(f"{center}, {distances}, {r}")
        #aabb = bih.AABB([center - r, center + r])
        aabb = shape.aabb(macro_el)
        candidates = micro_mesh.candidate_indices(aabb)
        assert candidates, f"MacroElShape AABB: {i_el} : {aabb} out of subproblem mesh AABB: {repr_aabb(micro_mesh.bih.aabb())}"
        subdomain_indices = [ie for ie in candidates
                         if shape.interact(macro_el, micro_mesh.elements[ie])]
        #logging.info(f"Subdomain candidates: {len(candidates)}, elements: {len(subdomain_indices)}")
        assert subdomain_indices, f"Empty subdomain {aabb}, {shape._center_radius(macro_el)} . {[micro_mesh.elements[ie].barycenter() for ie in candidates]}"
        # TODO: we should also check, that subdomain is covered by micro elements, otherwise, e.g.
        # porosity and conductivity would be wrong
        return Subdomain(micro_mesh, i_el, subdomain_indices)

    @property
    @report
    def weights(self):
        if self._weights is None:
            volumes = self.mesh.el_volumes[self.el_indices]
            self._weights = volumes / np.sum(volumes)
        return self._weights

    @report
    def average(self, element_vec_data: np.array):
        """
        :param element_vec_data: Values on all self.mesh elements in the same ordering (checked in get_statice
        :return:
        """
        if len(element_vec_data.shape) == 1:
            element_vec_data = element_vec_data[:, None]
        assert len(self.mesh.elements) == element_vec_data.shape[0]
        sub_domain_vector = element_vec_data[self.el_indices, :]
        avg = self.weights @ sub_domain_vector
        #if avg.shape[0] == 1:
        #    return avg[0]
        #else:
        return avg

# def micro_response(subdomains):
#     mesh = GmshIO("output/flow_fields.msh")
#     tree, el_ids = mesh_build_bih(mesh)
#     for x,y,z,r in subdomains:
#         center = np.array([x,y,z])
#         box = bih.AABB([center - r, center + r])
#         candidates = tree.find_box(box)
#         subdomain_els = [subdomain_interract for iel in candidates:


# def make_subdomains_old(cfg, subdomains_params):
#     fractures = []
#     i_pos = 0
#     mesh_file = "transport_micro.msh"
#     fine_micro_mesh(cfg.geometry, fractures, i_pos, mesh_file)
#     fine_mesh = Mesh(GmshIO(mesh_file))
#     subdomains = [Subdomain.for_sphere(fine_mesh, center, r)
#                   for center, r in subdomains_params]
#     return subdomains

@memoize
def subdomains_mesh(subdomains: List[Subdomain], output_name="fine_with_subdomains.msh"):
    """
    Fine mesh with duplicated elements for the subdomains,
    markeg by regions "sub_<isubdomain>"
    """
    mesh = subdomains[0].mesh
    output_mesh = copy.copy(mesh.gmsh_io)
    el_id = max(mesh.el_ids)
    for isub, sub in enumerate(subdomains):
        reg_name = f"sub_{isub}"
        tag = 1000 + isub
        output_mesh.physical[reg_name] = (tag, 3)
        els = {}
        for ie in sub.el_indices:
            type, tags, inodes = output_mesh.elements[mesh.el_ids[ie]]
            physical_tag, entity_tag = tags
            el_id = el_id + 1
            els[el_id] = (type, (tag, tag), inodes)
        output_mesh.elements.update(els)
    output_mesh.write(output_name)



class Homogenisation:
    """
    Support for (implicit) homogenisation. General algorithm:
    1. create a micro problem mesh
    2. for this micro mesh create subdomains (subsets of micro elements)
       could possibly be weighted by suitable kernel see Subdomain class
    3. create Homogenisation for the micro mesh and defined subdomains.
    4. Cut micromesh into ovelapping
    4. Run micro problem for some set of boundary or initial conditions,
       use Homogenisation methods to compute Subdomain averages of given scalar or vector quantities.
       Not stored but returned as numpy arrays.
    5. Use Homogenisation method to calculate ekvivalent tensor field for pair of vector fields.
    6. scalar quantities, meight be homogenised without running the micro problem, but we support them anyway.

    Possibility A:
    1. micro problem geometry
    2. microproblem mesh, define Subdomains
    3. micro mesh subproblems, as union of the Subdomains
    run all subpproblems for all loads => for every subdomain calculate equivalent tensor for all its available loads

    The fine mesh splitting cound be done by current Homogenisation. Need to have separate list of subdomains for every subproblem.
    Every subproblem only contributes to part of subdomains, need to deal with taht sparsity.

    Possiblility B:
    Subproblems formed at geometry level. more micro meshes. Has to construct single big BIH for all
    micro meshes.

    Could not be done within current Homogenisation. Needs common BIH for searching subdomains, possuble problems
    with inssuficient overlapping.
    Could we average accross the subproblems?

    AB common process:
    Given subdomains AABBs => compute subproblems AABBs
    Then apply subproblem AABBs to either geometry and get disconnected submeshes or apply to a fine mesh to get submeshes.
    """
    def __init__(self, subdomains: List[Subdomain]):
        self.subdomains = subdomains
        self.micro_mesh = self.subdomains[0].mesh
        # We assume common mesh for all subdomains, TODO: better design to have this automaticaly satisfied.
        assert all([sub.mesh is self.micro_mesh for sub in self.subdomains])










#
# def eval_conductivity_field(cfg_micro_conductivity, eval_points):
#     fine_mesh, max_radius, output_file):
#     c_min, c_max = cfg_micro_conductivity.range
#     x = bary_coords[:, 0]
#     y = bary_coords[:, 1]
#     z = bary_coords[:, 2]
#
#
#     horizontal_dist = y*y + (z*z*9)
#     vertical_dist = y*y*9 + z*z
#     r_sq = max_radius * max_radius
#     cond = c_min + c_max * np.exp(-horizontal_dist/r_sq) + c_max * np.exp(-vertical_dist/r_sq)
#
# def fine_conductivity_field(cfg, fine_mesh, output_file):
#     bary_coords = np.array([el.barycenter() for el in fine_mesh.elements])
#     conductivity = eval_conductivity_field((cfg.fine_conductivity, bary_coords)
#     return fine_mesh.write_fields(output_file, dict(conductivity=conductivity))

# TODO:
# - review main structure from macro transport
# - introduce new main test
# - new subtests
#