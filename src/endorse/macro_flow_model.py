import os
from typing import List

import numpy as np

from . import common
from .apply_fields import conductivity_mockup
from .common import dotdict, memoize, File, call_flow, workdir, report
from .mesh import container_position_mesh
from .homogenisation import MacroSphere, Subproblems
from .mesh_class import Mesh
from . import large_mesh_shift
from . import flow123d_inputs_path


def macro_transport(cfg:dotdict):
    work_dir = f"sandbox/run_macro_transport"
    macro_cfg = cfg.transport_macroscale
    large_model = File(macro_cfg.piezo_head_input_file)

    with common.workdir(work_dir, inputs=[large_model.path]):
        macro_mesh: Mesh = make_macro_mesh(cfg)
        # select elements with homogenized properties
        macro_model_el_indices = homogenized_elements(cfg.geometry, macro_mesh)
        conductivity_file = macro_conductivity(cfg, macro_mesh, macro_model_el_indices)
        # TODO:  run macro model

        template = flow123d_inputs_path.joinpath(macro_cfg.input_template)
        params = dict(
            mesh_file = macro_mesh.file.path,
            input_fields_file = conductivity_file.path,
            piezo_head_input_file = os.path.basename(large_model.path)

        )
        macro_model = common.call_flow(cfg.machine_config, template, params)


def fine_macro_transport(cfg):
    cfg_fine = cfg.transport_fine
    micro_mesh = make_micro_mesh(cfg)
    template = flow123d_inputs_path.joinpath(cfg_fine.input_template)
    conductivity_file = conductivity_mockup(cfg.geometry, cfg_fine.bulk_field_params, micro_mesh)
    large_model = large_model = File(cfg_fine.piezo_head_input_file)
    params = dict(
        mesh_file=micro_mesh.file.path,
        piezo_head_input_file=os.path.basename(large_model.path),
        input_fields_file = conductivity_file.path
    )
    with common.workdir("sandbox/fine_flow", inputs=[micro_mesh.file.path, large_model.path, conductivity_file.path], clean=False):
        common.call_flow(cfg.machine_config, template, params)

@memoize
def mesh_shift(mesh_file_in: File, shift) -> File:
    base, ext = os.path.splitext(mesh_file_in.path)
    new_file_name = f"{base}_local.msh2"
    return large_mesh_shift.shift(mesh_file_in.path, np.array(shift), new_file_name)
    return File(new_file_name)

@memoize
def homogenized_elements(cfg_geometry:dotdict, macro_mesh: Mesh):
    """
    Select elements of the macro model that would use homogenized properties (conductiity).
    Return list of selected element indices.
    """
    z_borehole_shift = cfg_geometry.borehole.z_pos
    el_indices = []
    for ie, e in enumerate(macro_mesh.elements):
        x,y,z = e.barycenter()
        z = (z-z_borehole_shift)
        if z*z + y*y < (cfg_geometry.edz_radius) ** 2:
            el_indices.append(ie)
    return el_indices


@memoize
def make_macro_mesh(cfg):
    macro_step = cfg.transport_macroscale.mesh_step
    mesh_file = container_position_mesh.macro_mesh(cfg.geometry, macro_step)
    return Mesh.load_mesh(mesh_file)

@memoize
def make_micro_mesh(cfg):
    mesh_file = container_position_mesh.fine_mesh(cfg.geometry, cfg.transport_microscale.mesh_params)
    return Mesh.load_mesh(mesh_file)



#@memoize
def macro_conductivity(cfg:dotdict, macro_mesh: Mesh, homogenized_els: List[int]) -> File:
    """
    - merge default conductvity and homogenized conductivity tensors
    - convert from voigt to full 3x3 tensor
    - write to file
    TOSO: introduce Field class and split these three steps to general functions
    :type macro_mesh: object
    :param cfg:
    :param macro_mesh:
    :param micro_model_els:
    :param conductivity_tensors:
    :return:
    """

    micro_mesh: Mesh = make_micro_mesh(cfg)
    macro_shape = MacroSphere(rel_radius=1)
    subdivision = np.array([2, 2, 1])
    #subprobs = make_subproblems(macro_mesh, micro_mesh, macro_shape, subdivision)

    #subdomains = [Subdomain.for_element(micro_mesh, macro_mesh.elements[ie]) for ie in homogenized_els]
    homo = Subproblems.create(macro_mesh, homogenized_els, micro_mesh, macro_shape, subdivision)
    # debugging output of the subdomains
    #subdomains_mesh(subdomains)

    cfg_micro = cfg.transport_microscale
    gen_load_responses = (micro_load_response(cfg, homo, il, load) for il, load in enumerate(cfg_micro.pressure_loads))
    loads, responses = zip(*gen_load_responses)
    conductivity_tensors = homo.equivalent_tensor_field(loads, responses)

    # Heterogeneous conductiity tensor stored in Voigt notation.
    dflt_cond = cfg.transport_macroscale.default_conductivity
    # TODO: possibly get just comutational elements (given computation regions)
    n_elements = len(macro_mesh.elements)
    conductivity = np.empty((n_elements, 9))
    conductivity[:, :] = np.array([dflt_cond, 0, 0, 0, dflt_cond, 0, 0, 0, dflt_cond])
    voigt_indices = [0, 5, 4, 5, 1, 3, 4, 3, 2]
    conductivity[homogenized_els[:], :] = conductivity_tensors[:, voigt_indices[:]]

    input_fields_file = cfg.transport_macroscale.input_fields_file
    macro_mesh.write_fields(input_fields_file,
                            dict(conductivity_tn=conductivity))
    return File(input_fields_file)


@memoize
def micro_load_response(cfg, subprobs:Subproblems, i_load, load):
    """
    1. run micro model(s)
    2. average over subdomains
    Return (n_subdomains, (load_avg, respons_avg))
    TODO: finish param to Flow, test
    """
    cfg_micro = cfg.transport_microscale
    cfg_flow = cfg.machine_config
    fine_conductivity_params=cfg

    def micro(iprob, subprob):
        tag = f"load_{i_load}_{iprob}"
        return conductivity_micro_problem(cfg, tag, subprob, fine_conductivity_params, load)

    subdomain_response, subdomain_load = zip(*[micro(iprob, subprob) for iprob, subprob in enumerate(subprobs.subproblems)])

    return subprobs.subdomains_average(subdomain_response), subprobs.subdomains_average(subdomain_load)

@report
@memoize
def subproblem_input(subproblem, cfg_geom, conductivity_params):
    mesh = subproblem.submesh
    return conductivity_mockup(cfg_geom, conductivity_params, mesh)

@report
def micro_postprocess(cfg_micro, subproblem, micro_model):
    def load():
        return Mesh.load_mesh(micro_model.hydro.spatial_file)
    output_mesh: Mesh = report(load)()
    response_field = cfg_micro.response_field_p0
    def response():
        return output_mesh.get_static_p0_values(response_field)
    response_el_values = report(response)()
    def load_f():
        return get_load_data(cfg_micro, output_mesh, response_el_values)
    load_el_values = report(load_f)()
    def avg_r():
        return subproblem.average(response_el_values)
    def avg_l():
        return subproblem.average(load_el_values)
    return report(avg_r)(), report(avg_l)()


def conductivity_micro_problem(cfg, tag, subproblem, fine_conductivity_params, load):
    cfg_micro = cfg.transport_microscale
    with workdir(f"load_{tag}", inputs=[]):
        fine_conductivity_file = subproblem_input(subproblem, cfg.geometry, cfg_micro.bulk_field_params)
        params = dict(
            mesh_file=fine_conductivity_file.path,
            pressure_grad=str(load),
            fine_conductivity=fine_conductivity_file.path
        )
        template = flow123d_inputs_path.joinpath(cfg_micro.input_template)
        micro_model = call_flow(cfg.machine_config, template, params)
        return micro_postprocess(cfg_micro, subproblem, micro_model)



def grad_for_p1(loc_el_mat, node_values):
    """
    For given list of nodal coordinates and given nodal values (P1 dofs)
    compute gradient vector of the P1 field.
    TODO: !! TEST YET
    """
    ref_grads = np.array([[1,0,0], [0,1,0], [0,0,1], [-1.0/3, -1.0/3, -1.0/3]])
    return loc_el_mat @ ref_grads.T @ np.atleast_1d(node_values)


@memoize
def get_load_data(cfg_micro:dotdict, output_mesh: Mesh, response_el_values:np.array):
    load_field = cfg_micro.get("load_grad_field_p0", "")
    if load_field:
        return output_mesh.get_static_p0_values(load_field)

    load_field = cfg_micro.get("load_field_p1", "")
    if load_field:
        load_data_p1 = output_mesh.get_static_p1_values(load_field)
        el_loads = [
            grad_for_p1(el.loc_mat(), load_data_p1[el.node_indices])
            for el in output_mesh.elements
        ]
        return np.array(el_loads)


    load_field = cfg_micro.get("load_field_indirect_conductivity", "")
    if load_field:
        conductivity = output_mesh.get_static_p0_values(load_field)
        return response_el_values / conductivity


