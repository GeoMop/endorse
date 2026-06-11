import math
import os, sys
import subprocess
import pickle
import logging
import shutil
import time
from typing import *
from pathlib import Path
import yaml

import numpy as np
import pyvista as pv

from bgem.upscale import homogenization
from endorse import common

from endorse.common import dotdict, File, report, memoize
from endorse.mesh import fracture_tools
from endorse.mesh_class import load_mesh
from endorse.indicator import Extractor
from bgem.stochastic import Fracture, Population
# from endorse import hm_simulation

import zarr_fuse as zf
import xarray as xr
import zarr
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

from endorse.fullscale_transport import compute_fields, fracture_map, apply_fields, output_times
from endorse.macro_flow_model import macro_conductivity

import chodby_trans.job as job
import chodby_trans.input_data as input_data
from chodby_trans.mesh.create_mesh import make_fractures, make_mesh
from chodby_trans import ot_sa

import chodby_trans.exception_wrapper as exp


from functools import wraps
from loky import ProcessPoolExecutor  # NOT the stdlib one

NULL_RESULT = 0, np.array([])

homogenization_mesh_name = "trans_mesh_homogenization"

def run_in_subprocess(func):
    """Execute the function in a separate process (loky) with picklable args/return."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # fresh executor each call ⇒ fresh process & clean main thread
        with ProcessPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(func, *args, **kwargs)
            return fut.result()
    return wrapper


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


def population_parametrized(fr_families, parameters):
    new_fr_families = fr_families.copy()
    for fr_fam in new_fr_families:
        fr_fam['p_32'] = fr_fam['p_32'] * parameters['p_32_scale'] 
        fr_fam['power'] = fr_fam['power'] * parameters['p_32_scale'] 
        fr_fam['tr_a'] = fr_fam['tr_a'] * parameters['p_32_scale'] 
        fr_fam['tr_b'] = fr_fam['tr_b'] * parameters['p_32_scale'] 
    return new_fr_families


def update_dfn_params(cfg_fr, param_dict):
    if "population_template" in cfg_fr:
        # replace random parameters in fracture population config
        fr_population_fname = "fr_population"
        common.substitute_placeholders(file_in=job.input.dir_path / cfg_fr.population_template,
                                       file_out=fr_population_fname,
                                       params=param_dict)
        with Path(fr_population_fname).open("r", encoding="utf-8") as file:
            content = file.read()
            fr_dict = yaml.safe_load(content)
            cfg_fr.population = dotdict.create(fr_dict)
            logging.info(f"param_dict:\n{param_dict}")
            logging.info(f"DFN REPO:\n{cfg_fr.population}")

    elif len(cfg_fr.population) > 0:
        # randomize fracture populations parameters with Forsmark data
        dfn_cfg = population_parametrized(cfg_fr.population, param_dict)
    else:
        raise Exception("Fracture population not set, neither template file given.")


def update_mesh_cfg(cfg_mesh, level_dict):

    mcfg = common.apply_variant(cfg_mesh, level_dict.params)
    # assert mcfg == cfg_mesh
    mcfg.mesh_name = mcfg.mesh_name + f"_L{level_dict.id}"
    return mcfg


@memoize
# @run_in_subprocess
def create_mesh(cfg_mesh, fr_set, n_large):

    mesh_seed_seq = ot_sa.Seed.get_seedsequence(cfg_mesh.meshing_seed)
    mesh_seed = int(mesh_seed_seq.generate_state(1)[0])

    mesh_file = make_mesh(cfg_mesh, fr_set, mesh_seed)

    # full_mesh = load_mesh(mesh_file, heal_tol=1e-4)
    full_mesh = load_mesh(mesh_file, heal_tol=None)  # already healed

    el_to_ifr = None
    if "fractures" in cfg_mesh.geometry.include and fr_set is not None and len(fr_set)>0:
        # modifies the regions: fr_large, fr_small
        el_to_ifr = fracture_map(full_mesh, fr_set, n_large, dim=3)
        mesh_modified_filepath = Path(mesh_file.path).stem + "_modified.msh2"
        mesh_modified_file = full_mesh.write_fields(mesh_modified_filepath)
        full_mesh.file = mesh_modified_file

    return full_mesh, el_to_ifr

#@memoize
# @run_in_subprocess
def prepare_fine_input(workdir, cfg_mesh, cfg_trans, fr_set, n_large):
    # when running in subprocess, global variables are lost
    # therefore we set the workdir again
    #job.set_workdir(workdir)

    input_msh_filepath = Path(f"input_fields.msh")
    if input_msh_filepath.exists():
        return File(str(input_msh_filepath))

    full_mesh, el_to_ifr = create_mesh(cfg_mesh, fr_set, n_large)

    fields, est_velocity = compute_fields(cfg_mesh, cfg_trans, full_mesh,
                                                     apply_fields.bulk_fields_mockup_tunnel,
                                                     el_to_ifr, fr_set, dim=3)
    input_fields_file = full_mesh.write_fields(input_msh_filepath, fields)
    return input_fields_file


#@memoize
# @run_in_subprocess
def prepare_coarse_input(workdir, cfg_mesh, cfg_trans, fr_set, n_large):
    # when running in subprocess, global variables are lost
    # therefore we set the workdir again
    job.set_workdir(workdir)

    full_mesh, el_to_ifr = create_mesh(cfg_mesh, fr_set, n_large)
    return full_mesh.file
    # TODO: pass homogenization fields
    # mesh_modified = load_mesh(mesh_modified_file)
    # input_fields_file, est_velocity = compute_fields(cfg_mesh, cfg_trans, full_mesh, apply_fields.bulk_fields_mockup_tunnel,
    #                                                  el_to_ifr, fractures, dim=3)
    #
    # # input_fields_file = File("input_fields.msh2")
    # input_msh_filepath = Path(input_fields_file.path).with_suffix(".msh")
    # shutil.move(input_fields_file.path, input_msh_filepath)
    # input_msh = File(input_msh_filepath)
    # return input_msh


# @memoize
def transport_prepare_run(cfg_mesh):
    fracture_box = cfg_mesh.fractures.clip_box_ratio * np.array(cfg_mesh.geometry.box_dimensions)
    logging.info(f"box: {cfg_mesh.geometry.box_dimensions}")
    logging.info(f"fracture_box: {fracture_box}")
    logging.info(f"DFN REPO:\n{cfg_mesh.fractures.population}")

    dfn_seed_seq = ot_sa.Seed.get_seedsequence(cfg_mesh.fractures.dfn_seed)
    dfn_seed = dfn_seed_seq.generate_state(1)[0]

    fr_pop, fracture_set, n_large = make_fractures(cfg_mesh, dfn_seed)

    fr_stats = fracture_tools.fracture_set_stats(fracture_set)
    logging.info(f"N fracture set: {len(fracture_set)}")
    logging.info(f"Fracture stats:\n"
                 f"min: {fr_stats['min_radius']},\n"
                 f"max: {fr_stats['max_radius']},\n"
                 f"avg: {fr_stats['avg_radius']},\n"
                 f"med: {fr_stats['med_radius']}")

    return fr_pop, fracture_set, n_large


def transport_run(cfg, level_id, tags, param_dict):
    update_dfn_params(cfg.mesh.fractures, param_dict)
    fr_pop, fr_set, n_large = transport_prepare_run(cfg.mesh)

    with common.workdir("fine_trans", clean=False):
        logging.info(f"fine dir: {os.getcwd()}")
        rc, slice = transport_fine_run(cfg, fr_set, level_id, n_large, tags, param_dict)

    logging.info(f"sample dir: {os.getcwd()}")
    if level_id < len(cfg.mlmc.levels) - 1:
        with common.workdir("macro_trans", clean=False):
            logging.info(f"macro dir: {os.getcwd()}")
            # rc, slice = transport_coarse_run(cfg, fr_set, level_id + 1, n_large, tags, param_dict)
            # rc, slice = transport_homo_run(cfg, fr_set, level_id, n_large, tags, param_dict)
            transport_macro(cfg, fr_set, n_large, level_id, tags, param_dict)

    return rc, slice


def prepare_common_homogenization_mesh(cfg):
    # macro homogenization: homogenization mesh
    variant = "homo"
    macro_level = cfg.mlmc.levels[-1]
    cfg_mesh = update_mesh_cfg(cfg.mesh, macro_level)
    cfg_mesh.mesh_name = homogenization_mesh_name

    homo_mesh, _ = create_mesh(cfg_mesh, [], 0)

    homo_msh_filepath = Path(f"{homogenization_mesh_name}.msh")

    # msh2 -> msh
    shutil.move(homo_mesh.file.path, homo_msh_filepath)
    return File(str(homo_msh_filepath))


def interpolate_conductivity_tensor(cfg, source_mesh, conductivity_file, target_mesh):
    conductivity_source_mesh = load_mesh(conductivity_file, heal_tol=None)
    conductivity_source = conductivity_source_mesh.get_static_p0_values("conductivity_tn")
    # conductivity_source.shape: N elemenets X 9
    source_bulk = source_mesh.el_dim_slice(dim=3)
    target_bulk = target_mesh.el_dim_slice(dim=3)
    source_points = source_mesh.el_barycenters()[source_bulk]
    target_points = target_mesh.el_barycenters()[target_bulk]

    default_cond = float(cfg.transport_macroscale.default_conductivity)
    conductivity_target = np.empty((len(target_mesh.elements), 9))
    conductivity_target[:] = np.array([default_cond, 0, 0, 0, default_cond, 0, 0, 0, default_cond])

    interpolated_bulk = np.empty((len(target_points), conductivity_source.shape[1]))
    for i_comp in range(conductivity_source.shape[1]):
        values = conductivity_source[source_bulk, i_comp]
        # linear = griddata(source_points, values, target_points, method="linear")
        # nearest = griddata(source_points, values, target_points, method="nearest")
        # interpolated_bulk[:, i_comp] = np.where(np.isnan(linear), nearest, linear)
        interpolated_bulk[:, i_comp] = griddata(source_points, values, target_points, method="nearest")

    conductivity_target[target_bulk, :] = interpolated_bulk
    return conductivity_target


def transport_macro(cfg, fracture_set, n_large, level_id, tags, param_dict):
    # micro: fine mesh of buffer domain
    variant = "micro"
    level = cfg.mlmc.levels[level_id]
    cfg_mesh = update_mesh_cfg(cfg.mesh, level)
    cfg_mesh.geometry.box_dimensions = [v + 2 * level.buffer_width for v in cfg_mesh.geometry.box_dimensions]
    cfg_mesh.geometry.main_tunnel.length += 2 * level.buffer_width
    cfg_mesh.mesh_name += f"_{variant}"
    micro_mesh, el_to_ifr = create_mesh(cfg_mesh, fracture_set, n_large)

    # load common homogenization mesh
    homogenization_mesh = load_mesh(File(job.scratch.dir_path / f"{homogenization_mesh_name}.msh" ), heal_tol=None)  # already healed

    # homogenization onto common coarse mesh
    micro_fields, est_velocity = compute_fields(cfg_mesh, cfg.transport_microscale, micro_mesh,
                                                apply_fields.bulk_fields_mockup_tunnel,
                                                el_to_ifr, fracture_set, dim=3)
    macro_el_bulk = homogenization_mesh.el_dim_slice(dim=3)
    conductivity_file = macro_conductivity(cfg, micro_mesh, homogenization_mesh, macro_el_bulk, micro_fields)

    # macro: target coarse mesh
    variant = "macro"
    macro_level = cfg.mlmc.levels[level_id + 1]
    cfg_mesh = update_mesh_cfg(cfg.mesh, macro_level)
    coarse_fracture_set = [fr for fr in fracture_set if fr.r > macro_level.fr_min_limit]
    logging.info(f"N macro fracture set: {len(coarse_fracture_set)} / {len(fracture_set)}")
    cfg_mesh.mesh_name += f"_{variant}"
    macro_mesh, el_to_ifr = create_mesh(cfg_mesh, coarse_fracture_set, n_large)

    # macro: bulk conductivity tensor
    conductivity_macro = interpolate_conductivity_tensor(
        cfg, homogenization_mesh, conductivity_file, macro_mesh
    )

    # macro: other fields (fr conductivity, porosity)
    macro_cfg_trans = cfg.transport_macroscale
    macro_fields, est_velocity = compute_fields(cfg_mesh, macro_cfg_trans, macro_mesh,
                                                apply_fields.bulk_fields_mockup_tunnel,
                                                el_to_ifr, coarse_fracture_set, dim=3)
    # macro: add bulk conductivity tensor
    macro_fields["conductivity_tn"] = conductivity_macro

    input_fields_path = Path(f"input_fields.msh")
    input_fields_file = macro_mesh.write_fields(input_fields_path, macro_fields)

    # test output to VTK
    macro_mesh.write_fields_vtu(input_fields_path.with_suffix(".vtu"), macro_fields)


    # TODO run Flow123d on macro mesh
    res, fo = parametrized_run(cfg, large_model=None, input_fields_file=input_fields_file, tags=tags, param_dict=param_dict)
    time.sleep(0.5)  # give the FS a moment (tune as needed)
    values = process_results(cfg, fo)
    return res, values


# @memoize
def transport_fine_run(cfg, fracture_set, level_id, n_large, tags, param_dict):
    """ Fine full-scale transport model"""
    variant = "fine"
    level = cfg.mlmc.levels[level_id]
    cfg_mesh = update_mesh_cfg(cfg.mesh, level)
    cfg_mesh.mesh_name += f"_{variant}"

    input_msh_filepath = Path(f"input_fields.msh")
    if input_msh_filepath.exists():
        input_msh = File(str(input_msh_filepath))
    else:
        input_msh = prepare_fine_input(job.output.dir_path, cfg_mesh, cfg.transport_fullscale, fracture_set, n_large)

    # DEBUG mesh
    return NULL_RESULT

    res, fo = parametrized_run(cfg, large_model=None, input_fields_file=input_msh, tags=tags, param_dict=param_dict)
    time.sleep(0.5)  # give the FS a moment (tune as needed)
    values = process_results(cfg, fo)
    return res, values


# def transport_homo_run(cfg, fracture_set, level_id, n_large, tags, param_dict):
#     """ Fine full-scale flow model (including homogenization buffer)"""
#     variant = "fine_buffer"
#     level = cfg.mlmc.levels[level_id]
#     cfg_mesh = update_mesh_cfg(cfg.mesh, level)
#     cfg_mesh.geometry.box_dimensions = [v + 2 * level.buffer_width for v in cfg_mesh.geometry.box_dimensions]
#     cfg_mesh.geometry.main_tunnel.length += 2 * level.buffer_width
#     cfg_mesh.mesh_name += f"_{variant}"
#
#     input_msh_filepath = Path(f"input_fields_{variant}.msh")
#     if input_msh_filepath.exists():
#         input_msh = File(str(input_msh_filepath))
#     else:
#         input_msh = prepare_fine_input(job.output.dir_path, cfg_mesh, cfg.transport_fullscale, fracture_set, n_large)
#
#     # DEBUG mesh
#     return NULL_RESULT
#
#     res, fo = parametrized_run(cfg, large_model=None, input_fields_file=input_msh, tags=tags, param_dict=param_dict)
#     time.sleep(0.5)  # give the FS a moment (tune as needed)
#     values = process_results(cfg, fo)
#     return res, values
#
#
# def transport_coarse_run(cfg, fracture_set, level_id, n_large, tags, param_dict):
#     """ Fine full-scale flow model (including homogenization buffer)"""
#     variant = "coarse"
#     level = cfg.mlmc.levels[level_id]
#     cfg_mesh = update_mesh_cfg(cfg.mesh, level)
#     coarse_fracture_set = [fr for fr in fracture_set if fr.r > level.fr_min_limit]
#     logging.info(f"N coarse fracture set: {len(coarse_fracture_set)}")
#     cfg_mesh.mesh_name += f"_{variant}"
#
#     input_msh_filepath = Path(f"input_fields_{variant}.msh")
#     if input_msh_filepath.exists():
#         input_msh = File(str(input_msh_filepath))
#     else:
#         input_msh = prepare_coarse_input(job.output.dir_path, cfg_mesh, cfg.transport_fullscale, coarse_fracture_set, n_large)
#
#     # DEBUG mesh
#     return NULL_RESULT
#
#     # large_model = input_dir / cfg_fine.piezo_head_input_file
#     large_model = None
#     res, fo = parametrized_run(cfg, large_model, input_fields_file=input_msh, tags=tags, param_dict=param_dict)
#     time.sleep(0.5)  # give the FS a moment (tune as needed)
#     values = process_results(cfg, fo)
#     return res, values


@exp.rethrow_as(exp.Flow123dException, "Flow123d exception")
def call_flow_wrap(cfg_machine:'dotdict', file_in:File, params: Dict[str,str]) -> common.FlowOutput:
    """Wrapper to catch Exception and set return code"""
    fo = common.call_flow(cfg_machine, file_in, params)
    return fo


def parametrized_run(cfg, large_model, input_fields_file, tags, param_dict):
    stdout_path = Path('.') / 'transport_fullscale_stdout'
    stderr_path = Path('.') / 'transport_fullscale_stderr'
    if stdout_path.exists():
        import subprocess
        completed = subprocess.CompletedProcess([], 0, None, None)
        fo = common.FlowOutput(completed, stdout_path, stderr_path)
    else:
        cfg_fine = cfg.transport_fullscale
        params = cfg_fine.copy()
        times = output_times(cfg_fine)

        new_params = dict(
            mesh_file=input_fields_file,
            # piezo_head_input_file=large_model,
            input_fields_file=input_fields_file,
            dg_penalty=cfg_fine.dg_penalty,
            end_time_years=cfg_fine.end_time,
            trans_solver__a_tol=cfg_fine.trans_solver__a_tol,
            trans_solver__r_tol=cfg_fine.trans_solver__r_tol,
            output_times=[[t, 'y'] for t in times]
            # max_time_step = dt,
            # output_step = 10 * dt
        )
        params.update(new_params)
        params.update(set_source_term(cfg))
        params.update(param_dict)
        template = job.input.dir_path / cfg_fine.input_template
        fo = call_flow_wrap(cfg.machine_config, template, params)

    if fo.process.returncode == 0:
        res = fo.failed_convergence_reason
    else:
        res = fo.process.returncode

    return res, fo


def process_results(cfg: dotdict, fo: common.FlowOutput):
    data_schema_path = job.input.data_schema_yaml
    with data_schema_path.open("r", encoding="utf-8") as file:
        content = file.read()
        data_schema = yaml.safe_load(content)
        grid_size = data_schema[cfg.data_schema_key]["ATTRS"]["grid_step"]

    grid, values = get_indicator(cfg, fo, grid_size)

    return values

    # TODO: get grid, output times, values -> zarr_fuse
    # times
    # kwargs =  {"WORKDIR": str(input_data.zarr_store_path), "STORE_URL": str(input_data.zarr_store_path)}
    # data_schema = zf.schema.deserialize(input_data.data_schema_yaml) # read data scheme
    # root_node = zf.open_storage(data_schema, **kwargs)
    # current_node = root_node[cfg.data_schema_key]
    # grid_size = current_node.schema.ATTRS["grid_step"]

    # print(f"sample tags:{tags}")
    # grid, values = get_indicator(cfg, fo, [20, 20])

    # write_zarr_slice(store_path=str(input_data.zarr_store_path),
    #                  sample_idx=tags[0],
    #                  qmc_idx=tags[1],
    #                  block_idx=tags[2],
    #                  slice_array=values)

    # param_names = [p.name for p in cfg.sensitivity.parameters]
    #
    # current_node.update_dense(dict(
    #     iid=[tags[0]],  # coords
    #     qmc=[tags[1]],
    #     param_name=param_names,
    #     time=times,
    #     X=grid.x,
    #     Y=grid.y,
    #     Z=grid.z,
    #     block=[tags[2]],  #values
    #     param= parameters[np.newaxis, np.newaxis, np.newaxis, :], # coords: [ "iid", "qmc", "param_name"]
    #     conc=slice_array[np.newaxis, np.newaxis, np.newaxis, ...] # coords: [ "iid", "qmc", "block", "time", "X", "Y", "Z"]
    # ))

    # current_node.update_dense()


def write_zarr_slice(store_path: str,
                     sample_idx: int,
                     qmc_idx: int,
                     block_idx: int,
                     slice_array: np.ndarray) -> None:
    """
    Write a slice of data into an existing Zarr store for given sample and qmc indices.

    Parameters
    ----------
    store_path : str
        Path to the Zarr store.
    sample_idx : int
        Index along the 'sample' dimension where data will be written.
    qmc_idx : int
        Index along the 'qmc' dimension where data will be written.
    block_idx : int
        Index along the 'block' dimension where data will be written.
    slice_array : np.ndarray
        NumPy array of shape (time, X, Y, Z) matching the store dimensions.
    """
    # Open the existing Zarr store as an Xarray dataset
    ds = xr.open_zarr(store_path, consolidated=False)

    # Validate slice_array shape
    expected_shape = (ds.sizes['time'], ds.sizes['X'], ds.sizes['Y'], ds.sizes['Z'])
    if slice_array.shape != expected_shape:
        raise ValueError(f"slice_array must have shape {expected_shape}, got {slice_array.shape}")

    # Wrap slice_array into a DataArray with new sample and qmc coords
    da = xr.DataArray(
        slice_array[np.newaxis, np.newaxis, np.newaxis, ...],  # add sample, qmc and block dims
        dims=('sample', 'qmc', 'block', 'time', 'X', 'Y', 'Z'),
        coords={
            'sample': [sample_idx],
            'qmc': [qmc_idx],
            'block': [block_idx],
            'time': ds.coords['time'],
            'X': ds.coords['X'],
            'Y': ds.coords['Y'],
            'Z': ds.coords['Z'],
        }
    )

    # Write the slice by specifying the region to overwrite
    da.to_dataset(name='data').to_zarr(
        store_path,
        mode='a',
        region={
            'sample': slice(sample_idx, sample_idx + 1),
            'qmc': slice(qmc_idx, qmc_idx + 1),
            'block': slice(block_idx, block_idx + 1),
            'time': 'auto',
            'X': 'auto',
            'Y': 'auto',
            'Z': 'auto',
        }
    )

# @report
def indicators(pvd_in : File, attr_name, z_loc, grid, intp_ver: int, output: bool): # -> List[IndicatorFn]:
    #extractor = Extractor.from_point_data(attr_name, z_loc)
    # extractor = Extractor.from_cell_data(attr_name, z_loc)
    
    try:
        print(pvd_in.path)
        pvd_content = pv.get_reader(pvd_in.path)
    except Exception as e:
        with open(pvd_in.path, "r", encoding="utf-8") as f:
            sys.stdout.write(f.read())
            sys.stdout.flush()
        
        raise Exception('pv.get_reader error') from e

    times = np.asarray(pvd_content.time_values)
    print("pvd times: ", times)

    if output:
        slice_out_dir = Path(pvd_in.path).parent / "slice_intp"
        slice_out_dir.mkdir(parents=True, exist_ok=True)
        slice_vtu_files = [slice_out_dir / f"slice_intp_{ti:02d}.vtu" for ti in range(len(times))]
        write_pvd(times, slice_vtu_files, slice_out_dir.parent / "slice_intp.pvd")
    else:
        slice_vtu_files = [None for ti in range(len(times))]

    result = np.empty((len(times), grid.n_cells), dtype=np.float64)
    for ti, t in enumerate(times):
        pvd_content.set_active_time_point(ti)
        dataset = pvd_content.read()

        if intp_ver == 1:
            values = interpolate_v1(dataset, z_loc, attr_name)
        elif intp_ver == 2:
            values = interpolate_v2(dataset, grid, attr_name, slice_vtu_files[ti])
        elif intp_ver == 3:
            values = interpolate_v3(dataset, grid, attr_name, slice_vtu_files[ti])
        elif intp_ver == 4:
            values = interpolate_v4(dataset, grid, attr_name, slice_vtu_files[ti])

        # interpolated.save(f"slice_intp_{ti:02d}.vtu", binary=False)
        # values = interpolated.cell_data[attr_name]

        logging.info(f"values[t {ti}]: max {np.max(values)}")
        result[ti, :] = values

    logging.info(f"result: max {np.max(result)}")
    return result


def write_pvd(times, vtu_files, pvd_path):
    """
    times:     iterable of floats, length N
    vtu_files: iterable of filenames (relative or absolute), length N
    pvd_path:  output .pvd file path
    """

    # Make filenames relative to the .pvd location (best for portability)
    rel_files = []
    for f in vtu_files:
        try:
            rel_files.append(str(f.relative_to(pvd_path.parent)))
        except ValueError:
            rel_files.append(str(f))  # fallback to absolute if needed

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t, f in zip(times, rel_files):
        lines.append(f'    <DataSet timestep="{float(t)}" group="" part="0" file="{f}"/>')
    lines += [
        '  </Collection>',
        '</VTKFile>',
    ]
    pvd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpolate_v1(dataset, z_loc, attr_name):
    plane1 = dataset.slice(normal=[0, 0, 1], origin=[0, 0, z_loc[0]])
    plane2 = dataset.slice(normal=[0, 0, 1], origin=[0, 0, z_loc[1]])
    surface = plane1[0].merge(plane2[0])
    return surface.cell_data[attr_name]


def interpolate_v2(dataset, grid, attr_name, filepath):
    interpolated = grid.sample(dataset[0],
                               # tolerance=1e-5,
                               # tolerance=1e-7,
                               # tolerance=1e-9,
                               tolerance=1e-11,
                               # tolerance=1e-16,
                               # locator='cell', # 'cell' 'obb_tree' 'static_cell'
                               # locator='cell_tree',
                               # locator='static_cell',
                               snap_to_closest_point=True)
    interpolated = interpolated.point_data_to_cell_data()
    # debug grid output
    if filepath is not None:
        grid.save(filepath, binary=False)
    return interpolated.cell_data[attr_name]


def interpolate_v3(dataset, grid, attr_name, filepath):
    # refinement coefficient (n splits in each dim)
    rf = 4

    # grid.field_data["origins"] = origins
    # grid.field_data["spacing_xy"] = spacing
    # grid.field_data["dims_xy"] = dims
    cNx = grid.field_data["dims_xy"][0] - 1 # coarse
    cNy = grid.field_data["dims_xy"][1] - 1
    Nx = cNx * rf + 1 # fine
    Ny = cNy * rf + 1
    fine_spacing = grid.field_data["spacing_xy"] / rf
    fine_grid = make_grid(dims=[Nx, Ny, 1],
                          origins=grid.field_data["origins"],
                          spacing=fine_spacing)
    # debug grid output - fine grid indexing
    # fine_grid.save(f"slice_fine_grid.vtu", binary=False)

    # 2 x smallest ball around a single fine cell
    cell_radius = 2 * np.linalg.norm(fine_spacing[:2])
    # weight ratio of closest and furthest points
    decline_ratio = 0.8
    # sharpness of the Gaussian kernel
    sharpness = math.sqrt(-math.log(decline_ratio))
    # interpolation
    fine_interpolated = fine_grid.interpolate(dataset[0], radius=cell_radius, sharpness=sharpness)
    fine_interpolated = fine_interpolated.point_data_to_cell_data()
    # fine_interpolated.save(f"slice_intp_fv3_{ti:02d}.vtu", binary=False) # debug output
    fine_values = fine_interpolated.cell_data[attr_name]
    # fine_values = fine_interpolated.cell_data["cell_id"]  # debug ids

    # map fine grid to coarse grid - find max over coarse cell
    coarse_values = fine_values.reshape((2,cNx, rf, cNy, rf)).max(axis=(2, 4))
    grid.cell_data[attr_name] = coarse_values.reshape(-1)
    # debug grid output
    if filepath is not None:
        grid.save(filepath, binary=False)
    return grid.cell_data[attr_name]




def nearest_to_grid_xyz(points_xyz: np.ndarray,
                        values: np.ndarray,
                        grid_xyz: np.ndarray,
                        *,
                        max_dist: float,
                        fill_value: float = 0.0) -> np.ndarray:
    """
    Nearest-neighbor transfer from 3D points to 3D query points.

    points_xyz : (n,3)
    values     : (n,)
    grid_xyz   : (m,3) query locations (x,y,z)
    max_dist   : max 3D distance to accept neighbor (else fill_value)
    returns    : (m,) interpolated values
    """
    pts = np.asarray(points_xyz, dtype=float)
    vals = np.asarray(values)
    q = np.asarray(grid_xyz, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must be (n,3), got {pts.shape}")
    if vals.ndim != 1 or vals.shape[0] != pts.shape[0]:
        raise ValueError(f"values must be (n,), got {vals.shape} for n={pts.shape[0]}")
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"grid_xyz must be (m,3), got {q.shape}")
    if max_dist < 0:
        raise ValueError("max_dist must be non-negative")

    m = q.shape[0]
    if pts.shape[0] == 0 or m == 0:
        return np.full((m,), fill_value, dtype=float)

    tree = cKDTree(pts)
    dist, idx = tree.query(q, k=1, distance_upper_bound=max_dist)

    n = pts.shape[0]
    ok = (idx < n) & np.isfinite(dist)

    out = np.full((m,), fill_value, dtype=float)
    out[ok] = vals[idx[ok]]
    return out


def interpolate_v4(dataset, grid, attr_name, filepath):
    # refinement coefficient (n splits in each dim)
    rf = 4

    # grid.field_data["origins"] = origins
    # grid.field_data["spacing_xy"] = spacing
    # grid.field_data["dims_xy"] = dims
    cNx = grid.field_data["dims_xy"][0] - 1 # coarse
    cNy = grid.field_data["dims_xy"][1] - 1
    Nx = cNx * rf + 1 # fine
    Ny = cNy * rf + 1
    fine_spacing = grid.field_data["spacing_xy"] / rf
    fine_grid = make_grid(dims=[Nx, Ny, 1],
                          origins=grid.field_data["origins"],
                          spacing=fine_spacing)
    # debug grid output - fine grid indexing
    # fine_grid.save(f"slice_fine_grid.vtu", binary=False)

    # conc_values = dataset[0].point_data[attr_name]
    # fvalues = nearest_to_grid_xyz(points_xyz=dataset[0].points, values=conc_values,
    #                               grid_xyz=fine_grid.points, max_dist=1.0)

    # If the .read() returns a MultiBlock, merge blocks first
    if isinstance(dataset, pv.MultiBlock):
        dataset = dataset.combine()
    conc_values = np.asarray(dataset.cell_data[attr_name])
    source_points = np.asarray(dataset.cell_centers().points)
    grid_points = np.asarray(fine_grid.cell_centers().points)
    fvalues = nearest_to_grid_xyz(points_xyz=source_points, values=conc_values,
                                  grid_xyz=grid_points, max_dist=1.0)

    fine_grid.cell_data[attr_name] = fvalues
    # fine_grid.point_data[attr_name] = fvalues
    # fine_grid = fine_grid.point_data_to_cell_data()

    # fine_grid.save(f"slice_intp_fv4_{ti:02d}.vtu", binary=False) # debug output
    # fine_values = fine_grid.cell_data[attr_name]
    # fine_values = fine_grid.cell_data["cell_id"]  # debug ids

    # map fine grid to coarse grid - find max over coarse cell
    coarse_values = fvalues.reshape((2,cNx, rf, cNy, rf)).max(axis=(2, 4))
    grid.cell_data[attr_name] = coarse_values.reshape(-1)
    # debug grid output
    if filepath is not None:
        grid.save(filepath, binary=False)
    return grid.cell_data[attr_name]


def plane_cell_ids(grid, plane_id):
    # npx = nx+1, npy = ny+1
    npx = grid.dimensions[0]
    npy = grid.dimensions[1]
    start = plane_id * npx * npy
    return np.arange(start, start + npx * npy)

def make_grid(dims, origins, spacing):
    # Create 2D grid in the XY plane (z=0)
    grid1 = pv.ImageData(dimensions=dims,
                         origin=origins[0],
                         spacing=spacing  # spacing in z doesn't matter for 2D
                         )
    grid2 = pv.ImageData(dimensions=dims,
                         origin=origins[1],
                         spacing=spacing  # spacing in z doesn't matter for 2D
                         )
    # debug grid output - cell indexing
    # grid1["cell_id"] = np.arange(grid1.n_cells)
    # grid2["cell_id"] = np.arange(grid2.n_cells) + grid1.n_cells
    grid = grid1.merge(grid2)

    grid.field_data["origins"] = origins
    grid.field_data["spacing_xy"] = spacing
    grid.field_data["dims_xy"] = dims
    return grid


def create_structured_grid(cfg_geom: dotdict, z_cuts, grid_step):
    # Define grid resolution
    nx, ny, nz = grid_step  # number of elements in x and y
    bx, by, bz = cfg_geom.box_dimensions
    tol = 1e-6
    bx, by = bx-tol, by-tol

    shift = np.array(cfg_geom.box_center)
    origin = shift + [-bx/2, -by/2, 0]
    # origin_x, origin_y = -bx/2, -by/2

    # Compute spacing
    dx = bx / nx
    dy = by / ny

    dims = [nx+1, ny+1, 1]
    origins = np.array([origin + [0,0,z_cuts[0]],
                        origin + [0,0,z_cuts[1]]])
    spacing = np.array([dx, dy, 1])

    grid = make_grid(dims, origins, spacing)
    grid.save(f"slice_grid.vtu", binary=False)
    return grid
# def quantity_times(o_times):
#     """
#     Denser times set.
#     """
#     times = []
#     for a, b in zip(o_times[:-1], o_times[1:]):
#         step = max((b - a) / 5.0 ,  1000)
#         times.extend(np.arange(a, b, step))
#     return times


def z_cuts_fn(cfg_geom: dotdict):
    z_dim = 0.9 * 0.5 * cfg_geom.box_dimensions[2]
    # z_shift = cfg.geometry.borehole.z_pos
    # z_shift = cfg.geometry.main_tunnel.center[2] - cfg.geometry.main_tunnel.height/2 - cfg.geometry.storage_borehole.length/2
    z_shift = 0
    return (z_shift - z_dim, z_shift + z_dim)

# @report
def get_indicator(cfg, fo, grid_step):
    cfg_fine = cfg.transport_fullscale
    z_cuts = z_cuts_fn(cfg.geometry)
    grid = create_structured_grid(cfg.geometry, z_cuts, grid_step)
    values = indicators(fo.solute.spatial_file, f"{cfg_fine.conc_name}_conc", z_cuts, grid,
                        intp_ver=4, output=not cfg.ot_sensitivity.clean_sample_dir)
    print(np.shape(values))
    n_times = np.shape(values)[0]
    block = values.reshape(n_times, *grid_step)
    print(np.shape(block))
    return grid, block
#     plots.plot_indicators(inds)
#     #itime = IndicatorFn.common_max_time(inds)  # not splined version, need slice data
#     #plots.plot_slices(fo.solute.spatial_file, f"{cfg_fine.conc_name}_conc", z_cuts, [itime-1, itime, itime+1])
#     q_times = quantity_times(output_times(cfg_fine))
#
#     ind_value = [ind.time_max()[1] for ind in inds]
#     ind_time = [ind.time_max()[0] for ind in inds]
#     ind_series = np.array([ind.spline(q_times) for ind in inds])
#     return np.concatenate((ind_time, ind_value, ind_series.flatten()))


def set_source_term(cfg):
    # borehole radius
    cfg_fine = cfg.transport_fullscale
    cfg_src = cfg_fine.sources_params
    cfg_geom = cfg.mesh.geometry
    cfg_bh = cfg_geom.storage_borehole

    dsb_idx = cfg_geom.damaged_storage_borehole

    source_params = dict(
        # UOS surface: S = pi * du * hu [m2]
        sources_uos_surface=np.pi * cfg_src.diameter * cfg_src.length,
        # container region volume: V = pi * dc^2/4 * hc [m3]
        sources_container_vol=np.pi * 0.25 * cfg_bh.diameter ** 2 * (cfg_bh.length - cfg_bh.plug),
        sources_buffer_thickness=cfg_src.buffer_thickness,
        conc_flux_file= job.input.dir_path / cfg_fine.conc_flux_file,

        storage_regions = [f"storage_{i}" for i in range(cfg_geom.n_storage_boreholes) if i != dsb_idx],
        plug_region = f"plug_{dsb_idx}",
        container_region = f"container_{dsb_idx}",
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

def main():

    # common.EndorseCache.instance().expire_all()

    conf_file = input_data.transport_config
    cfg = common.config.load_config(str(conf_file))

    seed = 101
    with common.workdir(str(input_data.work_dir), clean=False):
        transport_run(cfg, seed)


if __name__ == '__main__':
    main()
