from typing import *
import os
from bgem.gmsh import gmsh
from endorse.common import File, dotdict
from endorse.mesh import mesh_tools


"""
macro mesh - given part of borehole + larger neighbourhood , about 20m, no borhole representation
fine mesh - same outer geometry, but with cut borehole, fine around EDZ
homogenization mesh - subset of fine mesh, same geometry, but cut by given cylinder
           can use single mesh but we need to solve translation of underleing field (conductivity)
           here it is fine as it is calculated, should be calculated for every homogenisation mesh 
           or interpolated from a single field
"""


def macro_outer_box(cfg_geom, factory):
    b_cfg = cfg_geom.borehole
    x_size = 0.9 * b_cfg.length
    x_shift = x_size / 2
    yz_size = 5 * cfg_geom.edz_radius
    return factory.box([x_size, yz_size, yz_size]).translate([x_shift, 0, b_cfg.z_pos])

def macro_mesh(cfg_geom:dotdict, macro_mesh_step:float):
    """
    Make mesh of given container position `i_pos`.
    BREP and mesh writen to given mesh_file derived files.
    The EDZ transport coordinate system is preserved.
        X - in direction of storage boreholes
        Y - perpendicular horizontal
        Z - vertical
        origin: center of the center borehole on the interface with lateral tunnel
    """
    base = "macro_borehole"
    factory = gmsh.GeometryOCC(base, verbose=True)
    box = macro_outer_box(cfg_geom, factory)
    box.mesh_step(macro_mesh_step)
    mesh_file = base + ".msh"
    mesh_tools.edz_meshing(factory, [box], mesh_file)
    del factory
    return File(mesh_file)

def fine_mesh(cfg_geom:dotdict, cfg_mesh:dotdict):
    """
    macro mesh with cut borehole and refined around
    """
    b_cfg = cfg_geom.borehole

    base = "fine_borehole"
    mesh_file = base + ".msh"
    factory = gmsh.GeometryOCC(base, verbose=True)
    box = macro_outer_box(cfg_geom, factory)
    bh = factory.cylinder(b_cfg.radius, axis=[b_cfg.length, 0, 0]).translate([0, 0, b_cfg.z_pos])

    box_cut = box.copy().cut(bh.copy())
    domain = box.copy().fragment(bh.copy())
    outer = domain.select_by_intersect(box_cut).set_region("outer")
    borehole = domain.select_by_intersect(bh).set_region("borehole")



    # TODO: mesh EDZ cylinder as well and implement region selection after meshing
    #edz = factory.cylinder(cfg.borehole_radius, axis=[cfg.borehole_length, 0, 0])

    #factory.get_logger().start()
    #factory = gmsh.GeometryOCC(mesh_name)
    #gopt = options.Geometry()
    #gopt.Tolerance = 0.0001
    #gopt.ToleranceBoolean = 0.001
    factory.set_mesh_step_field(mesh_tools.edz_refinement_field(factory, cfg_geom, cfg_mesh))
    #factory.get_logger().stop()
    mesh_tools.edz_meshing(factory, [outer, borehole], mesh_file)
    # factory.show()
    del factory
    return File(mesh_file)

#def micro_mesh()
#Interval = Tuple[float, float]
def borehole_single_mesh(cfg:dotdict, x_size:float, yz_size_float, h_min:float, mesh_file: str = "borehole_mesh.msh2"):
    # Radius of the homogenization kernel, approximately macro mesh step
    base, ext = os.path.splitext(os.path.basename(mesh_file))
    factory = gmsh.GeometryOCC(base, verbose=True)
    container_period = mesh_tools.container_period(cfg)
    box_shift = x_size / 2  #+ mesh_tools.container_x_pos(cfg, i_pos)

    # TODO: homogenization x_size could be unrelated to the container size
    x_size = container_period + 2 * macro_mesh_step
    yz_size = 3 * (cfg.edz_radius + macro_mesh_step)
    box = factory.box([x_size, yz_size, yz_size]).translate([box_shift, 0, b_cfg.z_pos])
    bh = factory.cylinder(b_cfg.radius, axis=[b_cfg.length, 0, 0]).translate([0, 0, b_cfg.z_pos])
    box_cut = box.copy().cut(bh.copy())
    domain = box.copy().fragment(bh.copy())
    outer = domain.select_by_intersect(box_cut).set_region("outer")
    borehole = domain.select_by_intersect(bh).set_region("borehole")

    # TODO: mesh EDZ cylinder as well and implement region selection after meshing
    # that would allow creating various homogenisation submeshes from a single fine problem mesh

    #edz = factory.cylinder(cfg.borehole_radius, axis=[cfg.borehole_length, 0, 0])

    factory.get_logger().start()
    factory.set_mesh_step_field(mesh_tools.edz_refinement_field(factory, None, cfg))
    factory.get_logger().stop()
    mesh_tools.edz_meshing(cfg, factory, [outer, borehole], mesh_file)
    # factory.show()
    del factory
    return mesh_file




#def fine_micro_mesh(cfg:dotdict, fractures:List['Fracture'], i_pos:int, mesh_file: str):

    # Radius of the homogenization kernel, approximately macro mesh step
