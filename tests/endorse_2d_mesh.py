import os
from bgem.gmsh import gmsh,options, field
from endorse.mesh import mesh_tools
from endorse.common import EndorseCache, dotdict, workdir, File, call_flow
from endorse import flow123d_inputs_path


cfg = dotdict(
    flow_executable = [
        "/home/jb/workspace/flow123d/bin/fterm",
        "--no-term",
        "run"
    ],
    dim = [10, 30],
    #b_radius = 1,
    edz_radius = 1.5,
    edz_mesh_step = 1,
    boundary_mesh_step = 50,
    bh_mesh_step = 0.3,
    fr_mesh_step = 10
# fracture_mesh_step: 10
# # mesh step on outer boundary
# boundary_mesh_step: 50
# # mesh step on outer boundary of EDZ (just around boreholes right now)
# edz_mesh_step: 1
# # mesh step on inner boreholes
# boreholes_mesh_step: 0.3  # haxagon boreholes
# # boreholes_mesh_step: 0.3   # EDZ minimal resolution
# main_tunnel_mesh_step: 2

)

def make_geometry(factory: gmsh.GeometryOCC):
    dim_x, dim_z = cfg.dim
    domain = factory.rectangle([dim_x, dim_z], center=[dim_x/2, dim_z/2, 0])
    source = factory.line([0, 0, 0], [dim_x, 0, 0])
    outer = factory.line([0, dim_z, 0], [dim_x, dim_z, 0])
    fr = factory.line([2,2,0], [8, 20,0])

    box_fr, fractures_fr = factory.fragment(domain, fr)

    fractures_fr.set_region("fractures").mesh_step(cfg.fr_mesh_step)

    b_box_fr = box_fr.get_boundary().split_by_dimension()[1]
    src_boundary = b_box_fr.select_by_intersect(source.copy()).set_region(".source").mesh_step(cfg.bh_mesh_step)
    outer_boundary = b_box_fr.select_by_intersect(outer.copy()).set_region(".outer").mesh_step(cfg.boundary_mesh_step)
    box_fr = box_fr.set_region("box")
    bulk_geom = factory.group(box_fr, fractures_fr, src_boundary)

    return bulk_geom, src_boundary


def edz_field(factory, source):
    #b_cfg = cfg_geom.borehole
    #bx, by, bz = cfg_geom.box_dimensions
    #edz_radius = cfg_geom.edz_radius
    #center_line = factory.line([0, 0, 0], [b_cfg.length, 0, 0]).translate([0, 0, b_cfg.z_pos])

    n_sampling = int(cfg.dim[0] / 2)
    dist = field.distance(source, sampling=n_sampling)

    # INSIDE edz
    inner = field.geometric(dist, a=(0, cfg.edz_mesh_step * 0.9),
                            b=(cfg.edz_radius, cfg.edz_mesh_step))
    # OUT OF EDZ
    outer = field.polynomial(dist, a=(cfg.edz_radius, cfg.edz_mesh_step), b=(cfg.dim[1], cfg.boundary_mesh_step),
                             q=1.7)
    return field.maximum(inner, outer)


def edz_2d_mesh():
    factory = gmsh.GeometryOCC("endorse_2d", verbose=True)
    factory.get_logger().start()
    # factory = gmsh.GeometryOCC(mesh_name)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001
    bulk, source  = make_geometry(factory)

    factory.set_mesh_step_field(edz_field(factory, source))
    mesh_file = "endorse_2d.msh2"
    mesh_tools.edz_meshing(factory, [bulk], mesh_file)
    #factory.show()
    # del factory
    return File(mesh_file)



if __name__ == "__main__":
    EndorseCache.instance().expire_all()
    with workdir('sandbox/transport_2d'):
        mesh = edz_2d_mesh()


        params = dict(
            mesh_file=mesh.path,
            source_x0=0,
            source_x1=1,
            conc_flux_file=flow123d_inputs_path.joinpath("../../../tests/test_data/conc_flux_UOS_kg_y.csv")
        )
        template = os.path.join(flow123d_inputs_path, "transport_2d_tmpl.yaml")

        fo = call_flow(cfg, template, params)