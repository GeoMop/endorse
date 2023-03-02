import os
import numpy as np

import aux_functions

from bgem.gmsh import gmsh
# from bgem.gmsh import gmsh_io
from bgem.gmsh import options as gmsh_options
from bgem.gmsh import heal_mesh


class MeshFactory:

    @staticmethod
    def prepare_mesh(geom_config, output_dir, cut_tunnel):
        mesh_name = geom_config["mesh_name"]
        if cut_tunnel:
            mesh_name = mesh_name + "_cut"
        mesh_file = os.path.join(output_dir, mesh_name + ".msh")
        mesh_healed = os.path.join(output_dir, mesh_name + "_healed.msh")

        # if os.path.isfile(mesh_healed):
        #     shutil.copyfile(os.path.join(config_dict["common_files_dir"], mesh_healed), mesh_healed)
        #     return mesh_healed

        # if not os.path.isfile(mesh_healed):
        #     if not os.path.isfile(mesh_file):

        MeshFactory.make_mesh_A(geom_config, mesh_name, mesh_file, cut_tunnel=cut_tunnel)
        # self.make_mesh_B(config_dict, mesh_name, mesh_file, cut_tunnel=cut_tunnel)
        hm = heal_mesh.HealMesh.read_mesh(mesh_file, node_tol=1e-4)
        hm.heal_mesh(gamma_tol=0.01)
        hm.stats_to_yaml(os.path.join(output_dir, mesh_name + "_heal_stats.yaml"))
        hm.write(mesh_healed)
        print(hm.healed_mesh_name, mesh_healed, flush=True)
        # assert hm.healed_mesh_name == mesh_healed

    @staticmethod
    def make_mesh_A(geom_config, mesh_name, mesh_file, cut_tunnel):
        tunnel_mesh_step = geom_config['tunnel_mesh_step']
        max_elem_size = geom_config["max_elem_size"]
        dimensions = geom_config["box_dimensions"]
        tunnel_dims = np.array([geom_config["tunnel_dimX"], geom_config["tunnel_dimY"]])/2
        tunnel_center = geom_config["tunnel_center"]

        print("load gmsh api")
        factory = gmsh.GeometryOCC(mesh_name, verbose=True)
        gmsh_logger = factory.get_logger()
        gmsh_logger.start()
        gopt = gmsh_options.Geometry()
        gopt.Tolerance = 0.0001
        gopt.ToleranceBoolean = 0.001
        # gopt.MatchMeshTolerance = 1e-1
        gopt.OCCFixSmallEdges = True
        gopt.OCCFixSmallFaces = True

        # Main box
        box = factory.rectangle(dimensions).set_region("box")
        side = factory.line([-dimensions[0]/2, 0, 0], [dimensions[0]/2, 0, 0])
        sides = dict(
            bottom=side.copy().translate([0, -dimensions[1] / 2, 0]),
            top   =side.copy().translate([0, +dimensions[1] / 2, 0]),
            left  =side.copy().translate([0, +dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2),
            right =side.copy().translate([0, -dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2)
        )

        tunnel_disc = factory.disc(tunnel_center, *tunnel_dims)
        tunnel_select = tunnel_disc.copy()
        tunnel_ngh = factory.disc(tunnel_center, *(5*tunnel_dims))

        print("cutting and fragmenting...")
        box_drilled = box.cut(tunnel_disc)
        ngh_drilled = tunnel_ngh.cut(tunnel_disc)
        box_fr, ngh_fr, tunnel_fr = factory.fragment(box_drilled, ngh_drilled, tunnel_disc)

        # b_ngh_fr = ngh_fr.get_boundary()
        # isec_ngh = ngh_fr.select_by_intersect(b_ngh_fr)
        # isec_ngh.modify_regions("tunnel_ngh").mesh_step(2*tunnel_mesh_step)
        # ngh_fr.modify_regions("tunnel_ngh").mesh_step(4 * tunnel_mesh_step)
        ngh_fr.modify_regions("box").mesh_step(6 * tunnel_mesh_step)
        box_all = [box_fr, ngh_fr]

        print("marking boundary regions...")
        b_box_fr = box_fr.get_boundary()
        for name, side_tool in sides.items():
            isec = b_box_fr.select_by_intersect(side_tool)
            box_all.append(isec.modify_regions("." + name))

        if cut_tunnel:
            b_tunnel_select = tunnel_select.get_boundary()
            b_tunnel = b_box_fr.select_by_intersect(b_tunnel_select)
            b_tunnel.modify_regions(".tunnel").mesh_step(tunnel_mesh_step)
            box_all.extend([b_tunnel])
        else:
            tunnel = tunnel_fr.select_by_intersect(tunnel_select)
            tunnel.set_region("tunnel").mesh_step(tunnel_mesh_step)
            box_all.extend([tunnel])

        mesh_groups = [*box_all]

        print("meshing...")

        factory.keep_only(*mesh_groups)
        # factory.remove_duplicate_entities()
        factory.write_brep(mesh_file)

        min_el_size = tunnel_mesh_step
        max_el_size = max_elem_size
        # min_el_size = tunnel_mesh_step / 2
        # max_el_size = np.max(dimensions) / 10

        mesh = gmsh_options.Mesh()
        # mesh.Algorithm = gmsh_options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
        mesh.Algorithm = gmsh_options.Algorithm2d.Delaunay
        # mesh.Algorithm = gmsh_options.Algorithm2d.FrontalDelaunay
        # mesh.Algorithm3D = gmsh_options.Algorithm3d.Frontal
        # mesh.Algorithm3D = gmsh_options.Algorithm3d.Delaunay

        # mesh.Algorithm = gmsh_options.Algorithm2d.FrontalDelaunay
        # mesh.Algorithm3D = gmsh_options.Algorithm3d.HXT

        mesh.ToleranceInitialDelaunay = 0.01
        # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
        mesh.CharacteristicLengthFromPoints = True
        mesh.CharacteristicLengthFromCurvature = True
        mesh.CharacteristicLengthExtendFromBoundary = 2
        mesh.CharacteristicLengthMin = min_el_size
        mesh.CharacteristicLengthMax = max_el_size
        mesh.MinimumCirclePoints = 6
        mesh.MinimumCurvePoints = 2

        # factory.make_mesh(mesh_groups, dim=2)
        factory.make_mesh(mesh_groups)

        gmsh_log_msgs = gmsh_logger.get()
        gmsh_logger.stop()
        aux_functions.check_gmsh_log(gmsh_log_msgs)

        factory.write_mesh(format=gmsh.MeshFormat.msh2)
        os.rename(mesh_name + ".msh2", mesh_file)
        # factory.show()

    # @staticmethod
    # def make_mesh_B(self, config_dict, mesh_name, mesh_file, cut_tunnel):
    #     geom = config_dict["geometry"]
    #     tunnel_mesh_step = geom['tunnel_mesh_step']
    #     dimensions = geom["box_dimensions"]
    #     tunnel_dims = np.array([geom["tunnel_dimX"], geom["tunnel_dimY"]]) / 2
    #     tunnel_center = geom["tunnel_center"]

    #     print("load gmsh api")
    #     factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    #     gmsh_logger = factory.get_logger()
    #     gmsh_logger.start()
    #     gopt = gmsh_options.Geometry()
    #     gopt.Tolerance = 0.0001
    #     gopt.ToleranceBoolean = 0.001
    #     # gopt.MatchMeshTolerance = 1e-1
    #     gopt.OCCFixSmallEdges = True
    #     gopt.OCCFixSmallFaces = True

    #     # Main box
    #     box = factory.rectangle(dimensions).set_region("box")
    #     side = factory.line([-dimensions[0] / 2, 0, 0], [dimensions[0] / 2, 0, 0])
    #     sides = dict(
    #         # bottom=side.copy().translate([0, -dimensions[1] / 2, 0]),
    #         top=side.copy().translate([0, +dimensions[1] / 2, 0]),
    #         # left=side.copy().translate([0, +dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2),
    #         right=side.copy().translate([0, -dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2)
    #     )

    #     dir_factor = 0.01
    #     d = dir_factor * np.array(dimensions)
    #     dir_sides = dict(
    #         fix_bottom=factory.line([0, 0, 0], [d[0], 0, 0]).translate([-dimensions[0] / 2, -dimensions[1] / 2, 0]),
    #         bottom=factory.line([0, 0, 0], [dimensions[0] - d[0], 0, 0]).translate([-dimensions[0] / 2 + d[0], -dimensions[1] / 2, 0]),
    #         fix_left=factory.line([0, 0, 0], [0, d[1], 0]).translate([-dimensions[0] / 2, -dimensions[1] / 2, 0]),
    #         left = factory.line([0, 0, 0], [0, dimensions[1] - d[1], 0]).translate([-dimensions[0] / 2, -dimensions[1] / 2 + d[1], 0])
    #     )
    #     dir_sides_select = {key: value.copy() for key, value in dir_sides.items()}

    #     tunnel_disc = factory.disc(tunnel_center, *tunnel_dims)
    #     tunnel_select = tunnel_disc.copy()

    #     print("cutting and fragmenting...")
    #     box_drilled = box.cut(tunnel_disc)

    #     dlist = factory.fragment(box_drilled, tunnel_disc, *[v for v in dir_sides.values()])
    #     box_fr = dlist[0]
    #     tunnel_fr = dlist[1]

    #     print("marking boundary regions...")
    #     box_all = []

    #     b_box_fr = box_fr.get_boundary()
    #     for name, side_tool in sides.items():
    #         isec = b_box_fr.select_by_intersect(side_tool)
    #         box_all.append(isec.modify_regions("." + name))
    #     for name, side_tool in dir_sides_select.items():
    #         isec = b_box_fr.select_by_intersect(side_tool)
    #         box_all.append(isec.modify_regions("." + name))

    #     if cut_tunnel:
    #         b_tunnel_select = tunnel_select.get_boundary()
    #         b_tunnel = b_box_fr.select_by_intersect(b_tunnel_select)
    #         b_tunnel.modify_regions(".tunnel").mesh_step(tunnel_mesh_step)
    #         box_all.extend([box_fr, b_tunnel])
    #     else:
    #         tunnel = tunnel_fr.select_by_intersect(tunnel_select)
    #         tunnel.set_region("tunnel").mesh_step(tunnel_mesh_step)
    #         box_all.extend([box_fr, tunnel])

    #     mesh_groups = [*box_all]

    #     print("meshing...")

    #     factory.keep_only(*mesh_groups)
    #     # factory.remove_duplicate_entities()
    #     factory.write_brep()

    #     min_el_size = tunnel_mesh_step / 2
    #     max_el_size = np.max(dimensions) / 10

    #     mesh = gmsh_options.Mesh()
    #     # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
    #     # mesh.Algorithm = options.Algorithm2d.Delaunay
    #     # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay
    #     # mesh.Algorithm3D = options.Algorithm3d.Frontal
    #     # mesh.Algorithm3D = options.Algorithm3d.Delaunay

    #     # mesh.Algorithm = gmsh_options.Algorithm2d.FrontalDelaunay
    #     mesh.Algorithm3D = gmsh_options.Algorithm3d.HXT

    #     mesh.ToleranceInitialDelaunay = 0.01
    #     # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
    #     mesh.CharacteristicLengthFromPoints = True
    #     mesh.CharacteristicLengthFromCurvature = True
    #     mesh.CharacteristicLengthExtendFromBoundary = 2
    #     mesh.CharacteristicLengthMin = min_el_size
    #     mesh.CharacteristicLengthMax = max_el_size
    #     mesh.MinimumCirclePoints = 6
    #     mesh.MinimumCurvePoints = 2

    #     # factory.make_mesh(mesh_groups, dim=2)
    #     factory.make_mesh(mesh_groups)

    #     gmsh_log_msgs = gmsh_logger.get()
    #     gmsh_logger.stop()
    #     check_gmsh_log(gmsh_log_msgs)

    #     factory.write_mesh(format=gmsh.MeshFormat.msh2)
    #     os.rename(mesh_name + ".msh2", mesh_file)
    #     # factory.show()
