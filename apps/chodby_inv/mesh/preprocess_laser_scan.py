import os
import yaml
import numpy as np
from endorse import common

from bgem.gmsh import heal_mesh

script_dir = os.path.dirname(os.path.realpath(__file__))


def make_mesh(workdir, output_dir, cfg_file):
    conf_file = os.path.join(workdir, cfg_file)
    cfg = common.config.load_config(conf_file)
    cfg.output_dir = output_dir

    boundary_mesh_filename = os.path.join(cfg.output_dir, cfg.boundary_meshfile)

    # heal mesh
    boundary_mesh_healed_filename = os.path.join(cfg.output_dir, cfg.mesh_name + "_healed.msh2")
    if not os.path.exists(boundary_mesh_healed_filename):
        print("HEAL MESH")
        hm = heal_mesh.HealMesh.read_mesh(boundary_mesh_filename, node_tol=1e-4)
        hm.heal_mesh(gamma_tol=0.02)
        # hm.stats_to_yaml(os.path.join(output_dir, cfg.mesh_name + "_heal_stats.yaml"))
        hm.write(file_name=boundary_mesh_healed_filename)

    # the number of elements written by factory logger does not correspond to actual count
    # reader = gmsh_io.GmshIO(mesh_file.path)
    # print("N Elements: ", len(reader.elements))

    # print("Mesh file: ", mesh_file)


if __name__ == '__main__':
    # output_dir = None
    # len_argv = len(sys.argv)
    # assert len_argv > 1, "Specify input yaml file and output dir!"
    # if len_argv == 2:
    #     output_dir = os.path.abspath(sys.argv[1])
    output_dir = script_dir

    make_mesh(script_dir, output_dir, "./l5_mesh_config.yaml")

