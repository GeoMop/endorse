from process_point_cloud import *


def main():
    # Read the mesh
    md = load_vtk_mesh(cfg.workdir / "L5_local_3.vtk")
    # Apply filters
    md = md.apply_filter('meshing_remove_duplicate_vertices')
    md = md.apply_filter('meshing_remove_duplicate_faces')
    md = md.apply_filter('meshing_merge_close_vertices',
                         threshold=pymeshlab.PureValue(0.005)) # threshold in percentage of domain diagonal
    md = md.apply_filter('meshing_remove_folded_faces')
    md = md.apply_filter('compute_normal_per_vertex',
                         weightmode='As defined by N. Max') #'Simple Average', 'By Area', 'By Angle'
    # md = md.apply_filter('generate_surface_reconstruction_screened_poisson',
    #                         depth=9,
    #                         samplespernode=1.5,
    #                         pointweight=4,
    #                         scale=1.1,
    #                         linearsolver='Iterative',
    #                         iters=8,
    #                         confidence=False,
    #                         outputpolygons=False,
    #                         density=False,
    #                         normals=False,
    #                         manifold=False,
    #                         clean=False
    #                         )
    md = md.apply_filter('generate_surface_reconstruction_vcg',
                         voxsize=pymeshlab.PureValue(0.08),
                         geodesic=2,
                         smoothnum=1,   # Laplace iteration to smooth borders
                         widenum=10,     # How many voxels is size of holes to fill.
                         simplification=True,
                         normalsmooth=3
                         )

    # md = md.apply_filter('meshing_snap_mismatched_borders',
    #                      edgedistratio=1,
    #                      unifyvertices=True)

    # mesh_idx = md.to_meshset().current_mesh_id()
    # other_mesh_idx = md.to_meshset(force_copy=True).current_mesh_id()
    # md = md.apply_filter('compute_mls_projection_apss',
    #     controlmesh=mesh_idx, proxymesh=other_mesh_idx,
    #     maxprojectioniters=1,
    #     filterscale=2,
    #     sphericalparameter=0.6
    # )

    # md = md.apply_filter('meshing_repair_non_manifold_edges',
    #                      method='Remove Faces')

    # for hole_edges in np.geomspace(40, 8*100, 4):
    #     md = md.apply_filter('meshing_close_holes',
    #                      maxholesize=int(hole_edges),
    #                      refinehole=True,
    #                      refineholeedgelen=pymeshlab.PureValue(0.05)
    #                      )  # number of hole edges, tunnel holes have about 16*100 edges
    #     md = md.apply_filter('meshing_remove_duplicate_vertices')
    #     md = md.apply_filter('meshing_remove_duplicate_faces')
    #     md = md.apply_filter('meshing_remove_folded_faces')
    #
    #     # Merge close vertices
    #     md = md.apply_filter('meshing_merge_close_vertices',
    #                          threshold=pymeshlab.PureValue(0.005)) # threshold in percentage of domain diagonal

    #    break
    md.save_to_vtk(cfg.workdir / "L5_local_3.vtk")

if __name__ == "__main__":
    main()