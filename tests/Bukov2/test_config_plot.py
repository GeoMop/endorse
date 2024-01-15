# from endorse.Bukov2 import optimize_configs as oc
#
#
# configs = [
#     ()
# ]
#     for (i,ind) in enumerate(hof):
# -        print(toolbox.evaluate(ind), [(i[0], list(_bhs_opt_config[i[0]][i[1]][i[2]].packers)) for i in ind])
# -        export_vtk_optim_set(ind, "boreholes_opt_cfg." + str(i) + ".vtk", plot=False)
# +        bh_pk_ids = [(i[0], list(_bhs_opt_config[i[0]][i[1]][i[2]].packers)) for i in ind]
# +        print(toolbox.evaluate(ind), bh_pk_ids)
# +        # export_vtk_optim_set(ind, "boreholes_opt_cfg." + str(i) + ".vtk", plot=False)
# oc.export_vtk_bh_chamber_set(bh_pk_ids, "boreholes_opt_cfg." + str(i) + ".vtk", plot=False)