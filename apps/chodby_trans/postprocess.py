from typing import *
import numpy as np
import pandas as pd
import xarray as xr
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors, cm as cm

import endorse

from pathlib import Path
from chodby_trans import ot_sa
from chodby_trans import job
from chodby_trans import plots

def ot_samples(cfg: dict, seed: int) -> ot_sa.InputDesign: # shape: (n_all_samples, n_params)
    cfg_sens = cfg.ot_sensitivity
    problem = ot_sa.SensitivityAnalysis.from_cfg(cfg_sens)
    return problem.sample(seed)


def sampling_data(cfg, seed):
    # eval ... index of individual model evaluations 
    # i_sample ... index of samples in the input design; usualy power of 2
    # saltelli ... index of evaluation within sample according to Saltelli schema
    
    input_design = ot_samples(cfg, seed)
    store_path = str(job.output.zarr_store_path)
    ds = xr.open_zarr(store_path, consolidated=True)
    print(ds)
    #print("i_sample", ds['i_sample'].values)
    #print("i_saltelli", ds['i_saltelli'].values)
    
    print(ds.dims)
    #print("IID", ds2['IID'].values)
    #print("QMC", ds2['QMC'].values)
    #plots.conc_tail_ecdf_plot(ds,
    #             sample_dim = {'i_saltelli', 'i_sample'}, space_dim={'X','Y','Z'}, time_dim='sim_time',
    #             output_pdf=job.output.plots/"conc_ecdf.pdf")
    ds2 = ds.rename_dims({"i_sample": "IID", "i_saltelli": "QMC"})
    ds2 = ds2.rename({"i_sample": "IID", "i_saltelli": "QMC"})

    # plots.raw_conc_plot(ds2,
    #               sample_dim = {'QMC', 'IID'}, space_dim={'X','Y','Z'}, time_dim='sim_time',
    #               output_pdf=job.output.plots/"conc_ecdf.pdf")
    # # print samples with bad concentrations; limit them
    # 1) mask for out-of-range values (outside [-0.1, 1.1])
    lo, hi = 1e-40, 1.1
    reduce_dims = ("sim_time", "X", "Y", "Z", "QMC")

    # 1) per-(iid,qmc) min/max over space-time
    conc = ds2["conc"]
    conc_min = conc.min(dim=reduce_dims, skipna=True)
    conc_max = conc.max(dim=reduce_dims, skipna=True)

    # 2) mask over (iid,qmc) only
    mask_bad = (conc_min < lo) | (conc_max > hi).compute()

   # print offending iid,qmc with their min/max
    df_bad = (
        xr.Dataset({"conc_min": conc_min, "conc_max": conc_max})
        .where(mask_bad)
        .to_dataframe()
        .reset_index()
        .dropna(subset=["conc_min", "conc_max"])
        [['IID', "conc_min", "conc_max"]]
    )
    #print(df_bad.to_string(index=False))
    print("Number of bad value samples:", len(df_bad))

    q99_space_max_time = conc.quantile(0.99, dim=("X", "Y", "Z"), skipna=True).max(dim="sim_time")

    # 3) limit the dataset to those values -> ds3
    #    (keep only 'conc' filtered to the mask; drop coords where all values became NaN)
    ds3 = ds2.assign(log10_conc=np.log10(ds2["conc"].clip(min=lo, max=hi)))
    var_name = "log10_conc"

    #ds3 = ds2.assign(conc=ds2["conc"].clip(min=lo, max=hi))
    #var_name = "conc"
    
    # filter out failed runs
    # bad samples return -1000, good samples return 0
    bad_conc = ((q99_space_max_time < lo) | (q99_space_max_time > hi)).values
    bad_iid = (ds3['return_code'] < 0 | bad_conc).any(dim='QMC').compute().to_numpy()
    #print(bad_iid)
    ds_ok = ds3.sel(IID=~bad_iid)
    print("Remaining IIDs:", len(ds_ok['IID'].values))
    return input_design, ds3, var_name  #ds_ok


def q(da: xr.DataArray, q: float, dim) -> xr.DataArray:
    """Quantile reduction with NaN handling."""
    return da.quantile(q, dim=dim, skipna=True).reset_coords("quantile", drop=True)

def _five_stats_vars(da: xr.DataArray, reduce_dims, prefix: str) -> dict[str, xr.DataArray]:
    """
    Build a dict of five statistics over `reduce_dims` with names:
      {prefix}q025, {prefix}q500, {prefix}q975, {prefix}mean, {prefix}std
    """
    #q = da.quantile([0.025, 0.5, 0.975], dim=reduce_dims)
    out = {
        f"{prefix}q025": q(da, 0.025, dim=reduce_dims),
        f"{prefix}q500": q(da,0.5, dim=reduce_dims),
        f"{prefix}q975": q(da,0.975, dim=reduce_dims),
        f"{prefix}mean": da.mean(dim=reduce_dims),
        f"{prefix}std":  da.std(dim=reduce_dims),
    }
    return out


def compute_statistics(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    """
    Produce:
      conc_q025, conc_q500, conc_q975, conc_mean, conc_std
      conc_q99_XYZ, and conc_q99_XYZ_{q025,q500,q975,mean,std}
      conc_q99_time, and conc_q99_time_{q025,q500,q975,mean,std}

    All five stats are reduced over `sample_dims` (default: ('qmc','iid')).
    """
    sample_dims: tuple[str, str] = ("QMC", "IID")
    space_dims: tuple[str, str, str] = ("X", "Y", "Z")
    time_dim: str = "sim_time"

    if var_name not in ds:
        raise KeyError(f"Dataset must contain variable '{var_name}'.")
    
    da = ds[var_name]

    # 1) Five stats for conc over samples
    out_vars = _five_stats_vars(da, sample_dims, prefix=f"{var_name}_")

    # 2) q99 over space → conc_q99_XYZ (+ five stats over samples)
    out_vars[f"{var_name}_q99_XYZ"] = conc_q99_XYZ= q(da, 0.99, dim=space_dims)
    out_vars.update(_five_stats_vars(conc_q99_XYZ, sample_dims, prefix=f"{var_name}_q99_XYZ_"))

    # 3) q99 over time → conc_q99_time (+ five stats over samples)
    out_vars[f"{var_name}_q99_time"] = conc_q99_time = q(da, 0.99, dim=time_dim)
    out_vars.update(_five_stats_vars(conc_q99_time, sample_dims, prefix=f"{var_name}_q99_time_"))
    out_vars[f'{var_name}_q99'] = q(da, 0.99, dim=(time_dim, *space_dims))
    
    return xr.Dataset(out_vars).compute()

def compute_sobol(input_design: ot_sa.InputDesign, da: xr.DataArray) -> xr.Dataset:
    """_summary_
    Compute: S1, S1_ci, ST, ST_ci, S2 for: conc (optional), conc_q99_XYZ, conc_q99_time
    """
    if 'sim_time' in da.dims:
        da = da.isel(sim_time=slice(1, None))  # skip t=0'
    output = set(da.dims).union({'aux'}) - {'IID', 'QMC'}
    conc_2D = da.expand_dims({'aux': [0]}).stack(sample=('IID', 'QMC'), output=output)
    si_conc = input_design.compute_sobol(conc_2D.transpose('sample', 'output').compute())
    si_conc = si_conc.unstack('output')
    return si_conc

#
# def select_params(
#     ds: xr.Dataset,
#     var_threshold: float,
#     si_threshold: float = 0.0,
#
# ) -> (List[Tuple[str, str, Tuple[int, ...]]], List[str]):
#     """
#     Vectorized selection of dominant S1 and S2 terms across outputs.
#     Adds an 'others' row so that each output column sums to 1.0.
#
#     Inputs
#     ------
#     ds  : Dataset with coords group, group2, output, bound and vars:
#           S1(group, output), S2(group, group2, output),
#           S1_agg(group), S1_agg_ci(group, bound)
#           (S2 is always present; identity on diag if not computed.)
#     var_threshold : cumulative sum threshold applied per-output
#     si_threshold  : minimum SI to consider before cum-sum
#
#     Returns
#     -------
#     [('S1'|'S2', <index label>, (i_s1,)| (i_s2, j_s2), important_flag)]
#
#     TODO:
#     0. Assume ds with: S1, S2, ST, S1_agg, S2_agg, ST_agg, S1_agg_ci, ST_agg_ci
#     1. merge S1, S2; S1_agg, S2_agg; extend CI and ST and ST_agg -> new DS
#     2. ranking by SI_agg -> select mask
#     3. assemble selected + 'others' -> new DS
#     """
#
#     #outputs = ds[output_dim].values
#     #bound = ds.get("bound", xr.DataArray(["low", "high"], dims=("bound",))).values
#
#     # Merge S1 and S2 (upper triangle i<j)
#     S1 = ds["S1"].values                # (G, M)
#     G, M = S1.shape
#     S1 = S1[:, 0]                 # (G, ) M==1 is asserted in the function
#     assert M == 1
#     I, J = np.triu_indices(G, k=1)
#     S2_pairs = ds["S2"].values[I, J, 0]              # (P, M) with P = G*(G-1)/2
#     SI_all = np.concatenate([S1, S2_pairs])  # (T, ) T=G+P
#
#     # Per-output ranking → cum-sum → cut
#     order = np.argsort(-SI_all)                           # (T,) descending
#     sorted_vals = SI_all[order]      # (T,)
#
#     valid_mask = (sorted_vals >= si_threshold)                    # (T,)
#     cs = (sorted_vals * valid_mask).cumsum()
#
#     # select large enough Sobol indices from S1 and S2
#     # in original order
#     final_mask = np.zeros_like(valid_mask)  # (T,)
#     final_mask[order] = valid_mask * (cs <= var_threshold)
#
#     groups = ds["group"].values
#     g_labels = groups.astype(str)
#     labels_all = [("S1", str(g), (i,)) for i, g in enumerate(g_labels)]
#     labels_all.extend(
#          [("S2", f"{g_labels[i]}×{g_labels[j]}", (i, j)) for i, j in zip(I, J)]
#     )
#
#     # labels_all_sel = [labels_all[i] for i in range(len(labels_all)) if final_mask[i]]
#     # assert 0 <= len(labels_all_sel) <= len(labels_all), \
#     #     f"Wrong selection: 0 < {len(labels_all_sel)} <= {len(labels_all)}"
#     return labels_all
#

def select_params(
    ds: xr.Dataset,
    var_threshold: float,
    si_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Vectorized selection of dominant S1 and S2 terms across outputs.

    Returns DataFrame sorted by 'si' desc with columns:
      base, label, si, si_err, idx_0, idx_1, valid, selected
    """

    # S1: (G, M) and we assume M==1 (same behavior as original)
    S1 = ds["S1"].values
    assert S1.ndim == 2, f"S1 must have dims (group, output). Got shape={S1.shape}"
    G, M = S1.shape
    assert M == 1, f"Expected a single output (M==1), got M={M}"
    s1_vals = S1[:, 0].astype(float)

    # S2: (G, G, M)
    S2 = ds["S2"].values
    assert S2.ndim == 3, f"S2 must have dims (group, group2, output). Got shape={S2.shape}"
    assert S2.shape[0] == G and S2.shape[1] == G and S2.shape[2] == M, \
        f"S2 must have shape ({G}, {G}, {M}), got {S2.shape}"

    I, J = np.triu_indices(G, k=1)
    s2_vals = S2[I, J, 0].astype(float)

    si = np.concatenate([s1_vals, s2_vals])

    # Errors (optional): *_boot_err; if present, must match shapes
    if "S1_boot_err" in ds:
        s1_err_all = np.asarray(ds["S1_boot_err"].values, dtype=float)
        assert s1_err_all.shape == (G, M, 2), f"S1_boot_err must have shape {(G, M, 2)}, got {s1_err_all.shape}"
        s1_err = s1_err_all[:, 0, :]
    else:
        s1_err = np.full(G, np.nan, dtype=float)

    if "S2_boot_err" in ds:
        s2_err_all = np.asarray(ds["S2_boot_err"].values, dtype=float)
        assert s2_err_all.shape == (G, G, M, 2), \
            f"S2_boot_err must have shape {(G, G, M, 2)}, got {s2_err_all.shape}"
        s2_err = s2_err_all[I, J, 0, :]
    else:
        s2_err = np.full(len(I), np.nan, dtype=float)

    si_err = np.concatenate([s1_err, s2_err])

    # Labels
    groups = ds["group"].values if "group" in ds.coords else np.arange(G)
    g_labels = groups.astype(str)

    base = (["S1"] * G) + (["S2"] * len(I))
    label = ([str(g) for g in g_labels]) + [f"{g_labels[i]}×{g_labels[j]}" for i, j in zip(I, J)]
    idx_0 = list(range(G)) + list(I.tolist())
    idx_1 = ([np.nan] * G) + list(J.tolist())

    df = pd.DataFrame(
        {
            "base": base,
            "label": label,
            "si": si,
            "si_err_lower": si_err[:, 0],
            "si_err_upper": si_err[:, 1],
            "idx_0": idx_0,
            "idx_1": idx_1,
        }
    )
    df[["idx_0", "idx_1"]] = df[["idx_0", "idx_1"]].astype("Int64")
    df["valid"] = df["si"].ge(si_threshold) & np.isfinite(df["si"])
    df = df.sort_values("si", ascending=False, kind="mergesort").reset_index(drop=True)

    contrib = df["si"].where(df["valid"], 0.0).to_numpy()
    cs = np.cumsum(contrib)
    df["selected"] = df["valid"] & (cs <= var_threshold)
    print(df)
    return df

def select_sobol(
    df_si: pd.DataFrame,
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Assemble a Dataset with selected parameters and an 'others' row.

    Inputs
    ------
    df_si: DataFrame with summary of the Sobol indices for scalar QoF (quantile over both time and space)
    ds : Dataset with at least:
        S1(group,*out), S2(group,group2,*out), ST(group,*out),
        S1_agg(group), ST_agg(group),
        S1_agg_ci(group,bound), ST_agg_ci(group,bound)

    Outputs
    -------
    xr.Dataset with variables:
      SI(group_sel,*out), ST(group_sel,*out),
      SI_ci(group_sel,bound), ST_ci(group_sel,bound)
    """

    if "S1" not in ds or "S2" not in ds:
        raise ValueError("Dataset must contain S1 and S2.")

    group_dim = "group"
    s1_dims = list(ds["S1"].dims)
    out_dims = s1_dims[1:]
    #out_shape = tuple(ds["S1"].sizes[d] for d in out_dims)

    # --- accessor functions for SI ---
    getters = {
        "S1": lambda idx: ds["S1"].isel({group_dim: idx[0]}).values,
        "S2": lambda idx: ds["S2"].isel(
            {group_dim: min(idx), "group2": max(idx)}
        ).values,
    }

    # --- build SI rows for selected entries ---

    si_rows = [getters[base]((i, j )) for base, i, j  in zip(df_si.base, df_si.idx_0, df_si.idx_1)]
    si_array = np.stack(si_rows, axis=0)
    total_si_from_rows = si_array.sum(axis=0)
    si_sel_array = si_array[df_si.selected.to_numpy()]
    total_si_selected = si_sel_array.sum(axis=0)
    others = np.maximum(total_si_from_rows - total_si_selected, 0.0)
    # be resistant to possible loss of precision

    # Form dataframe for selected
    df_selected = df_si[df_si.selected]
    labels = df_selected.label.tolist() + ["others"]
    SI = np.concatenate((si_sel_array, others[None, ...]))

    # --- ST variable (only for S1 entries, else zeros) ---
    ST = np.zeros_like(SI)
    SI_ci = np.zeros((SI.shape[0], 2))  # (N_sel, 2) for 'low', 'high'
    ST_ci = np.zeros((SI.shape[0], 2))  # (N_sel, 2) for 'low', 'high'
    assert len(df_selected) == SI.shape[0] - 1
    for i, (base, i0) in enumerate(zip(df_selected.base, df_selected.idx_0)):
        if base == "S1":
            ST[i] = ds["ST"].isel({group_dim: i0}).values
            SI_ci[i] = ds["S1_agg_ci"].isel({group_dim: i0}).values
            ST_ci[i] = ds["ST_agg_ci"].isel({group_dim: i0}).values
    # --- coords and DataArrays ---
    bound = ds['bound']
    coords = {"group": np.array(labels, dtype=object), "bound": bound.values}
    for d in out_dims:
        coords[d] = ds[d].values

    return xr.Dataset(
        data_vars=dict(
            SI=(("group", *out_dims), SI),
            ST=(("group", *out_dims), ST),
            SI_ci=(("group", "bound"), SI_ci),
            ST_ci=(("group", "bound"), ST_ci),
        ),
        coords=coords,
    )

def plot_vtk(ds_stat: xr.Dataset, sobol_conc: xr.Dataset, sobol_conc_q99_time: xr.Dataset, 
             var_name: str,
             fout: Path = "transport_statistics.pvd") -> None:
    """Write 'transport_statistics.pvd' with stats & Sobol fields on two Z-surfaces as PointArrays.

    Assumptions
    -----------
    - ds_stat has coords: X (nodes), Y (nodes), Z (len==2), sim_time (float-like).
    - All fields written are **nodal** with dims (..., Z, Y, X).
    - Arrays are already on the node grid; we just flatten and attach as point_data.
    """

    # node coordinates (StructuredGrid points)
    X = np.asarray(ds_stat["X"].values, dtype=float)  # shape (nx)
    Y = np.asarray(ds_stat["Y"].values, dtype=float)  # shape (ny)
    Z = np.asarray(ds_stat["Z"].values, dtype=float)  # expect len==2
    time_values = np.asarray(ds_stat["sim_time"].values, dtype=float)

    # ---- helpers -------------------------------------------------------------
    def _safe_name(s: str) -> str:
        return str(s).replace("×", "x").replace(" ", "_")

    def _structured_surface(zval: float) -> pv.StructuredGrid:
        xx, yy = np.meshgrid(X, Y, indexing="ij")        # (nx, ny)
        zz = np.full_like(xx, float(zval), dtype=float)  # (nx, ny)
        return pv.StructuredGrid(xx, yy, zz)

    # Given nodal DataArray with dims including Z,Y,X, flatten all nodes layer-wise.
    # Fortran order per layer to match VTK structured point ordering, then stack [Z=0; Z=1]
    def _points(da: xr.DataArray) -> np.ndarray:
        arr_zyx = da.squeeze(drop=True).transpose("Z", "Y", "X").values                 # (2, ny, nx)
        per_layer = arr_zyx.reshape(arr_zyx.shape[0], -1, order="F") # (2, nx*ny)
        return per_layer.ravel(order="C")                            # [top nodes..., bottom nodes...]

    # ---- build merged mesh (two layers) -------------------------------------
    surf_top = _structured_surface(Z[0])
    surf_bot = _structured_surface(Z[1])
    merged_base = surf_top.merge(surf_bot, merge_points=False)  # points: top then bottom

    # ---- precompute time-independent (aggregated) Sobol arrays --------------
    agg_arrays: dict[str, np.ndarray] = {}
    if sobol_conc_q99_time:            
        for label in sobol_conc_q99_time["SI"]["group"].values.astype(str):
            agg_arrays[f"SI_agg_{_safe_name(label)}"] = _points(sobol_conc_q99_time["SI"].sel(group=label))
            agg_arrays[f"ST_agg_{_safe_name(label)}"] = _points(sobol_conc_q99_time["ST"].sel(group=label))

    stat_vars = [f"{var_name}_{suffix}" for suffix in ("q025", "q500", "q975", "mean", "std")]

    # ---- write PVD time series ----------------------------------------------
    with endorse.PVDWriter(fout) as writer:
        for t_idx, t_val in enumerate(time_values):
            mesh = merged_base.copy(deep=True)

            # stats (time-dependent)
            for name in stat_vars:
                mesh.point_data[name] = _points(ds_stat[name].isel(sim_time=t_idx))

            # Sobol (time-dependent)
            if sobol_conc:
                for label in sobol_conc["SI"]["group"].values.astype(str):
                    mesh.point_data[f"SI_{_safe_name(label)}"] = _points(
                        sobol_conc["SI"].sel(group=label).isel(sim_time=t_idx)
                    )
                    mesh.point_data[f"ST_{_safe_name(label)}"] = _points(
                        sobol_conc["ST"].sel(group=label).isel(sim_time=t_idx)
                    )

            # Sobol aggregated (time-independent)
            for nm, arr in agg_arrays.items():
                mesh.point_data[nm] = arr

            writer.write(mesh, float(t_val))




# def plot_conc(ds_stat, si_q99_XYZ, si_q99, var_name):
#     """_summary_
#     - VTK conc_mean, conc_std, conc_q025, conc_q500, conc_q975; space time fields
#     - space-time S1_<param>, ST_<param> fields

#     - conc_q99_space_<var>
#     Args:
#         ds_stat (_type_): _description_
#         cfg (_type_): _description_
#     """
#     out_dir = Path(cfg['output_dir'])
#     out_dir.mkdir(parents=True, exist_ok=True)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     times = ds_stat['sim_time'].values
#     conc_p01 = ds_stat[f{var_name}_q025"].mean(dim=('X', 'Y', 'Z')).values
#     conc_p50 = ds_stat.sel(quantile=0.50)['conc'].mean(dim=('X', 'Y', 'Z')).values
#     conc_p99 = ds_stat.sel(quantile=0.99)['conc'].mean(dim=('X', 'Y', 'Z')).values

#     ax.fill_between(times, conc_p01, conc_p99, color='lightgray', label='1-99 percentile')
#     ax.plot(times, conc_p50, color='blue', label='Median')
#     #ax.set_xscale('log')
#     #ax.set_yscale('log')
#     ax.set_xlabel('Simulation Time')
#     ax.set_ylabel('Concentration (averaged over space)')
#     ax.set_title('Concentration Statistics Over Time')
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(out_dir / "concentration_statistics.pdf")
#     plt.close()

# def valid_sample_data(input_design, ds):
#     store_path = str(input_data.zarr_store_path)
#     ds = xr.open_zarr(store_path, consolidated=True)
#     rc = ds['return_code'].to_numpy()
#     print('Return code:\n', rc)
#     print(f"Number of failed runs: {(rc < 0).sum()} / {rc.size}")
#     valid_iid = (ds['return_code'] >= 0).all(dim='QMC').to_numpy()  # mask failed runs
#     n_valid = int(valid_iid.sum()) 
#     print(f"Valid iid: {n_valid} ")
#     return input_design, ds


def sobol_ds_summary(ds: xr.Dataset, df_si:pd.DataFrame) -> None:
    """
    param_selection: Tuple['S1'|'S2', label, (i,)|(i,j), selected_flag]
    """
    tab_lines = []
    header = (f"{'ord':>3}  {'groups':<32}  {'S1':<7} < {'S1_low':<6}  {'S1_high':<6}> |  "
          f"{'ST':<7} < {'ST_low':<6}  {'ST_high':<6}>")
    tab_lines.append(header)
    double_fmt = "3.4f"

    def line(base, label, si_est, si_err, si_agg=None):
        line = (
            f"{label:<32}  "  # order, right-aligned in 3 chars
            f"{base:<3} "  # label, left-aligned in 20 chars
            f"{si_est: {double_fmt}}  "  # S1, width 10, 6 decimals
            f"<{si_err[0]: {double_fmt}}, "  # S1 CI low
            f"{si_err[1]: {double_fmt}}>  "  # S1 CI high
        )
        if si_agg is not None:
            line += (
                f" AGG: <{si_agg[0]: {double_fmt}}, "  # S1 CI low
                f"{si_agg[1]: {double_fmt}}>"  # S1 CI high
            )
        return line

    for i in range(len(df_si)):
        base, label, i, j = df_si.iloc[i][['base', 'label', 'idx_0', 'idx_1']]
        indices = (i,) if base == 'S1' else (i, j)
        si_est = (ds[base].values[indices])[0]
        si_err = (ds[f"{base}_boot_err"].values[indices])[0]
        assert si_err.shape == (2,), f"Unexpected shape for {base}_boot_err: {si_err.shape}"
        si_agg = (ds[f"{base}_agg_ci"].values[indices]) if base == 'S1' else None
        #assert si_agg.shape == (2,), f"Unexpected shape for {base}_agg_ci: {si_agg.shape}"
        tab_lines.append(
            line(base, label, si_est, si_err, si_agg)
        )
        if base == 'S1':
            tab_lines.append(
                line('ST', len(label)*" ",
                     (ds['ST'].values[indices, 0])[0],
                     (ds['ST_boot_err'].values[indices])[0],
                     (ds[f"ST_agg_ci"].values[indices])
                     )
            )

    return tab_lines

def make_transport_plots(cfg, seed):
    """
    Generates transport-related plots and saves them to out_dir.
    1. compute data arrays
    2. SIs for conc_q99_XYZ, conc_q99_time, and conc_q99
    """ 
    input_design, ds, var_name = sampling_data(cfg, seed)

    n = input_design.n_samples
    ds_stat = compute_statistics(ds, var_name)
    print(list(ds_stat.data_vars.keys()))
    sobol_boot = lambda conc_da: input_design.compute_sobol_xr(conc_da, n_boot=100)
    sobol = lambda conc_da: input_design.compute_sobol_xr(conc_da)

    print("Computing Sobol indices for 'conc.q99(time & space)' ...")
    si_conc = sobol_boot(ds_stat[f'{var_name}_q99'])
    df_si = select_params(si_conc, var_threshold=0.9, si_threshold=0.01)



    
    sobol_sel = lambda conc_da: select_sobol(df_si, sobol(conc_da))
    #plot_vtk(ds_stat, sobol_sel(ds['conc']), sobol_sel(ds_stat['conc_q99_time']))
    print(f"Computing Sobol indices for '{var_name}_q99(space)' ...")
    si_q99 = sobol_sel(ds_stat[f'{var_name}_q99'])
    si_q99_XYZ = sobol_sel(ds_stat[f'{var_name}_q99_XYZ'])

    si_table = sobol_ds_summary(si_conc, df_si)
    print("Sorted S1, S2 values:\n")
    print("\n".join(si_table))

    print("Plot conc & SI ...")
    plots.save_conc_and_si_pdf(ds_stat, si_q99_XYZ, si_q99, var_name, job.output.dir_path.stem,
                               figsize=(11, 5), si_ci_level=0.90, si_table=si_table,
                               out_pdf_path=job.output.plots / "conc_and_si.pdf")

    print("Plot VTK ...")
    plot_vtk(ds_stat, None, sobol_sel(ds_stat[f'{var_name}_q99_time']), var_name, 
             fout=job.output.plots/"transport_statistics.pvd")
    #plot_vtk(ds_stat, None, None)
    
    # input_design_val, ds_val = valid_sample_data(input_design, ds)
    # rc = ds['return_code'].to_numpy()
    # print('Return code:\n', rc)
    # print(f"Number of failed runs: {(rc < 0).sum()} / {rc.size}")
    # valid_iid = (ds['return_code'] >= 0).all(dim='qmc').to_numpy()  # mask failed runs
    # n_valid = int(valid_iid.sum()) 
    # print(f"Valid iid: {n_valid} ")
    # conc = ds['conc'].isel(sim_time=slice(1, None), iid=valid_iid)  # skip t=0
    # conc_max_space = conc.quantile(0.99, dim=('X', 'Y', 'Z')).compute()

    # si_space_agg = si_ds.copy()
    # space_dims = ('X', 'Y', 'Z')
    # for var in ['S1', 'ST', 'S2']:
    #     si_space_agg[var] = si_space_agg[var].weighted(conc_vars).mean(dim=space_dims)
    # for d in space_dims:
    #     si_space_agg = si_space_agg.drop_vars(d)

    # filtered_ds = vsp.select_sobol_terms_with_others(si_space_agg, 0.9, output_dim='sim_time', si_threshold=0.05)
    # vsp.plot_sobol_stacked(filtered_ds, figsize=(10, 5), out_path=output_dir / "sobol_stacked.pdf")

