# Sources for the whole Endorse application including GUI.



- `flow123d_inputs` : Templates for the main Flow123d input file, always contains mesh file as the parameter. Meshes are mostly generated using BGEM.

- `meshes` : package with specific mesh generating functions 

## `bayes`
TODO: move whole functionality of `bayes_orig` here

## `bayes_orig`
Bayes inversion applied to TSX.

## `common` 
Package with auxiliary (technical) classes and functions specific to the project

## `fitting` 
Processing Borehole measurements.

## `flow123d_inputs`
flow123d templates. TODO: move to problem specific 

## `laser_scan`
Processing laser scan data to surface meshes.


## `mesh`

## `mlmc`
submodule, TODO: modularize, add DASK support

## `scripts`
High level executables.

## `sobol`
Fast, but incomplete sobol estimation.


apply_fields.py
aux_functions.py
flow123d_simulation.py
fullscale_transport.py
hm_simulation.py
HM_transport.py
homogenisation.py
indicator.py
__init__.py
large_mesh_shift.py
macro_flow_model.py
mesh_class.py
mesh_plots.py
plots.py
README.md

