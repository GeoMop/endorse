# Homogenization study

This application studies the random homogenized conductivity tensor field produced by a microscale box containing
multiple conductivity regions and a discrete fracture network.

The first stage is a deterministic problem:

1. Create the microscale test mesh and its material fields.
2. Create one fixed homogenization mesh.
3. Run the three Flow123d pressure-load problems used by the existing homogenization algorithm.
4. Return six tensor components for every bulk element of the homogenization mesh.

The later sampling stage will run this problem as single-level Monte Carlo through the MLMC paired-sampling path. MLMC
HDF storage will manage samples and restart state, while Zarr will store the complete tensor fields and diagnostics.

The application is intentionally independent of `chodby_trans`. Similar modules should keep compatible concepts and
names where practical so stable common behavior can later move into `src/endorse`.
