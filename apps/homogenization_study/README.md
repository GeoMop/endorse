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

Sampling writes the schema-defined dataset to the `tensors` group of `homogenization_sampling`. Convert a completed
local store to the Zarr Fuse destination declared by `input_data/data_schema_empty.yaml` with:

```bash
python zarr_to_fuse.py workdir/homogenization_sampling --overwrite
```

Use `--store-url` to select another local or S3 destination and `--endpoint-url` to override the S3 endpoint. Zarr
Fuse reads `ZF_S3_ACCESS_KEY` and `ZF_S3_SECRET_KEY` from the environment. Without `--overwrite`, an existing
destination is merged according to the schema. The converter also normalizes the previous root-level
`param_val(sample_id,parameter)` layout when exporting an older completed store.
