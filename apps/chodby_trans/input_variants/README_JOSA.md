# Sensitivity analysis for near-field transport in crystalline rock

## `extensive_technical_report`

A technical report with a detailed description of the forward model and sampling process.
Some cases are described with smaller sample sizes.
Sensitivity analysis is in chapter 5.
We are preparing an SA-focused and updated version.


## Datasets

All cases have target sample size N=2048, but some simulations failed due to meshing issues resulting
in about 1500-2000 independent samples with all Saltelli A, B, AB, and BA evaluations completed correctly.
The provided datasets contain just the completed samples.

Export formats:
- `*.csv` - text representation
- `*.parquet` - binary tabular format that Pandas can read efficiently
- `*.zarr` - reduced xarray dataset; the raw values on the top and bottom evaluation surfaces are also available for readers that need them.

Datasets files:
- `log10_conc_q99_XYZ.[csv|parquet]` - time-dependent outcome, i.e. q99 over space only
- `log10_conc_q99.[csv|parquet]` - scalar, time-independent outcome, i.e. q99 over both time and space
- `parameter.[csv|parquet]` - parameter values for each valid evaluation
- `group_parameters.[csv|parquet]` - group-level parameter values for each valid evaluation
- `parameter_group_map.[csv|parquet]` - mapping from parameter name to parameter group


## Columns

The reduced statistics are written as flat tables. The available columns depend on the file:

- `log10_conc_q99.parquet` and `log10_conc_q99.csv`
  - `log10_conc_q99` - q99 over all simulated time steps and spatial coordinates

- `log10_conc_q99_XYZ.parquet` and `log10_conc_q99_XYZ.csv`
  - `sim_time` - time in years from contaminant release
  - `log10_conc_q99_XYZ` - q99 over the spatial dimensions `X`, `Y`, and `Z`

- `parameter.parquet` and `parameter.csv`
  - `IID` - index of the independent sample row in the Saltelli design
  - `QMC` - index within the Saltelli layout
  - one column per parameter name - the parameter value for that sample

- `group_parameters.parquet` and `group_parameters.csv`
  - `IID` - index of the independent sample row in the Saltelli design
  - `QMC` - index within the Saltelli layout
  - one column per parameter group - the group-level value for that sample

- `parameter_group_map.parquet` and `parameter_group_map.csv`
  - `param_name` - parameter name
  - `parameter_group_map` - name of the group that the parameter belongs to

The `.zarr` store contains the same reduced statistics in xarray form, plus the sample-level
`parameter`, `group_parameters`, and `parameter_group_map` variables.


## Cases
Described in detail in the report.

CASE 0
- reference scenario
- 1497/2048 valid samples
- 27132/28672 valid evaluations

CASE 1
- conservative scenario, no DFN
- 2038/2048 valid samples
- 20470/20480 valid evaluations

CASE 2
- DFN parameter groups refined: dfn_transport and dfn_pop
- bulk and backfill parameters grouped together (mat_BB)
- 1500/2048 valid samples
- 27123/28672 valid evaluations

CASE 3
- increased hydraulic conductivity and dispersivity
- 1495/2048 valid samples
- 27117/28672 valid evaluations
