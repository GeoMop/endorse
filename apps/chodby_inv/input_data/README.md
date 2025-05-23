# Principal input data for inversion

- place here all original unprocessed data and configuration files 
  used during any stage of the inversion

- do not use file names in the code directly but through
  the variables in `__init__.py`, e.g.

  ```
  from ..input_data import input_dir
  ```
  or
  ```
  from .. import input_data as inputs
  
  bh_file = inputs.bh_cfg_file
  ```
  
## Overview
`blast_events.xlsx` 
table with blast events according to excavation diary, contains both original data from diary
and manual modifications to fit to the pressure measurements
                  
`boreholes.yaml`
geometry of the boreholes

`piezo_*.xlsx`
appended table with continuous pressure measurement, shortes versioned directly, 
subsequent large tables through DVC

`wpt_2024_02_with_flux.xlsx`
first full water pressure tests including water flux measurements
on large sections of the whole boreholes

`wpt_2025_04_with_flux_on_multipacker.xlsx`
measurement by GDS (small fluxes) or weighting the expansion vessel
must be merged with piezo file for full picture


`resources` - supplementary documents from where the machine readable data were extracted 
