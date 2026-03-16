# COMPUTATIONAL VARIANTS FOR PROJECT "CHODBY"

---------------------------------------------------------
## CASE 0

- workdir_41d_ot_2048 - FINISHED
  - dfn_params: dfn_transport and dfn_pop
  - [mat_bulk, mat_edz, mat_backfill, meshing_seed, dfn_params, dfn_seed]
  - shorter time axis
  - fixed EDZ params

---------------------------------------------------------
## CASE 1

- 40a_ot_2048 - limit 512, FINISHED
  - [mat_bulk, mat_edz, mat_backfill, meshing_seed]
  - without DFN
  - shorter time axis
  - fixed EDZ params

- 40b_ot_2048 - FINISHED
  - same as 40a_ot_2048
  - source_sigma pulse

- 40c_ot_2048 - FINISHED
  - same as 40b_ot_2048
  - increased cond and disp_L

---------------------------------------------------------
## CASE 2
- 42b_ot_2048 - FINISHED
  - mat_BB: group mat_backfill and mat_bulk
  - [mat_BB, mat_edz, meshing_seed, dfn_pop, dfn_transport, dfn_seed]

- 42c_ot_2048 - FINISHED
  - same as 42b_ot_2048
  - shorter time axis
  - fixed EDZ params

---------------------------------------------------------
## CASE 0a
- 43a_ot_2048 - limit 1024, FINISHED
  - higher dispersion by an order on bulk:
    - bulk_disp_L: log10(5.0)
  - shorter time axis
  - fixed EDZ params

- 43b_ot_2048 - limit 1024, FINISHED
  - same as 41d_ot_2048
  - source_sigma pulse, increased cond and disp_L 
