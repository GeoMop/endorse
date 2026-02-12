# COMPUTATIONAL VARIANTS FOR PROJECT "CHODBY"

---------------------------------------------------------
## CASE 1
- workdir_40_ot_2048 - limit 512, FINISHED
  - [mat_bulk, mat_edz, mat_backfill, meshing_seed]
  - dopocitat 512
  - bez puklin
  - kolekce interpolaci => invalidni time run

- workdir_40a_ot_2048 - limit 512, FINISHED
  - stejne jako workdir_40_test_ot_2048
  - zkracena casova rada
  - fixed edz params

- workdir_40b_ot_2048 - queued default, 12h, limit 1024
  - stejne jako workdir_40a_ot_2048
  - source_sigma pulse

---------------------------------------------------------
## CASE 0

- workdir_41d_ot_2048 - queued charon, 48h, limit 1024, FINISHED
  - dfn_params: dfn_transport and dfn_pop
  - [mat_bulk, mat_edz, mat_backfill, meshing_seed, dfn_params, dfn_seed]
  - zkracena casova rada
  - fixed edz params

---------------------------------------------------------
## CASE 2
- workdir_42b_ot_2048 - FINISHED
  - mat_BB: group mat_backfill and mat_bulk
  - [mat_BB, mat_edz, meshing_seed, dfn_pop, dfn_transport, dfn_seed]

- workdir_42c_ot_2048 - limit 1024, FINISHED
  - workdir_42b_ot_2048 s kratsi casovou osou (jako workdir_41d_ot_2048)
  - fixed edz params

---------------------------------------------------------
## CASE 0a
- workdir_43a_ot_2048 - queued 2d, 48h, limit 1024
  - vyssi disperze o rad na bulku:
    - bulk_disp_L: log10(5.0)
  - 1024 samplu
  - casova jako workdir_41d_ot_2048
  - fixed edz params
