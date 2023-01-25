#!/bin/bash

main_mlmc=/storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/mlmc/mlmc_main.py
work_dir=/storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/

mlmc_lib=/storage/liberec3-tul/home/martin_spetlik/Endorse_full_transport/submodules/MLMC
singularity_path=/storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/mlmc/endorse.sif
endorse_repository=/storage/liberec3-tul/home/martin_spetlik/Endorse_full_transport


./pbs_submit_2.sh ${main_mlmc} ${work_dir} ${mlmc_lib} ${singularity_path} ${endorse_repository}