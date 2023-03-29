import pytest
import os
from endorse import common
from endorse.bayes import measured_data as md

script_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.skip
def test_run_measured_data():
    print(os.getcwd(), flush=True)
    conf_file = os.path.join(script_dir, "test_data/config_homo_tsx.yaml")
    cfg = common.load_config(conf_file)
#
# files_to_copy = ["test_data/accepted_parameters.csv"]
# with common.workdir(f"sandbox/hm_model_{seed}", inputs=files_to_copy, clean=False):

    files_to_copy = ["test_data/tsx_measured_data"]
    work_dir_name = f"sandbox/measured_data"
    with common.workdir(work_dir_name, inputs=files_to_copy, clean=False):
        print(os.getcwd(), flush=True)
        mdata = md.MeasuredData(cfg.tsx_hm_model)
        mdata.initialize()

        mdata.plot_all_data()
        mdata.plot_interp_data()

        boreholes = ["HGT1-5", "HGT1-4", "HGT2-4", "HGT2-3"]
        times, values = mdata.generate_measured_samples(boreholes)

        print(times, values)


if __name__ == "__main__":

    test_run_measured_data()
