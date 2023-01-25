import os
import pytest
import shutil
from endorse import common
#from endorse.scripts.endorse_mlmc import FullScaleTransport
import subprocess
script_dir = os.path.dirname(os.path.realpath(__file__))
endorse_dir = os.path.join(script_dir, "../..")



def run_script(args, workdir=None):
    script_args = ['python', os.path.join(endorse_dir, 'src/endorse/scripts/endorse_mlmc.py')]
    # TODO: run as subprocess
    if workdir is None:
        workdir = os.path.join(script_dir, '../sandbox/mlmc_run')

    cfg = common.load_config('../test_data/config.yaml', collect_files=True)
    inputs = cfg._file_refs
    with common.workdir(workdir, inputs):
        subprocess.run(script_args + args)


# collect samples
@pytest.mark.skip
def test_FullScaleTransport_run():
    #common.EndorseCache.instance().expire_all()
    case='edz_pos02'
    #case='edz_pos10'
    #case='noedz_pos02'
    #case='noedz_pos10'

    argv = [ 'run', f'sandbox/{case}', '--clean', '--debug']
    args = FullScaleTransport.get_arguments(argv)
    pr = FullScaleTransport(f"test_data/cfg_mlmc_{case}.yaml", args)

@pytest.mark.skip
def test_plot_cases():
    #run_script(['plot', 'cases', '*', '2'])
    #run_script(['plot', 'cases', 'base dg_1 dg_3 dg_30 tol_low tol_high', '2'])
    source_dir="../test_data/collected"
    workdir="../sandbox/mlmc_run_plots"
    shutil.rmtree(workdir)
    shutil.copytree(source_dir, workdir)

    run_script(['plot', 'mc_cases', 'edz_0,noedz_0', '2,5,10'], workdir=workdir)

#@pytest.mark.skip
def test_plot_cases_variants():
    #run_script(['plot', 'cases', '*', '2'])
    #run_script(['plot', 'cases', 'base dg_1 dg_3 dg_30 tol_low tol_high', '2'])
    for i in [0,4]:
        workdir=f"../test_data/collected_charon_run3/calc_230122_02_{i}"
        run_script(['plot', 'mc_cases', f'edz_{i},noedz_{i}', '2,5,10'], workdir=workdir)

@pytest.mark.skip
def test_script_sample():
    for i in [0,1,2,3,4]:
        run_script(['run', '-c', '-nt=2', '-np=1', f'edz_{i},noedz_{i}', '2'])


@pytest.mark.skip
def test_script_sample_2d():
    #run_script(['run', '*', '2 10'])
    #run_script(['run', '-c', 'edz', '2'])
    #run_script(['run', '-c', 'edz_base edz_lower_tol edz_high_gamma edz_both', '2'])
    run_script(['run', '-c', '-nt=2', '-np=2', '--dim=2', 'edz', '2, 10'])

