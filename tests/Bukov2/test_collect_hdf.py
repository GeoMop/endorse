import pytest
import os, sys, subprocess
from pathlib import Path
import collect_hdf
script_dir = Path(__file__).absolute().parent
workdir = script_dir/"samples_tst"


def test_collect_main():
    for f in ["chunked_mean.pkl", "sampled_collected.h5", "sampled_fixed.h5"]:
        (workdir / f).unlink(missing_ok=True)
    collect_hdf.main(workdir)
    collect_hdf.main(workdir)

@pytest.mark.skip
def test_collect():
    print()
    for f in ["chunked_mean.pkl", "sampled_collected.h5", "sampled_fixed.h5"]:
        (workdir / f).unlink(missing_ok=True)
    args = [
        sys.executable, script_dir / "collect_hdf.py", workdir
    ]
    subprocess.run(args,  check=True)
    subprocess.run(args, check=True)
