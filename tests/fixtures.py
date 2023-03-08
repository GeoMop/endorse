"""
Common code for tests.
"""
import os
from pathlib import Path

def sandbox_fname(base, base_name=None):
    work_dir = os.path.join(base, "sandbox")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    if base_name is not None:
        return os.path.join(work_dir, f"{base_name}")
    else:
        return work_dir


