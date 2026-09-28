"""Print HM convergence evidence from Flow123d logs (no solver execution)."""

import json
from pathlib import Path
import re
import sys


def summarize(path):
    steps = []
    failures = []
    for line in path.read_text().splitlines():
        time = re.search(r"TG\[HM\].*?t:\s*(\S+)", line)
        if time:
            steps.append({"time": time[1], "iterations": []})
        iteration = re.search(
            r"HM Iteration (\d+) abs\. difference: (\S+)\s+rel\. difference: (\S+)", line
        )
        if iteration:
            steps[-1]["iterations"].append(list(iteration.groups()))
        if "convergence reason -" in line or "Program Error:" in line:
            failures.append(line.strip())
    return {
        "log": str(path),
        "steps": len(steps),
        "first_ten": steps[0]["iterations"][:10] if steps else [],
        "last_step": steps[-1] if steps else None,
        "failures": failures,
    }


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print(json.dumps(summarize(Path(filename)), indent=2))
