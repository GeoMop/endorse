"""
Flow123d launcher for Windows + Docker Desktop, replacing the plain .bat wrapper for use with
endorse's call_flow.

Why: endorse passes ABSOLUTE host paths (e.g. the rendered main input YAML) to the flow executable.
On Linux the container mounts mirror host paths, so absolute paths survive; on Windows
'C:\\...' is meaningless inside the Linux container. This wrapper translates every Windows-style
absolute path argument (and the working directory) to the container form '/C/...', mirroring the
mounts of the original flow-noterm .bat, then runs flow123d in the same Docker image.

Usage (machine_config):
    flow_executable:
      - <venv python.exe>
      - <this file>
      # optional extra flow123d arguments may follow; the main input YAML is appended by call_flow
"""
import os
import re
import subprocess
import sys

DOCKER = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
IMAGE = "flow123d/ci-gnu:4.0.3dev_e651b9"

_WIN_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def to_container(path: str) -> str:
    m = _WIN_PATH.match(path)
    if not m:
        return path
    drive, rest = m.group(1), m.group(2).replace("\\", "/")
    return f"/{drive}/{rest}"


def main() -> int:
    cwd = os.getcwd()
    drive = cwd[0]
    container_cwd = to_container(cwd)
    args = [to_container(a) for a in sys.argv[1:]]

    cmd = [
        DOCKER, "run", "--rm",
        "-v", f"{drive.upper()}:\\:/{drive.upper()}/",
        "-v", f"{drive.lower()}:\\:/{drive.lower()}/",
        "-w", container_cwd,
        IMAGE, "flow123d", *args,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
