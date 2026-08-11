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
