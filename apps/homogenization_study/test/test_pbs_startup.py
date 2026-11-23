"""Exercise PBS shell orchestration without a cluster or physical simulation."""

import ast
import os
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[3]
APP = REPO / "apps/homogenization_study"


@pytest.fixture(params=["intact", "first_line"])
def cluster(tmp_path: Path, request: pytest.FixtureRequest) -> dict[str, str]:
    """Exercise intact argv and inferred PBS truncation of multiline arguments."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, script in {
        "pbsdsh": '''#!/bin/bash
node=$2
shift 3
if [[ "${PBS_TRANSPORT_FAILURE:-0}" != 0 ]]; then
    echo "pbsdsh: failed to contact MOM" >&2
    exit "$PBS_TRANSPORT_FAILURE"
fi
export SCRATCHDIR="$TEST_ROOT/$node"
export HOSTNAME="$node"
if [[ "$PBS_ARGV_MODE" == first_line ]]; then
    args=()
    for arg in "$@"; do args+=("${arg%%$'\\n'*}"); done
    set -- "${args[@]}"
fi
"$@" >> "$TEST_ROOT/remote-task.log" 2>&1
remote_status=$?
echo "pbsdsh: spawned task 0x00000000 on logical node 40 event 2"
case "${PBS_NOTICE:-normal}" in
    normal) echo "pbsdsh: task 0x00000000 exit status $remote_status" >&2 ;;
    duplicate)
        echo "pbsdsh: task 0x00000000 exit status $remote_status"
        echo "pbsdsh: task 0x00000001 exit status $remote_status"
        ;;
    malformed) echo "pbsdsh: task 0x00000000 exit status unknown" ;;
    missing) : ;;
esac
# Some PBS installations report remote failure but return success locally.
exit 0
''',
        "rsync": '''#!/bin/bash
set -eu
while [[ "$1" == -* ]]; do shift; done
if [[ "${FAIL_STAGE:-}" == "$HOSTNAME" && "$1" == "$TEST_ROOT/output/" ]]; then exit 23; fi
if [[ "${FAIL_LOGS:-}" == "$HOSTNAME" && "$1" == */logs/ ]]; then exit 23; fi
if [[ "${NOOP_LOGS:-}" == "$HOSTNAME" && "$1" == */logs/ ]]; then exit 0; fi
if [[ "${FAILURE:-}" == collect && "$1" == "$TEST_ROOT/remote/workdir/" ]]; then exit 23; fi
if [[ "${FAILURE:-}" == noop && "$1" == "$TEST_ROOT/remote/workdir/" ]]; then exit 0; fi
mkdir -p "$2"
cp -a "$1/." "$2/"
''',
        "pkill": "#!/bin/bash\nexit 0\n",
        "clean_scratch": '#!/bin/bash\necho cleaned >> "$TEST_ROOT/events"\n',
    }.items():
        executable = binaries / name
        executable.write_text(script)
        executable.chmod(0o755)
    for node in ("head", "remote"):
        (tmp_path / node).mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "input.txt").write_text("input")
    nodefile = tmp_path / "nodes"
    nodefile.write_text("head\nremote\n")
    return dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}", TEST_ROOT=str(tmp_path),
                PBS_NODEFILE=str(nodefile), SCRATCHDIR=str(tmp_path / "head"), PBS_ARGV_MODE=request.param,
                ENDORSE_DASK_APP_DIR=str(APP), ENDORSE_DASK_APP_MODULE="homogenization_study",
                ENDORSE_DASK_WORKER_MODULE="homogenization_study.problem_simulation",
                ENDORSE_DASK_SCHEDULER_MODULE="homogenization_study.mlmc_worker")


def run_helper(cluster: dict[str, str], commands: str) -> subprocess.CompletedProcess[str]:
    """Run the real helper functions without dispatching its command-line entry point."""
    source = (REPO / "dask/dask_cluster.sh").read_text().split('cmd=${2:-help}')[0]
    # BASH_SOURCE must point into the real dask directory when locating sibling scripts.
    source = source.replace('$(dirname "${BASH_SOURCE[0]}")', str(REPO / "dask"))
    return subprocess.run(["bash", "-c", source + "\n" + commands, "bash",
                           str(Path(cluster["TEST_ROOT"]) / "output")],
                          env=cluster, text=True, capture_output=True, timeout=10)


def test_remote_directories_and_log_collection(cluster: dict[str, str]) -> None:
    """Collect logs with real rsync under each node's own scratch root."""
    (Path(cluster["TEST_ROOT"]) / "bin/rsync").unlink()
    result = run_helper(cluster, '''stage_inputs
for node in head remote; do echo "$node" > "$TEST_ROOT/$node/logs/worker.log"; done
stop_cluster
''')
    assert result.returncode == 0, result.stdout + result.stderr
    root = Path(cluster["TEST_ROOT"])
    assert (root / "output/logs_head/worker.log").read_text() == "head\n"
    assert (root / "output/logs_remote/worker.log").read_text() == "remote\n"


@pytest.mark.parametrize(
    ("remote_status", "notice", "transport_status", "expected_status"),
    [(0, "normal", 0, 0), (23, "normal", 0, 23), (137, "normal", 0, 137),
     (0, "missing", 0, 1), (0, "malformed", 0, 1), (0, "duplicate", 0, 1),
     (0, "normal", 42, 42)],
)
def test_remote_completion_status(
    cluster: dict[str, str], remote_status: int, notice: str, transport_status: int, expected_status: int,
) -> None:
    """Use one PBS completion notice without depending on captured remote stdout."""
    cluster.update(PBS_NOTICE=notice, PBS_TRANSPORT_FAILURE=str(transport_status))
    result = run_helper(cluster, f'''pbs_remote remote bash -c '
echo remote-payload
echo executed > "$SCRATCHDIR/executed"
exit "$1"
' bash {remote_status}
''')
    assert result.returncode == expected_status, result.stdout + result.stderr
    assert "remote-payload" not in result.stdout
    if transport_status == 0:
        assert (Path(cluster["TEST_ROOT"]) / "remote/executed").read_text() == "executed\n"
        assert "pbsdsh: spawned task" in result.stdout
    else:
        assert "failed to contact MOM" in result.stdout + result.stderr


def test_worker_launch_failure_aborts_startup(cluster: dict[str, str]) -> None:
    """Reject a remote worker-launch failure even when pbsdsh returns zero."""
    result = run_helper(cluster, '''HEAD_NODE=head
SCHED_ADDR=tcp://head:8786
HOST_COUNTS="1 remote"
start_workers
echo unexpected-success
''')
    assert result.returncode != 0
    assert "unexpected-success" not in result.stdout
    assert not (Path(cluster["SCRATCHDIR"]) / "DASK_EXPECTED_WORKERS.txt").exists()


def test_remote_arguments_remain_literal(cluster: dict[str, str]) -> None:
    """Preserve empty and multiline arguments without executing embedded shell syntax."""
    payload = 'spaces \' " \\ $SCRATCHDIR\n$(touch "$TEST_ROOT/injected") `touch "$TEST_ROOT/injected"`'
    cluster["TEST_PAYLOAD"] = payload
    result = run_helper(cluster, '''pbs_remote remote bash -c '
printf "%s\\0" "$@" > "$SCRATCHDIR/arguments"
' bash "$TEST_PAYLOAD" "" 'last argument'
''')
    assert result.returncode == 0, result.stdout + result.stderr
    root = Path(cluster["TEST_ROOT"])
    assert (root / "remote/arguments").read_bytes() == payload.encode() + b"\0\0last argument\0"
    assert not (root / "injected").exists()


@pytest.mark.parametrize("failure", ["transfer", "missing", "noop"])
def test_log_collection_failure_continues_cleanup(cluster: dict[str, str], failure: str) -> None:
    """Report failed log transfers while still collecting other nodes and retaining scratch."""
    if failure == "transfer":
        cluster["FAIL_LOGS"] = "head"
    elif failure == "noop":
        cluster["NOOP_LOGS"] = "head"
    cluster["LOG_FAILURE"] = failure
    result = run_helper(cluster, '''stage_inputs
for node in head remote; do echo "$node" > "$TEST_ROOT/$node/logs/worker.log"; done
if [[ "$LOG_FAILURE" == missing ]]; then rm -r "$TEST_ROOT/head/logs"; fi
stop_cluster
''')
    assert result.returncode != 0
    root = Path(cluster["TEST_ROOT"])
    if failure != "missing":
        assert (root / "head/logs/worker.log").exists()
    assert (root / "output/logs_remote/worker.log").read_text() == "remote\n"


@pytest.mark.parametrize("failure", ["none", "stage", "start", "driver", "collect", "stop", "missing", "noop"])
def test_pbs_job_status_and_cleanup(cluster: dict[str, str], failure: str) -> None:
    """Keep job failure status and retain scratch when cleanup cannot preserve outputs."""
    root = Path(cluster["TEST_ROOT"])
    if failure == "none":
        (root / "bin/rsync").unlink()
    app = root / "app"
    app.mkdir()
    wrapper = app / "dask_cluster_env.sh"
    wrapper.write_text('''#!/bin/bash
echo "$2" >> "$TEST_ROOT/events"
case "$2" in
  start)
    for node in head remote; do
      [[ -f "$TEST_ROOT/$node/input.txt" ]] || exit 21
      mkdir -p "$TEST_ROOT/$node/workdir"
      echo "$node" > "$TEST_ROOT/$node/workdir/sample.txt"
    done
    [[ "$FAILURE" != start ]] || exit 18
    ;;
  run)
    if [[ "$FAILURE" == missing ]]; then rm -r "$TEST_ROOT/remote/workdir"; fi
    [[ "$FAILURE" != driver ]] || exit 19
    ;;
  stop) [[ "$FAILURE" != stop ]] || exit 20 ;;
esac
''')
    tree = ast.parse((APP / "sensitivity_sampling.py").read_text())
    template = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "PBS_SCRIPT_TEMPLATE"
                            for target in node.targets))
    script = template.format(n_chunks=2, n_cpus=1, mem="1gb", scratch="1gb", walltime="00:01:00",
                             queue="test", job_name="test", out_log=root / "out.log",
                             outputdir=root / "output", script_path=app / "sampling.py", app_cmd="meta")
    # The temporary app uses the repository's real remote-command helper.
    script = script.replace('"$PROJECT_DIR/../../dask/pbs_remote.sh"',
                            f'"{REPO}/dask/pbs_remote.sh"')
    env = dict(cluster, FAILURE=failure)
    if failure == "stage":
        env["FAIL_STAGE"] = "remote"
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, timeout=10)
    assert (result.returncode == 0) == (failure == "none"), result.stdout + result.stderr
    events = (root / "events").read_text().splitlines()
    assert ("cleaned" in events) == (failure not in ("collect", "stop", "missing", "noop"))
    if failure == "stage":
        assert "start" not in events
    else:
        assert "stop" in events
        assert (root / "output/workdir_head/sample.txt").read_text() == "head\n"
        if failure not in ("collect", "missing", "noop"):
            assert (root / "output/workdir_remote/sample.txt").read_text() == "remote\n"
    if failure in ("start", "driver", "stage"):
        assert "FINISHED SAMPLING" not in result.stdout
    if failure == "driver":
        assert result.returncode == 19
