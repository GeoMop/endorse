import sys
from pathlib import Path
from types import SimpleNamespace

from chodby_trans import prepare_sampling_jobs as prep


def test_prepare_job_dirs_chunks_input_data_and_rewrites_limit_samples(tmp_path: Path, monkeypatch):
    case_dir = tmp_path / "CASE_0_32k"
    input_data = case_dir / "input_data"
    input_data.mkdir(parents=True)
    (input_data / "payload.txt").write_text("payload", encoding="utf-8")
    (input_data / "_ot_sensitivity.yaml").write_text(
        "n_samples: 1200\n"
        "limit_samples: [0, 1200]\n"
        "# keep me\n",
        encoding="utf-8",
    )
    (input_data / "trans_mesh_config.yaml").write_text(
        "machine_config:\n"
        "  __default__:\n"
        "    pbs:\n"
        "      pbs_name: trans_case_0_base\n"
        "# keep mesh comment\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, cwd, check):
        calls.append((cmd, cwd, check))
        return None

    monkeypatch.setattr(prep.subprocess, "run", fake_run)

    job_dirs = prep.prepare_job_dirs(case_dir, samples_per_job=500, jobs_root=tmp_path / "jobs")
    assert [job_dir.name for job_dir in job_dirs] == [
        "job_00_00000_00500",
        "job_01_00500_01000",
        "job_02_01000_01200",
    ]

    expected_ranges = [(0, 500), (500, 1000), (1000, 1200)]
    for job_index, (job_dir, sample_range) in enumerate(zip(job_dirs, expected_ranges)):
        copied_cfg_text = (job_dir / "input_data" / "_ot_sensitivity.yaml").read_text(encoding="utf-8")
        assert f"limit_samples: [{sample_range[0]}, {sample_range[1]}]" in copied_cfg_text
        assert "# keep me" in copied_cfg_text
        mesh_cfg_text = (job_dir / "input_data" / "trans_mesh_config.yaml").read_text(encoding="utf-8")
        assert f"pbs_name: trans_case_0_base_{job_index:02d}" in mesh_cfg_text
        assert "# keep mesh comment" in mesh_cfg_text
        assert (job_dir / "input_data" / "payload.txt").read_text(encoding="utf-8") == "payload"

    prep.run_job(job_dirs[0])
    assert calls
    cmd, cwd, check = calls[0]
    assert cmd[1:] == ["sensitivity_sampling.py", str(job_dirs[0]), "submit", "meta"]
    assert cwd == Path(prep.__file__).resolve().parent
    assert check is True


def test_prepare_job_dirs_reuses_existing_job_dir(tmp_path: Path):
    case_dir = tmp_path / "CASE_0_32k"
    input_data = case_dir / "input_data"
    input_data.mkdir(parents=True)
    (input_data / "_ot_sensitivity.yaml").write_text(
        "n_samples: 600\n"
        "limit_samples: [0, 600]\n",
        encoding="utf-8",
    )
    (input_data / "trans_mesh_config.yaml").write_text(
        "pbs_name: trans_case_0\n",
        encoding="utf-8",
    )
    existing_job = case_dir / "job_00_00000_00500"
    existing_job.mkdir(parents=True, exist_ok=True)
    (existing_job / "marker.txt").write_text("keep", encoding="utf-8")

    job_dirs = prep.prepare_job_dirs(case_dir, samples_per_job=500)

    assert job_dirs[0] == existing_job
    assert (existing_job / "marker.txt").exists()


def test_archive_job_submission_files_renames_present_files(tmp_path: Path):
    job_dir = tmp_path / "job_00_00000_00500"
    job_dir.mkdir()
    (job_dir / "logs.tar.gz").write_text("logs", encoding="utf-8")
    (job_dir / "sensitivity_sampling.pbs").write_text("pbs", encoding="utf-8")
    (job_dir / "trans_case_0_00.out").write_text("out", encoding="utf-8")

    archived = prep.archive_job_submission_files(job_dir)

    assert [path.name for path in archived] == [
        "logs.tar.gz.rerun_01",
        "sensitivity_sampling.pbs.rerun_01",
        "trans_case_0_00.out.rerun_01",
    ]
    assert not (job_dir / "logs.tar.gz").exists()
    assert not (job_dir / "sensitivity_sampling.pbs").exists()
    assert not (job_dir / "trans_case_0_00.out").exists()


def test_rerun_incomplete_jobs_submits_continue_only_for_jobs_with_none(tmp_path: Path, monkeypatch):
    job_a = tmp_path / "job_00_00000_00500"
    job_b = tmp_path / "job_01_00500_01000"
    job_a.mkdir()
    job_b.mkdir()

    archived = []
    submitted = []

    monkeypatch.setattr(prep, "count_none_samples", lambda job_dir: 3 if job_dir == job_a else 0)
    monkeypatch.setattr(prep, "archive_job_submission_files", lambda job_dir: archived.append(job_dir) or [])
    monkeypatch.setattr(prep, "run_job", lambda job_dir, app_cmd="meta": submitted.append((job_dir, app_cmd)))

    rerun_jobs = prep.rerun_incomplete_jobs([job_a, job_b])

    assert rerun_jobs == [job_a]
    assert archived == [job_a]
    assert submitted == [(job_a, "continue")]


def test_count_none_samples_filters_to_job_limit_samples(tmp_path: Path, monkeypatch):
    job_dir = tmp_path / "job_00_00000_00500"
    input_data = job_dir / "input_data"
    input_data.mkdir(parents=True)
    (input_data / "_ot_sensitivity.yaml").write_text(
        "limit_samples: [0, 500]\n",
        encoding="utf-8",
    )

    fake_job = SimpleNamespace(
        output=SimpleNamespace(plots=job_dir / "plots"),
        set_workdir=lambda _workdir: None,
    )
    fake_sampling = SimpleNamespace(
        read_parameters_by_rc=lambda _codes, make_plots=False: (
            [
                (10, 10, 0),
                (11, 499, 1),
                (12, 500, 0),
                (13, 700, 1),
            ],
            None,
        )
    )

    monkeypatch.setitem(sys.modules, "chodby_trans.job", fake_job)
    monkeypatch.setitem(sys.modules, "chodby_trans.sensitivity_sampling", fake_sampling)

    assert prep.count_none_samples(job_dir) == 2
