from pathlib import Path

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

    calls = []

    def fake_run(cmd, cwd, check):
        calls.append((cmd, cwd, check))
        return None

    monkeypatch.setattr(prep.subprocess, "run", fake_run)

    job_dirs = prep.prepare_job_dirs(case_dir, samples_per_job=500, jobs_root=tmp_path / "jobs")
    assert [job_dir.name for job_dir in job_dirs] == [
        "job_000_00000_00500",
        "job_001_00500_01000",
        "job_002_01000_01200",
    ]

    expected_ranges = [(0, 500), (500, 1000), (1000, 1200)]
    for job_dir, sample_range in zip(job_dirs, expected_ranges):
        copied_cfg_text = (job_dir / "input_data" / "_ot_sensitivity.yaml").read_text(encoding="utf-8")
        assert f"limit_samples: [{sample_range[0]}, {sample_range[1]}]" in copied_cfg_text
        assert "# keep me" in copied_cfg_text
        assert (job_dir / "input_data" / "payload.txt").read_text(encoding="utf-8") == "payload"

    prep.run_job(job_dirs[0])
    assert calls
    cmd, cwd, check = calls[0]
    assert cmd[1:] == ["sensitivity_sampling.py", str(job_dirs[0]), "submit", "meta"]
    assert cwd == Path(prep.__file__).resolve().parent
    assert check is True
