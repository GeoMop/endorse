import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chodby_trans import collect_sampling_jobs as collect


class FakeArray:
    def __init__(self, data: np.ndarray):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, value):
        self.data[item] = value


def _store_arrays(tag: int, sample_range: tuple[int, int]) -> dict[str, np.ndarray]:
    start, stop = sample_range
    span = stop - start
    arrays = {
        "i_eval": np.zeros((4, 2), dtype=int),
        "return_code": np.zeros((4, 2), dtype=int),
        "eval_time": np.full((4, 2), np.nan),
        "parameter": np.full((4, 2, 2), np.nan),
        "conc": np.full((4, 2, 2), np.nan),
        "param_name": np.array(["a", "b"], dtype=object),
    }
    arrays["i_eval"][start:stop, :] = np.arange(tag * 10, tag * 10 + span * 2).reshape(span, 2)
    arrays["return_code"][start:stop, :] = tag
    arrays["eval_time"][start:stop, :] = float(tag)
    arrays["parameter"][start:stop, :, :] = float(tag)
    arrays["conc"][start:stop, :, :] = float(tag)
    return arrays


def test_list_job_dirs_filters_sampling_dirs(tmp_path: Path) -> None:
    case_dir = tmp_path / "CASE_0_32k"
    (case_dir / "job_00" / "transport_sampling").mkdir(parents=True)
    (case_dir / "job_01" / "input_data").mkdir(parents=True)
    (case_dir / "notes").mkdir(parents=True)

    assert collect.list_job_dirs(case_dir) == [case_dir / "job_00"]


def test_collect_job_stores_merges_i_sample_ranges(tmp_path: Path, monkeypatch) -> None:
    case_dir = tmp_path / "CASE_0_32k"
    job0 = case_dir / "job_00_00000_00002"
    job1 = case_dir / "job_01_00002_00004"
    output_store = case_dir / "transport_sampling"

    stores = {
        job0 / "transport_sampling": _store_arrays(tag=7, sample_range=(0, 2)),
        job1 / "transport_sampling": _store_arrays(tag=9, sample_range=(2, 4)),
    }
    sample_ranges = {job0: (0, 2), job1: (2, 4)}

    def fake_copy_template(source_store: Path, dest_store: Path, force: bool) -> None:
        stores[dest_store] = {name: data.copy() for name, data in stores[source_store].items()}

    def fake_open(path: Path, mode: str):
        return FakeArray(stores[path.parent][path.name])

    monkeypatch.setattr(collect, "list_job_dirs", lambda _: [job0, job1])
    monkeypatch.setattr(collect, "_merge_array_names", lambda _: ["i_eval", "return_code", "eval_time", "parameter", "conc"])
    monkeypatch.setattr(collect, "_copy_template_store", fake_copy_template)
    monkeypatch.setattr(collect, "_validate_store_layout", lambda *_: None)
    monkeypatch.setattr(collect, "_job_sample_range", lambda job_dir: sample_ranges[job_dir])
    monkeypatch.setattr(collect.zarr, "open", fake_open)

    merged_store = collect.collect_job_stores(case_dir)

    assert merged_store == output_store.resolve()
    np.testing.assert_array_equal(stores[output_store]["return_code"][:2], np.full((2, 2), 7))
    np.testing.assert_array_equal(stores[output_store]["return_code"][2:], np.full((2, 2), 9))
    np.testing.assert_array_equal(stores[output_store]["i_eval"][:2], np.array([[70, 71], [72, 73]]))
    np.testing.assert_array_equal(stores[output_store]["i_eval"][2:], np.array([[90, 91], [92, 93]]))
    np.testing.assert_array_equal(stores[output_store]["param_name"], np.array(["a", "b"], dtype=object))


def test_count_samples_by_rc_filters_to_job_limit_samples(tmp_path: Path, monkeypatch) -> None:
    job_dir = tmp_path / "job_00_00000_00500"
    input_data = job_dir / "input_data"
    input_data.mkdir(parents=True)
    (input_data / "_ot_sensitivity.yaml").write_text(
        "limit_samples: [0, 500]\n",
        encoding="utf-8",
    )

    fake_job = SimpleNamespace(set_workdir=lambda _workdir: None)
    fake_sampling = SimpleNamespace(
        read_parameters_by_rc=lambda _codes, make_plots=False: (
            [
                (10, 10, 0),
                (11, 499, 1),
                (12, 500, 0),
                (13, 700, 1),
            ],
            None,
            {},
        )
    )
    monkeypatch.setitem(sys.modules, "chodby_trans.job", fake_job)
    monkeypatch.setitem(sys.modules, "chodby_trans.sensitivity_sampling", fake_sampling)

    assert collect.count_samples_by_rc(job_dir, [-2000]) == 2


def test_write_job_return_code_report_writes_summary(tmp_path: Path, monkeypatch) -> None:
    case_dir = tmp_path / "CASE_0_32k"
    job0 = case_dir / "job_00_00000_00500"
    job1 = case_dir / "job_01_00500_01000"
    report_path = case_dir / "job_return_code_stats.txt"

    monkeypatch.setattr(collect, "list_job_dirs", lambda _: [job0, job1])
    monkeypatch.setattr(
        collect,
        "job_return_code_stats",
        lambda job_dir: {
            "job_dir": job_dir,
            "limit_samples": (0, 500) if job_dir == job0 else (500, 1000),
            "limit_range_size": 500,
            "n_results_non_none": 6990 if job_dir == job0 else 7000,
            "counts": {-2000: 10 if job_dir == job0 else 0, 0: 6990 if job_dir == job0 else 7000},
        },
    )

    written_path = collect.write_job_return_code_report(case_dir, output_path=report_path)
    text = report_path.read_text(encoding="utf-8")

    assert written_path == report_path.resolve()
    assert "Case:" in text
    assert "job_00_00000_00500" in text
    assert "limit_range_size: 500" in text
    assert "n_results_non_none: 6990" in text
    assert "NONE [-2000]: 10" in text
    assert "OK [0]: 7000" in text


def test_job_return_code_stats_reads_storage_once(tmp_path: Path, monkeypatch) -> None:
    case_dir = tmp_path / "CASE_0_32k"
    job_dir = case_dir / "job_00_00000_00500"
    input_data = job_dir / "input_data"
    input_data.mkdir(parents=True)
    (input_data / "_ot_sensitivity.yaml").write_text(
        "limit_samples: [0, 500]\n",
        encoding="utf-8",
    )

    calls = []
    fake_job = SimpleNamespace(set_workdir=lambda _workdir: None)

    def fake_read(codes, make_plots=False):
        calls.append((tuple(codes), make_plots))
        return (
            [
                (10, 10, 0),
                (11, 499, 1),
                (12, 500, 0),
                (13, 700, 1),
            ],
            None,
            {
                -2000: np.array([10, 12, 13]),
                0: np.array([11]),
            },
        )

    fake_sampling = SimpleNamespace(read_parameters_by_rc=fake_read)
    monkeypatch.setitem(sys.modules, "chodby_trans.job", fake_job)
    monkeypatch.setitem(sys.modules, "chodby_trans.sensitivity_sampling", fake_sampling)

    stats = collect.job_return_code_stats(job_dir)

    assert calls == [((-2000, -1999, -1100, -1020, -1010, -1003, -1002, -1001, -1000, 0), False)]
    assert stats["counts"][-2000] == 1
    assert stats["counts"][0] == 1
