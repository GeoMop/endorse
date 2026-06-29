from pathlib import Path

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
