from pathlib import Path

from candle.data.pipeline import pipeline_dir


def _clean_file(file: Path):
    if file.name == "boom.txt":
        raise ValueError("boom")
    if file.name == "wrong.txt":
        return 42
    return file.read_text()


def test_pipeline_dir_parallel_matches_sequential(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "boom.txt").write_text("boom")
    (tmp_path / "wrong.txt").write_text("wrong")

    sequential = pipeline_dir(_clean_file, str, num_workers=1)(tmp_path)
    parallel = pipeline_dir(_clean_file, str, num_workers=2)(tmp_path)

    assert sorted(sequential.successes) == sorted(parallel.successes) == ["a", "b"]
    assert {path.name for path, _ in sequential.failures} == {
        "boom.txt",
        "wrong.txt",
    }
    assert {path.name for path, _ in parallel.failures} == {
        "boom.txt",
        "wrong.txt",
    }
    assert {type(error) for _, error in parallel.failures} == {
        ValueError,
        TypeError,
    }
