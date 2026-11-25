from pathlib import Path

from candle.data import pipeline


def _clean_file(file: Path):
    if file.name == "boom.txt":
        raise ValueError("boom")
    if file.name == "wrong.txt":
        return 42
    return file.read_text()


def test_pipeline_result_initializer():
    # Allow no successes
    result = pipeline.PipelineResult(str, [], [("file", ValueError("boom"))])

    assert result.num_successes == 0
    assert result.num_failures == 1
    assert result.success_rate == 0.0

    # Allow empty
    result = pipeline.PipelineResult(str, [], [])

    assert result.num_successes == 0
    assert result.num_failures == 0
    assert result.success_rate == 0.0


def test_pipeline_result_record():
    result = pipeline.PipelineResult(str)

    result.record("a.txt", "a", None)
    result.record("boom.txt", None, ValueError("boom"))
    result.record("wrong.txt", 42, None)

    assert result.successes == ["a"]
    assert [item for item, _ in result.failures] == ["boom.txt", "wrong.txt"]
    assert isinstance(result.failures[0][1], ValueError)
    assert isinstance(result.failures[1][1], TypeError)
    assert result.num_successes == 1
    assert result.num_failures == 2
    assert result.success_rate == 1 / 3


def test_pipeline_dir_parallel_matches_sequential(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "boom.txt").write_text("boom")
    (tmp_path / "wrong.txt").write_text("wrong")

    sequential = pipeline.dir(_clean_file, str, num_workers=1)(tmp_path)
    parallel = pipeline.dir(_clean_file, str, num_workers=2)(tmp_path)

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
