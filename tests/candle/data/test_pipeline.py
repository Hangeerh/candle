from candle.data import pipeline


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
