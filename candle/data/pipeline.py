from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from .utility import _norm_path
from pathlib import Path

__all__ = ["pipeline_dir"]


class PipelineResult:
    # TODO: This may seem redundant to make results a class, but I plan on adding
    #   features such as success rate statistics and such, so it is good.
    def __init__(self, successes: list, failures: list):
        self.successes = successes
        self.failures = failures


def _clean_one_instance(clean_func: Callable, file: Path):
    """Run ``clean_func`` on one file, capturing any exception it raises."""
    try:
        return clean_func(file), None
    except Exception as e:
        return None, e


def pipeline_dir(
    clean_func: Callable,
    target_type: type,
    num_workers: int = 0,
):
    def wrapped_clean(
        working_dir: str | Path, recursive: bool = False
    ) -> PipelineResult:
        cwd = _norm_path(working_dir)
        assert cwd.is_dir()

        pattern = "**/*" if recursive else "*"
        files = [f for f in cwd.glob(pattern) if f.is_file()]

        successes = []
        failures = []

        def record_result(file, cleaned, error):
            if error is not None:
                failures.append((file, error))
            elif not isinstance(cleaned, target_type):
                failures.append(
                    (
                        file,
                        TypeError(
                            f"Expected type {target_type}, got type {type(cleaned)}"
                        ),
                    )
                )
            else:
                successes.append(cleaned)

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = executor.map(
                    partial(_clean_one_instance, clean_func),
                    files,
                )
                for file, (cleaned, error) in zip(files, results):
                    record_result(file, cleaned, error)
        else:
            for file in files:
                cleaned, error = _clean_one_instance(clean_func, file)
                record_result(file, cleaned, error)

        return PipelineResult(successes, failures)

    return wrapped_clean
