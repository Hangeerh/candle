from collections.abc import Callable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from candle.data.utility import _norm_path
from pathlib import Path

__all__ = ["dir"]


class PipelineResult:
    def __init__(self, successes: list, failures: list):
        if len(successes) < 1:
            raise ValueError(f"Empty list passed for argument successes")

        self.successes = successes
        self.failures = failures

        self.num_successes = len(successes)
        self.num_failures = len(failures)

        self.success_rate = self.num_successes / (
            self.num_successes + self.num_failures
        )

    def __str__(self) -> str:
        return (
            f"PipelineResult("
            f"num_successes={self.num_successes}, "
            f"num_failures={self.num_failures}, "
            f"success_rate={self.success_rate})"
        )


def _clean_one_instance(clean_func: Callable, *args, **kwargs):
    """Run ``clean_func`` on one file, capturing any exception it raises."""
    try:
        return clean_func(*args, **kwargs), None
    except Exception as e:
        return None, e


def Directory(filter: Callable[..., bool], group: Callable[..., str]) -> Callable:
    def wrapped_gather(directory: str | Path, recursive: bool) -> list:
        cwd = _norm_path(directory)
        if not cwd.is_dir():
            raise ValueError(f"{directory} must be a directory")

        pattern = "**/*" if recursive else "*"
        files = [
            file for file in cwd.glob(pattern) if (file.is_file() and filter(file))
        ]

        # The group function assigns ids to files. To make two files go to the same
        # group, assign the same id to them.
        grouped_samples = defaultdict(list)
        for file in files:
            grouped_samples[group(file)].append(file)

        return [member for member in grouped_samples.values()]

    return wrapped_gather


def dir(
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


# TODO: finish implementation
def tar(clean_func: Callable, target_type: type, num_workers: int = 0):
    def wrapped_clean(tar_path: str | Path) -> PipelineResult:
        successes = []
        failures = []

        tarball = _norm_path(tar_path)

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

        return PipelineResult(successes, failures)

    return wrapped_clean
