from collections.abc import Callable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from candle.data.utility import _norm_path
from pathlib import Path

__all__ = ["Directory", "Cleaner"]


class PipelineResult:
    def __init__(
        self,
        target_type: type,
        successes: list | None = None,
        failures: list | None = None,
    ):
        self.target_type = target_type
        self.successes = successes if successes is not None else []
        self.failures = failures if failures is not None else []

    @property
    def num_successes(self) -> int:
        return len(self.successes)

    @property
    def num_failures(self) -> int:
        return len(self.failures)

    @property
    def success_rate(self) -> float:
        total = self.num_successes + self.num_failures
        return self.num_successes / total if total else 0.0

    def record(self, item, cleaned, error) -> None:
        if error is not None:
            self.failures.append((item, error))
        elif not isinstance(cleaned, self.target_type):
            self.failures.append(
                (
                    item,
                    TypeError(
                        f"Expected type {self.target_type}, got type {type(cleaned)}"
                    ),
                )
            )
        else:
            self.successes.append(cleaned)

    def __str__(self) -> str:
        return (
            f"PipelineResult("
            f"num_successes={self.num_successes}, "
            f"num_failures={self.num_failures}, "
            f"success_rate={self.success_rate})"
        )


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


def _clean_one_instance(clean_func: Callable, *args, **kwargs):
    """Run ``clean_func`` on one file, capturing any exception it raises."""
    try:
        return clean_func(*args, **kwargs), None
    except Exception as e:
        return None, e


def Cleaner(clean: Callable, target_type: type, num_workers: int = 0) -> Callable:
    def wrapped_clean(groups: list) -> PipelineResult:
        result = PipelineResult(target_type)

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = executor.map(
                    partial(_clean_one_instance, clean),
                    groups,
                )

                for group, (cleaned, error) in zip(groups, results):
                    result.record(group, cleaned, error)
        else:
            for group in groups:
                cleaned, error = _clean_one_instance(clean, group)
                result.record(group, cleaned, error)

        return result

    return wrapped_clean
