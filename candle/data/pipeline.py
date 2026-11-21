from collections.abc import Callable
from .utility import _norm_path
from pathlib import Path

__all__ = ["pipeline_dir"]


class PipelineResult:
    # TODO: This may seem redundant to make results a class, but I plan on adding
    #   features such as success rate statistics and such, so it is good.
    def __init__(self, successes: list, failures: list):
        self.successes = successes
        self.failures = failures


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

        if num_workers > 1:
            raise NotImplementedError("Not done yet!")

        successes = []
        failures = []

        for file in files:
            try:
                cleaned = clean_func(file)

                if not isinstance(cleaned, target_type):
                    failures.append(
                        (
                            file,
                            TypeError(
                                f"Expected type {target_type}, got type {type(cleaned)}"
                            ),
                        )
                    )
                    continue

                successes.append(cleaned)

            except Exception as e:
                failures.append((file, e))

        return PipelineResult(successes, failures)

    return wrapped_clean


class Pipeline[T]:
    def __init__(self, target: type):
        self.target = target
        self.pipes: dict[str, Callable] = {}

    def register(self, name: str, pipe: Callable):
        if name in self.pipes:
            raise ValueError(f"Already registered {name}")

        self.pipes[name] = pipe

    def run(self, records: list, pipe: str):
        if not pipe in self.pipes:
            raise ValueError(f"Unknown pipe {pipe}")

        pipef = self.pipes[pipe]

        successes = []
        failures = []

        for raw in records:
            try:
                result = pipef(raw)

                if not isinstance(result, self.target):
                    raise TypeError(
                        f"Expected {self.target.__name__}, got {type(result).__name__}"
                    )

                successes.append(result)

            except:
                failures.append(raw)

        return successes, failures
