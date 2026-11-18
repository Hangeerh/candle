from collections.abc import Callable


class Pipeline[T]:
    def __init__(self, target: type):
        self.target = target
        self.pipes: dict[str, Callable] = {}

    def register(self, name: str, pipe: Callable):
        if name in self.pipes:
            raise ValueError(f"Already registered {name}")

        self.pipes[name] = pipe

    def run(self, records, pipe: str):
        if not pipe in self.pipes:
            raise ValueError(f"Unknown pipe {pipe}")

        pipef = self.pipes[pipe]

        successes = []
        failures = []

        for i, raw in enumerate(records):
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
