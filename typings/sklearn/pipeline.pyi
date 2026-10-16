# A step is any object with fit (and transform, for every step but the last); sklearn checks
# at fit time.
class Pipeline:
    steps: list[tuple[str, object]]
    def __init__(self, steps: list[tuple[str, object]], *, verbose: bool = False) -> None: ...
    @property
    def named_steps(self) -> dict[str, object]: ...
