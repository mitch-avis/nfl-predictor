"""The typed stand-ins tests swap in with ``monkeypatch.setattr``."""

from tests.fakes import returning


def test_returning_ignores_every_argument_and_returns_the_value() -> None:
    # Arrange
    stub = returning("model")

    # Act
    result = stub(1, "two", three=3)

    # Assert
    assert result == "model"


def test_returning_hands_back_the_same_object_on_every_call() -> None:
    # Arrange
    value: list[int] = []
    stub = returning(value)

    # Act
    results = (stub(), stub("again"))

    # Assert
    assert results[0] is value
    assert results[1] is value
