import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Allows CI to verify that the integration tests import and compile."""
