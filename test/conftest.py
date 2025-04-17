import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks test as slow running test")
    config.addinivalue_line(
        "markers", "hardware: marks test that require physical hardware"
    )
