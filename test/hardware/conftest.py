import pytest

from qscope.util.system_check import check_systems


@pytest.fixture(scope="session")
def available_systems():
    """Get dictionary of available systems and their status."""
    return check_systems()


def requires_system(system_name):
    """Decorator to skip test if required system is not available."""

    def decorator(func):
        @pytest.mark.usefixtures("available_systems")
        def wrapper(available_systems, *args, **kwargs):
            if (
                system_name not in available_systems
                or not available_systems[system_name][0]
            ):
                pytest.skip(f"System {system_name} not available")
            return func(*args, **kwargs)

        return wrapper

    return decorator
