import ttnn
import pytest


def with_device(func):
    def wrapper(*args, **kwargs):
        device = ttnn.open_device(device_id=0)
        try:
            result = func(device, *args, **kwargs)
        finally:
            ttnn.close_device(device)
        return result

    return wrapper


@pytest.fixture(scope="session")
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="session")
def compiler_options(device):
    yield ttnn.TorchTtnnOption(device=device)
