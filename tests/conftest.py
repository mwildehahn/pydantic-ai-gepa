import asyncio
from collections.abc import Iterator
import pytest


@pytest.fixture(scope='session')
def anyio_backend():
    return 'asyncio'


@pytest.fixture(scope='session', autouse=True)
def event_loop() -> Iterator[None]:
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    yield
    new_loop.close()
