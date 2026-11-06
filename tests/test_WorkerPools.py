"""Unit tests for WorkerPools.py."""
import time
from concurrent.futures.process import BrokenProcessPool

import pytest

from WorkerPools import LazyProcessPool


@pytest.fixture
def makePool():
    """Creates pools that are all killed once the test ends."""
    pools = []

    def make(size) -> LazyProcessPool:
        pools.append(LazyProcessPool(size=size))
        return pools[-1]

    yield make
    for pool in pools:
        pool.kill()


def test_starts_on_first_submit(makePool):
    """No pool runs until a job is submitted, and later jobs reuse it."""
    pool = makePool(2)

    assert pool._pool is None
    assert pool.submit(abs, -3).result() == 3
    started = pool._pool
    assert pool.submit(abs, -4).result() == 4
    assert pool._pool is started


def test_drop_restarts_at_the_new_size(makePool):
    """After dropping, the next job starts a pool of whatever size the size function now returns."""
    size = 1
    pool = makePool(lambda: size)
    pool.submit(abs, 0).result()
    assert pool._pool._max_workers == 1

    size = 2
    pool.drop()
    pool.submit(abs, 0).result()
    assert pool._pool._max_workers == 2


def test_kill_fails_running_jobs(makePool):
    """Killing fails running jobs with BrokenProcessPool, and the next job starts a new pool."""
    pool = makePool(1)
    running = pool.submit(time.sleep, 30)
    time.sleep(0.5)

    pool.kill()

    with pytest.raises(BrokenProcessPool):
        running.result(timeout=10)
    assert pool.submit(abs, -5).result() == 5


def test_workers_cannot_submit(makePool):
    """A worker submitting a job raises instead of starting a pool nested in it."""
    pool = makePool(1)
    nested = pool.submit(exec, "from WorkerPools import FRAME_POOL; FRAME_POOL.submit(abs, -6)")

    with pytest.raises(RuntimeError, match="Only the main process"):
        nested.result(timeout=30)
