import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context, parent_process
from typing import Any

from Globals import g

# Any child process forked _directly_ from the parent process inherits the
# thread state of the parent, but none of the threads!  This risks deadlocks
# on child processes.
#
# "forkserver" solves this problem by always forking from a clean slate. It's
# already the default on Python 3.14+ on Linux, so this just makes us
# future-proof.
PROCESS_CONTEXT = get_context("forkserver" if sys.platform == "linux" else None)

class LazyProcessPool:
    """A process pool that only starts once a job is submitted to it.

    It also can only start a pool from the root process, which avoids the issue
    of nested process pools that get created accidentally and do no work.
    They're relatively harmless but leak semaphores.
    """

    def __init__(self, size: int | Callable[[], int]) -> None:
        """Creates the pool without starting it.

        Args:
            size: The number of workers, or a function returning it, called each time the
                pool starts.
        """
        self._size = size
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        """Submits a job to a worker, starting the pool first if it isn't running.

        Args:
            fn: The function the worker runs, which must be picklable.
            *args: Its positional arguments.
            **kwargs: Its keyword arguments.

        Returns:
            The job's future.

        Raises:
            RuntimeError: If called from a child process, which would start a pool nested in it.
        """
        if(parent_process() is not None):
            raise RuntimeError("Only the main process can submit jobs to a pool")
        with self._lock:
            if(self._pool is None):
                size = self._size() if callable(self._size) else self._size
                self._pool = ProcessPoolExecutor(size, mp_context=PROCESS_CONTEXT)
            return self._pool.submit(fn, *args, **kwargs)

    def drop(self) -> None:
        """Gracefully stop all workers and "forget" the pool. submit() lazily creates a new pool."""
        with self._lock:
            if(self._pool is not None):
                self._pool.shutdown(wait=False)
                self._pool = None

    def kill(self) -> None:
        """Forcefully kill all workers and "forget" the pool. submit() lazily creates a new pool."""
        with self._lock:
            if(self._pool is not None):
                # TODO: Replace with self._pool.kill_workers() once we're on Python 3.14+.
                for process in self._pool._processes.values():
                    process.kill()
                self._pool = None

# This is the main pool that performs loading, aligning, stacking, etc.  It's
# size is controlled by the "CPUs" setting in the UI.
FRAME_POOL = LazyProcessPool(size=lambda: g.nThreads)

# Separate from FRAME_POOL since its workers keep state between jobs: each one
# keeps one color channel's wavelet coefficients, so slider changes only send
# the settings. That needs exactly three workers, and those workers need to
# persist for the lifetime of the application.
CHANNEL_POOL = LazyProcessPool(size=3)
