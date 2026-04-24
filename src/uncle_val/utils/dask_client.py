from contextlib import suppress

from dask.distributed import Client as _DaskClient


class Client(_DaskClient):
    """Dask ``Client`` with project-standard defaults.

    Sets ``threads_per_worker=1`` by default, prints the dashboard URL on
    construction, and closes gracefully on context manager exit.

    Parameters
    ----------
    n_workers:
        Number of Dask workers to spawn.
    memory_limit:
        Memory limit per worker (passed to :class:`dask.distributed.Client`).
    close_timeout:
        Seconds to wait when closing the client on context manager exit.
        ``TimeoutError`` is suppressed if the timeout is exceeded.
    **kwargs:
        Forwarded to :class:`dask.distributed.Client`.

    Examples
    --------
    >>> with Client(n_workers=4) as client:  # doctest: +SKIP
    ...     result = my_dask_collection.compute()
    """

    def __init__(self, n_workers: int, memory_limit: str = "32GB", close_timeout: int = 60, **kwargs):
        kwargs.setdefault("threads_per_worker", 1)
        self._close_timeout = close_timeout
        super().__init__(n_workers=n_workers, memory_limit=memory_limit, **kwargs)

    @property
    def dashboard_link(self) -> str | None:
        """Dashboard URL, or ``None`` if unavailable."""
        try:
            return super().dashboard_link
        except (KeyError, AttributeError):
            return None

    def __exit__(self, *args):
        with suppress(TimeoutError):
            self.close(timeout=self._close_timeout)
