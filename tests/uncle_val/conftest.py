from contextlib import nullcontext

import pytest
from dask.distributed import Client


def get_dask_client(client):
    """Creates a context manager for the dask client.

    Parameters
    ----------
    client : str or None
        The client type to create. Can be:
        - "dask": Creates a Dask distributed client with 2 workers
        - None: Returns a null context (no client)

    Returns
    -------
    context manager
        A context manager that yields the client or None.

    Raises
    ------
    ValueError
        If an unsupported client type is provided.
    """
    if client == "dask":
        return Client(n_workers=2)
    if client is None:
        return nullcontext(None)
    raise ValueError(f"Unsupported client type: {client}")


@pytest.fixture(params=[None, "dask"])
def dask_client(request):
    """Pytest fixture that parametrizes tests to run with and without a Dask client.

    This fixture automatically runs the test twice:
    - Once with client=None (no distributed context)
    - Once with client="dask" (with a distributed client)

    Usage
    -----
    def test_something(dask_client):
        with dask_client as client:
            # Your test code here
            pass
    """
    with get_dask_client(request.param) as client:
        yield client
