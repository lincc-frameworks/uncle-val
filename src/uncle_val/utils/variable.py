from typing import Generic, TypeVar

import dask.distributed

T = TypeVar("T")


class Variable(Generic[T]):
    """Wrapper around Dask's Variable class to run without a client"""

    def __init__(self, value: T):
        self.value = value
        if self._within_distributed_context():
            self.variable = dask.distributed.Variable()
            self.variable.set(value)
        else:
            self.variable = None

    @staticmethod
    def _within_distributed_context() -> bool:
        try:
            dask.distributed.get_worker()
            return True
        except ValueError:
            pass
        try:
            dask.distributed.get_client()
            return True
        except ValueError:
            pass
        return False

    def get(self) -> T:
        """Get the value"""
        if self._within_distributed_context():
            if self.variable is None:
                raise RuntimeError(
                    "Variable is initialized without a Dask Client context, "
                    "but accessed within a Client context"
                )
            return self.variable.get()

        if self.variable is not None:
            raise RuntimeError(
                "Variable is initialized with a Dask Client context, but accessed without Client"
            )
        return self.value

    def set(self, value: T):
        """Set the value"""
        self.value = value

        if self._within_distributed_context():
            if self.variable is None:
                raise RuntimeError(
                    "Variable is initialized without a Dask Client context, "
                    "but accessed within a Client context"
                )
            self.variable.set(value)
            return

        if self.variable is not None:
            raise RuntimeError(
                "Variable is initialized with a Dask Client context, but accessed without Client"
            )
