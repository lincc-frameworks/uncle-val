import pytest
from dask.distributed import Client
from uncle_val.utils.variable import Variable


def test_variable_initialization(dask_client):
    """Test that Variable can be initialized with and without a Dask client."""
    value = 42
    var = Variable(value)

    # Value should be stored
    assert var.value == value

    # Dask variable should only be created within distributed context
    if dask_client is None:
        assert var.variable is None
    else:
        assert var.variable is not None


def test_variable_get(dask_client):
    """Test that Variable.get() works correctly in both contexts."""
    value = "test_value"
    var = Variable(value)

    # get() should return the correct value
    assert var.get() == value


def test_variable_set(dask_client):
    """Test that Variable.set() works correctly in both contexts."""
    initial_value = 10
    new_value = 20

    var = Variable(initial_value)
    assert var.get() == initial_value

    var.set(new_value)
    assert var.get() == new_value
    assert var.value == new_value


def test_variable_multiple_operations(dask_client):
    """Test multiple set and get operations."""
    # Use tuples to avoid list->tuple conversion in Dask
    var = Variable((1, 2, 3))
    assert var.get() == (1, 2, 3)

    var.set((4, 5, 6))
    assert var.get() == (4, 5, 6)

    var.set({"key": "value"})
    assert var.get() == {"key": "value"}


def test_variable_without_client_context():
    """Test Variable behavior when created and accessed without any client."""
    value = "no_client_value"
    var = Variable(value)

    assert var.value == value
    assert var.variable is None
    assert var.get() == value

    new_value = "updated_value"
    var.set(new_value)
    assert var.get() == new_value


def test_variable_with_client_context():
    """Test Variable behavior when created and accessed with a client."""
    with Client(n_workers=2):
        value = "client_value"
        var = Variable(value)

        assert var.value == value
        assert var.variable is not None
        assert var.get() == value

        new_value = "updated_client_value"
        var.set(new_value)
        assert var.get() == new_value


def test_variable_error_created_without_accessed_with_client():
    """Test error when Variable is created without client but accessed with one."""
    # Create variable without client
    var = Variable("value")
    assert var.variable is None

    # Try to access within client context
    with Client(n_workers=2):
        with pytest.raises(RuntimeError, match="initialized without a Dask Client context"):
            var.get()

        with pytest.raises(RuntimeError, match="initialized without a Dask Client context"):
            var.set("new_value")


def test_variable_error_created_with_accessed_without_client():
    """Test error when Variable is created with client but accessed without one."""
    # Create variable within client context
    with Client(n_workers=2):
        var = Variable("value")
        assert var.variable is not None

    # Try to access outside client context
    with pytest.raises(RuntimeError, match="initialized with a Dask Client context"):
        var.get()

    with pytest.raises(RuntimeError, match="initialized with a Dask Client context"):
        var.set("new_value")


def test_within_distributed_context():
    """Test the _within_distributed_context static method."""
    # Without any client
    assert Variable._within_distributed_context() is False

    # With a client
    with Client(n_workers=2):
        assert Variable._within_distributed_context() is True

    # After closing client
    assert Variable._within_distributed_context() is False


def test_variable_with_different_types():
    """Test Variable with various data types."""
    # Test values that work consistently with and without Dask
    test_values = [
        42,
        3.14,
        "string",
        {"a": 1, "b": 2},
        (1, 2, 3),
        None,
        True,
    ]

    for value in test_values:
        var = Variable(value)
        assert var.get() == value

        # Test with client context as well
        with Client(n_workers=2):
            var_with_client = Variable(value)
            assert var_with_client.get() == value
