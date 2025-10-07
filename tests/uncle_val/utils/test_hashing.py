import numpy as np
from uncle_val.utils.hashing import uniform_hash


def test_output_range():
    """Test that output values are in [0, 1) range"""
    data = np.array([1, 2, 3, 4, 5])
    result = uniform_hash(data)
    assert np.all(result >= 0.0), "All values should be >= 0"
    assert np.all(result < 1.0), "All values should be < 1"


def test_deterministic():
    """Test that hashing is deterministic"""
    data = np.array([1, 2, 3, 4, 5])
    result1 = uniform_hash(data)
    result2 = uniform_hash(data)
    np.testing.assert_array_equal(result1, result2)


def test_different_values_different_hashes():
    """Test that different values produce different hashes"""
    data1 = np.array([1, 2, 3])
    data2 = np.array([1, 2, 4])
    result1 = uniform_hash(data1)
    result2 = uniform_hash(data2)
    # At least one value should be different
    assert not np.allclose(result1, result2)


def test_dtype_int64():
    """Test with int64 dtype"""
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = uniform_hash(data)
    assert result.shape == data.shape
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_dtype_int32():
    """Test with int32 dtype"""
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = uniform_hash(data)
    assert result.shape == data.shape
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_dtype_uint32():
    """Test with uint32 dtype"""
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    result = uniform_hash(data)
    assert result.shape == data.shape
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_dtype_float32():
    """Test with float32 dtype"""
    data = np.array([1.5, 2.3, 3.7, 4.2, 5.9], dtype=np.float32)
    result = uniform_hash(data)
    assert result.shape == data.shape
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_dtype_float64():
    """Test with float64 dtype"""
    data = np.array([1.5, 2.3, 3.7, 4.2, 5.9], dtype=np.float64)
    result = uniform_hash(data)
    assert result.shape == data.shape
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_dtype_bytes():
    """Test with byte string dtype (S|)"""
    data = np.array([b"hello", b"world", b"test"], dtype="S10")
    result = uniform_hash(data)
    assert result.shape == data.shape
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_dtype_unicode():
    """Test with unicode string dtype (U|)"""
    data = np.array(["hello", "world", "test"], dtype="U10")
    result = uniform_hash(data)
    assert result.shape == data.shape
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_scalar():
    """Test with 0-dimensional array (scalar)"""
    data = np.array(42)
    result = uniform_hash(data)
    assert result.shape == ()  # 0-dimensional
    assert result.dtype == np.float64
    assert 0.0 <= result < 1.0


def test_scalar_float():
    """Test with scalar float"""
    data = np.array(3.14159)
    result = uniform_hash(data)
    assert result.shape == ()
    assert result.dtype == np.float64
    assert 0.0 <= result < 1.0


def test_1d_array():
    """Test with 1-dimensional array"""
    data = np.array([1, 2, 3, 4, 5])
    result = uniform_hash(data)
    assert result.shape == (5,)
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_2d_array():
    """Test with 2-dimensional array"""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    result = uniform_hash(data)
    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_3d_array():
    """Test with 3-dimensional array"""
    data = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    result = uniform_hash(data)
    assert result.shape == (2, 2, 2)
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))
    assert np.all(result[0][0] == result)


def test_empty_array():
    """Test with empty array"""
    data = np.array([])
    result = uniform_hash(data)
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_single_element_array():
    """Test with single element array"""
    data = np.array([42])
    result = uniform_hash(data)
    assert result.shape == (1,)
    assert result.dtype == np.float64
    assert 0.0 <= result[0] < 1.0


def test_dtype_affects_hash():
    """Test that dtype affects the hash value"""
    value = 100
    data_int32 = np.array([value], dtype=np.int32)
    data_int64 = np.array([value], dtype=np.int64)
    result_int32 = uniform_hash(data_int32)
    result_int64 = uniform_hash(data_int64)
    # Different byte representations should produce different hashes
    assert result_int32[0] != result_int64[0]


def test_list_input():
    """Test that list input is accepted and converted"""
    data = [1, 2, 3, 4, 5]
    result = uniform_hash(data)
    assert result.shape == (5,)
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))


def test_tuple_input():
    """Test that tuple input is accepted and converted"""
    data = (1, 2, 3)
    result = uniform_hash(data)
    assert result.shape == (3,)
    assert result.dtype == np.float64
    assert np.all((result >= 0.0) & (result < 1.0))
