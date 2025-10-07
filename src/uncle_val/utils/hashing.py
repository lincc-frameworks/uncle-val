import numpy as np
from farmhash import Fingerprint32
from numpy.typing import ArrayLike, NDArray

FLOAT_TWO_POWER_32 = float(2**32)


def uniform_hash(a: ArrayLike) -> NDArray[float]:
    """Hash array values to the [0; 1) range.

    We use FarmHash's 32-bit Fingerprint for that.

    Parameters
    ----------
    a : array_like
        Array to be hashed.

    Returns
    -------
    np.ndarray of float
        Array of hashed values, the same shape as `a`.
    """
    a = np.asanyarray(a)

    # Handle scalar (0-dimensional) arrays
    is_scalar = a.ndim == 0
    if is_scalar:
        a = a.reshape(1)

    # Handle empty arrays
    if a.size == 0:
        return np.array([], dtype=np.float64)

    byte_view = a.view(np.uint8).reshape(*a.shape, -1)
    int32_hashes = np.apply_along_axis(Fingerprint32, -1, byte_view)
    float_hashes = int32_hashes / FLOAT_TWO_POWER_32

    return float_hashes[0] if is_scalar else float_hashes
