AB_ZP_NJY = 8.9 + 9 * 2.5


def mag2flux(mag):
    """AB magnitude to flux density in nJy convertion

    Parameters
    ----------
    mag : array-like
        Input AB magnitude

    Returns
    -------
    flux : array-like
        Flux density in nJy
    """
    return 10 ** (-0.4 * (mag - AB_ZP_NJY))
