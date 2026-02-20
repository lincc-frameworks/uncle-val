import math

AB_ZP_NJY = 8.9 + 9 * 2.5
LN10_0_4 = math.log(10) * 0.4
LG_E_2_5 = 2.5 / math.log(10)


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


def fluxerr2magerr(*, flux, flux_err):
    """Flux error to magnitude error convertion

    Parameters
    ----------
    flux : array-like
        Flux density
    flux_err : array-like
        Flux error, same units and shape as flux

    Returns
    -------
    mag_err : array-like
        Magnitude error obtained by the error propagation formula.
    """
    return LG_E_2_5 * flux_err / flux


def magerr2fluxerr(*, flux, mag_err):
    """Magnitude error to flux error convertion

    Parameters
    ----------
    flux : array-like
        Flux density
    mag_err : array-like
        Magnitude error

    Returns
    -------
    flux_err : array-like
        Flux error obtained by the error propagation formula.
    """
    return LN10_0_4 * flux * mag_err
