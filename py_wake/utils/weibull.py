import numpy as np
from scipy.special import gamma


def mean(A, k):
    return A * gamma(1 + 1 / k)


def cdf(ws, A, k):
    return 1 - np.exp(-(1 / A * ws) ** k)


def WeightedPower(u, power, A, k):
    """Calculate the weibull weighted power

    Parameters
    ----------
    Power : xarray DataArray
        Power
    Returns
    -------
    y : array_like
        y is

    Notes
    ------
    bla bla
    """

    # see https://orbit.dtu.dk/en/publications/european-wind-atlas, page 95
    def G(alpha, k):
        # 1/k times the incomplete gamma function of the two arguments 1/k and alpha^k
        # Note, the scipy incomplete gamma function, gammainc, must be multiplied with gamma(k) to match the
        # the G function used in the European Wind Atlas
        import scipy.special as sc
        return 1 / k * sc.gamma(1 / k) * sc.gammainc(1 / k, alpha**k)

    u0, u1 = u[:-1], u[1:]
    alpha0, alpha1 = (u0 / A), (u1 / A)
    p0, p1 = power[..., :-1], power[..., 1:]

    res = (p0 * np.exp(-alpha0**k) +  # eq 6.5, p0 * cdf(u0:)
           (p1 - p0) / (alpha1 - alpha0) * (G(alpha1, k) - G(alpha0, k)) -  # eq 6.4 linear change p0 to p1
           p1 * np.exp(-alpha1**k)
           )  # eq 6.5, - p1 * cdf(u1:)

    return res
