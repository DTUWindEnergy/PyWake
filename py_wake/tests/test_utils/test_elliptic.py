import numpy as np
from py_wake.utils.elliptic import ellipticPiCarlson
from scipy.special import ellipk, ellipe
from py_wake.tests import npt


def test_Elliptic_behavior():
    npt.assert_almost_equal(len(ellipticPiCarlson([], [])), 0, decimal=9)


def test_Elliptic_points():
    # Example points as given in book: Branlard - Wind Turbine Aerodynamics, p.626
    npt.assert_almost_equal(ellipticPiCarlson(0.5, 0.6), 2.86752, decimal=5)
    npt.assert_almost_equal(ellipticPiCarlson(-0.5, -0.6), 1.15001, decimal=5)

    # --- PI(-3,0) = pi/(2*sqrt(4))
    npt.assert_almost_equal(ellipticPiCarlson(-3, 0), np.pi / (2 * np.sqrt(4)), decimal=9)
    # ellippi(3,0)= pi/(2*sqrt(-2)) = (0.0 - 1.11072073453959156175397j)


# def test_Elliptic_mpmath():
#     # --- Compare results with mpmath function
#     from itertools import product
#     from mpmath import ellippi

#     nn = 3
#     X = np.concatenate((-np.linspace(10**-6, 10**6, nn), np.linspace(10**-6, 1 - 10**-6, nn)))
#     M = np.array([m for m, _ in product(X, X)])
#     N = np.array([n for _, n in product(X, X)])
#     PI_C = ellipticPiCarlson(N, M)
#     PI_M = np.zeros(M.shape)
#     for i, (m, n) in enumerate(zip(M, N)):
#         PI_M[i] = ellippi(n, m)
#     npt.assert_almost_equal(PI_C, PI_M, decimal=7)


def test_Elliptic_property():
    # Useful vector, m \in ]-infty,0[ U ]0,1[
    nn = 6
    m = np.concatenate((-np.linspace(10**-6, 10**6, nn), np.linspace(10**-6, 1 - 10**-6, nn)))

    # --- Property C.75, as given in book: Branlard - Wind Turbine Aerodynamics, p.627
    npt.assert_almost_equal(ellipticPiCarlson(m, m), 1 / (1 - m) * ellipe(m), decimal=7)
    # --- PI(0,m) = K(m)
    npt.assert_almost_equal(ellipticPiCarlson(0 * m, m), ellipk(m), decimal=9)
