# --- General
import numpy as np
import unittest
from scipy.special import ellipk, ellipe


def ellipticPiCarlson(n, m):
    """
    Elliptic integral of the third kind using the method of Carlson
        PI(n,m)=int(1/((1-n*sin(t)^2)*sqrt(1-m*sin(t)^2)),t=0,pi/2)
     AUTHOR: N. Troldborg
     REF: B.C. Carlson (1979) "Computing Elliptic Integrals by Duplication"
     NOTE: m <=1
    """
    # --- Performance parameters
    RES = 1.e-12
    ITMAX = 20      # 20 iterations usually sufficient

    # --- Subfunctions RF, RJ, RC
    def ellipticRF(y=None):
        xo = np.zeros(y.shape)
        yo = y
        zo = np.ones(y.shape)
        nIt = 0
        res = 1
        RFo = np.zeros(yo.shape)
        while res > RES and nIt < ITMAX:
            lambda_ = (xo * yo)**0.5 + (xo * zo)**0.5 + (yo * zo)**0.5
            mu = (xo + yo + zo) / 3.
            xn = (xo + lambda_) / 4.
            yn = (yo + lambda_) / 4.
            zn = (zo + lambda_) / 4.
            X = 1 - xo / mu
            Y = 1 - yo / mu
            Z = 1 - zo / mu
            X2 = X**2
            X3 = X2 * X
            Y2 = Y**2
            Y3 = Y2 * Y
            Z2 = Z**2
            Z3 = Z2 * Z
            s1 = (X2 + Y2 + Z2) / 4
            s2 = (X3 + Y3 + Z3) / 6
            s12 = s1 ** 2
            s13 = s12 * s1
            r = 5. / 26 * s13 + 3. / 26 * s2**2
            RF = (mu**-0.5) * (1 + s1 / 5 + s2 / 7 + s12 / 6 + 3 / 11 * s1 * s2 + r)
            res = np.amax(np.abs(RF - RFo))
            RFo = RF
            xo = xn
            yo = yn
            zo = zn
            nIt = nIt + 1
        return RF

    def ellipticRJ(y=None, rho=None):
        b1 = rho > 0
        xt = np.zeros(y.shape)
        yt = y
        zt = np.ones(y.shape)
        rhot = rho
        # --- Dealing first with positive values
        xo = xt[b1]
        yo = yt[b1]
        zo = zt[b1]
        rhoo = rhot[b1]
        RJ = np.full(y.shape, np.inf)
        if np.any(b1):
            nIt = 0
            res = 1
            RJo = np.zeros(xo.shape)
            rhs1 = np.zeros(xo.shape)
            while res > RES and nIt < ITMAX:
                lambda_ = (xo * yo)**0.5 + (xo * zo)**0.5 + (yo * zo)**0.5
                mu = (xo + yo + zo + 2 * rhoo) / 5.
                xn = (xo + lambda_) / 4.
                yn = (yo + lambda_) / 4.
                zn = (zo + lambda_) / 4.
                rhon = (rhoo + lambda_) / 4.
                X = 1 - xo / mu
                Y = 1 - yo / mu
                Z = 1 - zo / mu
                RHO = 1 - rhoo / mu
                X2 = X ** 2
                X3 = X2 * X
                X4 = X3 * X
                X5 = X4 * X
                Y2 = Y**2
                Y3 = Y2 * Y
                Y4 = Y3 * Y
                Y5 = Y4 * Y
                Z2 = Z**2
                Z3 = Z2 * Z
                Z4 = Z3 * Z
                Z5 = Z4 * Z
                RHO2 = RHO**2
                RHO3 = RHO2 * RHO
                RHO4 = RHO3 * RHO
                RHO5 = RHO4 * RHO
                s1 = (X2 + Y2 + Z2 + 2 * RHO2) / 4.
                s2 = (X3 + Y3 + Z3 + 2 * RHO3) / 6.
                s3 = (X4 + Y4 + Z4 + 2 * RHO4) / 8.
                s4 = (X5 + Y5 + Z5 + 2 * RHO5) / 10.
                s12 = s1 * s1
                s13 = s12 * s1
                r = -1. / 10 * s13 + 3. / 10 * s2**2 + 3. / 5 * s1 * s3
                alfa = (rhoo * (xo**0.5 + yo**0.5 + zo**0.5) + (xo * yo * zo)**0.5)**2
                bet = rhoo * (rhoo + lambda_)**2
                rhs1 = rhs1 + 3 * 4 ** - nIt * ellipticRC(alfa, bet)
                rhs2 = (4**-(nIt + 1) * mu**-1.5) * (1 + 3 / 7 * s1 + s2 / 3 + 3 / 22 * s12 + 3 / 11 * s3 + 3 / 13 * (s1 * s2 + s4) + r)
                RJLoc = rhs1 + rhs2
                res = np.amax(np.abs(RJLoc - RJo))
                RJo = RJLoc
                xo = xn
                yo = yn
                zo = zn
                rhoo = rhon
                nIt = nIt + 1
            RJ[b1] = RJLoc
        return RJ

    def ellipticRC(x=None, y=None):
        # Computes Carlson's Degenerate Elliptic Integral
        # RC(x,y)=1/2int_0^infty (t+x)^-0.5*(t+y)^-1dt
        # Carlson, B.C. (1979) "Computing Elliptic Integrals by Duplication"
        # AUTHOR: N. Troldborg
        nIt = 0
        res = 1
        xo = x
        yo = y
        RCo = np.zeros(x.shape)
        while res > RES and nIt < ITMAX:
            lambda_ = 2 * (xo * yo)**0.5 + yo
            xn = (xo + lambda_) / 4
            yn = (yo + lambda_) / 4
            mu = (xo + 2 * yo) / 3
            s = (yo - xo) / (3 * mu)
            s2 = s ** 2
            s3 = s2 * s
            s4 = s3 * s
            s5 = s4 * s
            s6 = s5 * s
            RC = (mu**-0.5) * (1 + 3 / 10 * s2 + s3 / 7 + 3 / 8 * s4 + 9 / 22 * s5 + 159 / 208 * s6)
            res = np.amax(np.abs(RC - RCo))
            RCo = RC
            xo = xn
            yo = yn
            nIt = nIt + 1
        return RC

    # --- Main corpus
    if type(m) is not np.ndarray:
        m = np.array(m)
        n = np.array(n)
    if m.shape == (0,):
        return np.array([])
    RF = ellipticRF(1 - m)
    RJ = ellipticRJ(1 - m, 1 - n)
    PI = RF + 1 / 3 * n * RJ
    return PI
