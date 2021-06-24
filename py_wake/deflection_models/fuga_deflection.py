from numpy import newaxis as na
import matplotlib.pyplot as plt
import numpy as np
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.tests.test_files import tfp
from py_wake.utils.fuga_utils import FugaUtils
from py_wake.utils.grid_interpolator import GridInterpolator
from scipy.interpolate.interpolate import RegularGridInterpolator


class FugaDeflection(FugaUtils, DeflectionModel):
    args4deflection = ['WS_ilk', 'WS_eff_ilk', 'yaw_ilk', 'ct_ilk', 'D_src_il']

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00', on_mismatch='raise'):
        FugaUtils.__init__(self, path=LUT_path, on_mismatch=on_mismatch)
        if len(self.zlevels) == 1:
            tabs = self.load_luts(['VL', 'VT']).reshape(2, -1, self.nx)
        else:
            # interpolate to hub height
            jh = np.floor(np.log(self.zHub / self.z0) / self.ds)
            zlevels = [jh, jh + 1]
            tabs = self.load_luts(['VL', 'VT'], zlevels).reshape(2, 2, -1, self.nx)
            t = np.modf(np.log(self.zHub / self.z0) / self.ds)[0]
            tabs = tabs[:, 0] * (1 - t) + t * tabs[:, 1]

        VL, VT = tabs
        VL = -VL
        self.VL, self.VT = VL, VT

        nx0 = self.nx0
        ny = self.ny // 2

        fL = np.cumsum(np.concatenate([np.zeros((ny, 1)), ((VL[:, :-1] + VL[:, 1:]) / 2)], 1), 1)
        fT = np.cumsum(np.concatenate([np.zeros((ny, 1)), ((VT[:, :-1] + VT[:, 1:]) / 2)], 1), 1)

        # subtract rotor center
        fL = (fL - fL[:, nx0:nx0 + 1]) * self.dx
        fT = (fT - fT[:, nx0:nx0 + 1]) * self.dx

        self.fLtab = fL = np.concatenate([-fL[::-1], fL[1:]], 0)
        self.fTtab = fT = np.concatenate([fT[::-1], fT[1:]], 0)
        self.fLT = GridInterpolator([self.x, self.mirror(self.y, anti_symmetric=True)], np.array([fL, fT]).T)

    def calc_deflection(self, dw_ijl, hcw_ijl, dh_ijl, WS_ilk, WS_eff_ilk, yaw_ilk, ct_ilk, D_src_il, **_):
        I, L, K = ct_ilk.shape
        X = int(np.max(D_src_il) * 3 / self.dy + 1)
        J = dw_ijl.shape[1]

        WS_hub_ilk = WS_ilk

        theta_ilk = np.deg2rad(yaw_ilk)
        cos_ilk, sin_ilk = np.cos(theta_ilk), np.sin(theta_ilk)

        F_ilk = ct_ilk * (WS_eff_ilk)**2 / (WS_ilk * WS_hub_ilk)

        """
        For at given cross wind position in the lookup tables, yp, the deflection is lambda2p(yp), i.e.
        the real position (corresponding to the output hcw), is yp = y - lambda2p(yp) = y - lambda2(y),
        where y is the input hcw. I.e.:
        lambda(y) = lambda(yp + lp(yp)) = lp(yp)
        and
        yp = y - lambda(y)
        """

        if 0:  # J < 1000:
            # To find lut_y we calculate y for a range of lut_y grid points around y and interpolate
            lut_y_ijlx = (np.round(hcw_ijl[:, :, :, na] / self.dy) + np.arange(-X // 2, X // 2)[na, na, na]) * self.dy
            dw_ijlx = np.repeat(dw_ijl[:, :, :, na], X, 3)
            # calculate deflection, lambda(lut_y) =  # F * (cos(yaw) * fT(lut_y) + sin(yaw) * fL(lut_y)
            fL, fT = self.fLT(np.array([dw_ijlx.flatten(), lut_y_ijlx.flatten()]).T).T

            lambda_ijlkx = F_ilk[:, na, :, :, na] * (fL.reshape(I, J, L, 1, X) * cos_ilk[:, na, :, :, na] +
                                                     fT.reshape(I, J, L, 1, X) * sin_ilk[:, na, :, :, na])
            # Calcuate deflected y
            y_ijlkx = lut_y_ijlx[:, :, :, na] + lambda_ijlkx

            hcw_ijlk = np.array([[[[np.interp(hcw_ijl[i, j, l], y_ijlkx[i, j, l, k], lut_y_ijlx[i, j, l])
                                    for k in range(K)]
                                   for l in range(L)]
                                  for j in range(J)]
                                 for i in range(I)])
        else:
            x, y = self.fLT.x

#             def get_hcw_jk(i, l):
#                 x_idx = (np.searchsorted(x, [dw_ijl.min(), dw_ijl.max()]) + np.array([-1, 1]))
#                 m_x = len(x) - 1
#                 x_slice = slice(*np.minimum([m_x, m_x], np.maximum([0, 0], x_idx)))
#
#                 y_idx = (np.searchsorted(y, [hcw_ijl.min(), hcw_ijl.max()]) + np.array([-20, 20]))
#                 m_y = len(y) - 1
#                 y_slice = slice(*np.minimum([m_y, m_y], np.maximum([0, 0], y_idx)))
#
#                 x_ = x[x_slice]
#                 y_ = y[y_slice]
#                 VLT = self.fLT.V[x_slice, y_slice]
#
#                 def get_hcw_j(i, l, k):
#                     lambda2p = F_ilk[i, l, k] * \
#                         np.sum(VLT * [np.cos(theta_ilk[i, l, k]), np.sin(theta_ilk[i, l, k])], -1)
#                     lambda2 = RegularGridInterpolator(
#                         (x_, y_), [np.interp(y_, y_ + l2p_x, l2p_x) for l2p_x in lambda2p])
#
#                     hcw_j = hcw_ijl[i, :, l].copy()
#                     m = (hcw_ijl[i, :, l] > y_[0]) & (hcw_ijl[i, :, l] < y_[-1])
#                     hcw_j[m] -= lambda2((dw_ijl[i, :, l][m], hcw_ijl[i, :, l][m]))
#                     return hcw_j
#                 return [get_hcw_j(i, l, k) for k in range(K)]
#
#             hcw_ijlk_old = np.moveaxis([[get_hcw_jk(i, l)
#                                          for l in range(L)]
#                                         for i in range(I)], 3, 1)

            hcw_ijlk = np.array([self.get_hcw_jlk(i, K, L, x, y, dw_ijl, hcw_ijl, F_ilk, theta_ilk)
                                 for i in range(I)])
#             npt.assert_array_almost_equal(hcw_ijlk_old, hcw_ijlk, 4)

        return dw_ijl[:, :, :, na], hcw_ijlk, dh_ijl[..., na]

    def get_hcw_jlk(self, i, K, L, x, y, dw_ijl, hcw_ijl, F_ilk, theta_ilk):
        if (K == 1 and L > 1 and np.all(dw_ijl == dw_ijl[:1, :, :1]) and np.all(hcw_ijl == hcw_ijl[:1, :, :1]) and
                len(np.unique(theta_ilk[i, :, 0])) < L):
            hcw_jlk = np.zeros((dw_ijl.shape[1], L, K))
            for theta, l in zip(*np.unique(theta_ilk[i], return_index=True)):
                hcw_jlk[:, theta_ilk[i, :, 0] == theta] = np.array(self.get_hcw_jk(
                    i, l, K, x, y, dw_ijl, hcw_ijl, F_ilk, theta_ilk)).T[:, na]
            return hcw_jlk

        else:
            return np.moveaxis([self.get_hcw_jk(i, l, K, x, y, dw_ijl, hcw_ijl, F_ilk, theta_ilk)
                                for l in range(L)], 2, 0)

    def get_hcw_jk(self, i, l, K, x, y, dw_ijl, hcw_ijl, F_ilk, theta_ilk):
        x_idx = (np.searchsorted(x, [dw_ijl.min(), dw_ijl.max()]) + np.array([-1, 1]))
        m_x = len(x) + 1
        x_slice = slice(*np.minimum([m_x, m_x], np.maximum([0, 0], x_idx)))

        y_idx = (np.searchsorted(y, [hcw_ijl.min(), hcw_ijl.max()]) + np.array([-20, 20]))
        m_y = len(y) + 1
        y_slice = slice(*np.minimum([m_y, m_y], np.maximum([0, 0], y_idx)))

        x_ = x[x_slice]
        y_ = y[y_slice]
        VLT = self.fLT.V[x_slice, y_slice]
        return [self.get_hcw_j(i, l, k, F_ilk, VLT, theta_ilk, x_, y_, hcw_ijl, dw_ijl) for k in range(K)]

    def get_hcw_j(self, i, l, k, F_ilk, VLT, theta_ilk, x_, y_, hcw_ijl, dw_ijl):
        lambda2p = F_ilk[i, l, k] * \
            np.sum(VLT * [np.cos(theta_ilk[i, l, k]), np.sin(theta_ilk[i, l, k])], -1)
        lambda2 = RegularGridInterpolator(
            (x_, y_), [np.interp(y_, y_ + l2p_x, l2p_x) for l2p_x in lambda2p])

        hcw_j = hcw_ijl[i, :, l].copy()
        m = (hcw_ijl[i, :, l] > y_[0]) & (hcw_ijl[i, :, l] < y_[-1])
        hcw_j[m] -= lambda2((dw_ijl[i, :, l][m], hcw_ijl[i, :, l][m]))
        return hcw_j


def main():
    if __name__ == '__main__':
        from py_wake import Fuga
        from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
        import matplotlib.pyplot as plt

        site = IEA37Site(16)
        x, y = [0, 600, 1200], [0, 0, 0]  # site.initial_position[:2].T
        windTurbines = IEA37_WindTurbines()
        path = tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/'
        noj = Fuga(path, site, windTurbines, deflectionModel=FugaDeflection(path))
        yaw = [-30, 30, 0]
        noj(x, y, yaw=yaw, wd=270, ws=10).flow_map().plot_wake_map()
        plt.show()


main()
