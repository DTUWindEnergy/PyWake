from numpy import newaxis as na
import matplotlib.pyplot as plt
import numpy as np
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.utils.fuga_utils import FugaUtils
from py_wake.utils.grid_interpolator import GridInterpolator
import time


class FugaDeflection(FugaUtils, DeflectionModel):
    args4deflection = ['WS_ilk', 'WS_eff_ilk', 'yaw_ilk', 'ct_ilk', 'D_src_il']

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.00014617Zi=00399Zeta0=0.00E+0/', on_mismatch='raise'):
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
        # VL[0] *= -1  # change sign of center line to match notebook

        nx0 = self.nx0
        ny = self.ny // 2

        fL = np.cumsum(np.concatenate([np.zeros((ny, 1)), ((VL[:, :-1] + VL[:, 1:]) / 2)], 1), 1)
        fT = np.cumsum(np.concatenate([np.zeros((ny, 1)), ((VT[:, :-1] + VT[:, 1:]) / 2)], 1), 1)

        # subtract rotor center
        fL = (fL - fL[:, nx0 - 1:nx0]) * self.dx
        fT = (fT - fT[:, nx0 - 1:nx0]) * self.dx

        self.fLtab = fL = np.concatenate([-fL[::-1], fL[1:]], 0)
        self.fTtab = fT = np.concatenate([fT[::-1], fT[1:]], 0)
        self.fLT = GridInterpolator([self.x + self.dx, self.mirror(self.y, anti_symmetric=True)], np.array([fL, fT]).T)

    def calc_deflection(self, dw_ijl, hcw_ijl, dh_ijl, WS_ilk, WS_eff_ilk, yaw_ilk, ct_ilk, D_src_il, **_):
        I, L, K = ct_ilk.shape
        X = int(np.max(D_src_il) * 3 / self.dy + 1)
        J = dw_ijl.shape[1]

        WS_hub_ilk = WS_ilk

        cos_ilk, sin_ilk = np.cos(yaw_ilk), np.sin(yaw_ilk)

        F_ilk = ct_ilk * (WS_eff_ilk * cos_ilk)**2 / (WS_ilk * WS_hub_ilk)

        # For at given cross wind position in the lookup tables, lut_y, the deflection is lambda(lut_y), i.e.
        # the real position (corresponding to hcw), is y = lut_y + lambda(lut_y)
        # To find lut_y we calculate y for a range of lut_y grid points around y and interpolate
        lut_y_ijlx = (np.round(hcw_ijl[:, :, :, na] / self.dy) + np.arange(-X // 2, X // 2)[na, na, na]) * self.dy

        # calculate deflection, lambda(lut_y) =  # F * (cos(yaw) * fT(lut_y) + sin(yaw) * fL(lut_y)
        dw_ijlx = np.repeat(dw_ijl[:, :, :, na], X, 3)
        if J < 1000:
            fL, fT = self.fLT(np.array([dw_ijlx.flatten(), lut_y_ijlx.flatten()]).T).T

            lambda_ijlkx = F_ilk[:, na, :, :, na] * (fL.reshape(I, J, L, 1, X) * cos_ilk[:, na, :, :, na] +
                                                     fT.reshape(I, J, L, 1, X) * sin_ilk[:, na, :, :, na])
            # Calcuate deflected y
            y_ijlkx = lut_y_ijlx[:, :, :, na] + lambda_ijlkx

            # assert (np.all(hcw_ijl < y_ijlkx.max((3, 4))))
            # assert (np.all(hcw_ijl > y_ijlkx.min((3, 4))))

            hcw_ijlk = np.array([[[[np.interp(hcw_ijl[i, j, l], y_ijlkx[i, j, l, k], lut_y_ijlx[i, j, l])
                                    for k in range(K)]
                                   for l in range(L)]
                                  for j in range(J)]
                                 for i in range(I)])
        else:
            dw_ijl_round = np.round(dw_ijl, 10)
            x, y = self.fLT.x

            def get_hcw(i, l, k):
                lambda_xy = F_ilk[i, l, k] * \
                    np.sum(self.fLT.V * [np.cos(yaw_ilk[i, l, k]), np.sin(yaw_ilk[i, l, k])], -1)

                hcw_j = hcw_ijl[i, :, k]
                for v in np.unique(dw_ijl_round[i, :, l]):
                    xi = np.searchsorted(x, v)
                    lambda_y = (v - x[xi]) / self.dx * lambda_xy[xi + 1] + (x[xi + 1] - v) / self.dx * lambda_xy[xi]
                    idx = dw_ijl_round[i, :, l] == v
                    hcw_j[idx] = np.interp(hcw_j[idx], y + lambda_y, y)
                return hcw_j

            hcw_ijlk = np.moveaxis([[[get_hcw(i, l, k)
                                      for k in range(K)]
                                     for l in range(L)]
                                    for i in range(I)], -1, 1)

        return dw_ijl[:, :, :, na], hcw_ijlk, dh_ijl[..., na]
