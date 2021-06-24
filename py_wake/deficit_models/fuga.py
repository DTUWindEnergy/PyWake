from numpy import newaxis as na

import numpy as np
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorCenter
from py_wake.superposition_models import LinearSum
from py_wake.tests.test_files import tfp
from py_wake.utils.fuga_utils import FugaUtils
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from scipy.interpolate.fitpack2 import RectBivariateSpline


class FugaDeficit(WakeDeficitModel, BlockageDeficitModel, FugaUtils):
    ams = 5
    invL = 0
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'dw_ijlk', 'hcw_ijlk', 'dh_ijlk', 'h_il', 'ct_ilk', 'D_src_il']

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/', remove_wriggles=False,
                 method='linear'):
        """
        Parameters
        ----------
        LUT_path : str
            Path to folder containing 'CaseData.bin', input parameter file (*.par) and loop-up tables
        remove_wriggles : bool
            The current Fuga loop-up tables have significan wriggles.
            If True, all deficit values after the first zero crossing (when going from the center line
            and out in the lateral direction) is set to zero.
            This means that all speed-up regions are also removed
        """
        BlockageDeficitModel.__init__(self, upstream_only=True)
        FugaUtils.__init__(self, LUT_path, on_mismatch='input_par')
        self.remove_wriggles = remove_wriggles
        x, y, z, du = self.load()
        err_msg = "Method must be 'linear' or 'spline'. Spline is supports only height level only"
        assert method == 'linear' or (method == 'spline' and len(z) == 1), err_msg

        if method == 'linear':
            self.lut_interpolator = LUTInterpolator(x, y, z, du)
        else:
            du_interpolator = RectBivariateSpline(x, y, du[0].T)

            def interp(xyz):
                x, y, z = xyz
                assert np.all(z == self.z[0]), f'LUT table contains z={self.z} only'
                return du_interpolator.ev(x, y)
            self.lut_interpolator = interp

    def zeta0_factor(self):
        def psim(zeta):
            return self.ams * zeta

        if not self.zeta0 >= 0:  # pragma: no cover
            # See Colonel.u2b.psim
            raise NotImplementedError
        return 1 / (1 - (psim(self.zHub * self.invL) - psim(self.zeta0)) / np.log(self.zHub / self.z0))

    def load(self):

        mdUL = self.load_luts(['UL'])[0]

        du = -np.array(mdUL, dtype=np.float32) * self.zeta0_factor()  # minus because it is deficit

        if self.remove_wriggles:
            # remove all positive and negative deficits after first zero crossing in lateral direction
            du *= (np.cumsum(du < 0, 1) == 0)

        # smooth edges to zero
        n = 250
        du[:, :, :n] = du[:, :, n][:, :, na] * np.arange(n) / n
        du[:, :, -n:] = du[:, :, -n][:, :, na] * np.arange(n)[::-1] / n
        n = 50
        du[:, -n:, :] = du[:, -n, :][:, na, :] * np.arange(n)[::-1][na, :, na] / n

        return self.x, self.y, self.z, du

    def interpolate(self, x, y, z):
        # self.grid_interplator(np.array([zyx.flatten() for zyx in [z, y, x]]).T, check_bounds=False).reshape(x.shape)
        return self.lut_interpolator((x, y, z))

    def _calc_layout_terms(self, dw_ijlk, hcw_ijlk, h_il, dh_ijlk, D_src_il, **_):

        self.mdu_ijlk = self.interpolate(dw_ijlk, np.abs(hcw_ijlk), (h_il[:, na, :, na] + dh_ijlk)) * \
            ~((dw_ijlk == 0) & (hcw_ijlk <= D_src_il[:, na, :, na])  # avoid wake on itself
              )

    def calc_deficit(self, WS_ilk, WS_eff_ilk, dw_ijlk, hcw_ijlk, dh_ijlk, h_il, ct_ilk, D_src_il, **kwargs):
        if not self.deficit_initalized:
            self._calc_layout_terms(dw_ijlk, hcw_ijlk, h_il, dh_ijlk, D_src_il, **kwargs)
        return self.mdu_ijlk * (ct_ilk * WS_eff_ilk**2 / WS_ilk)[:, na]

    def wake_radius(self, D_src_il, dw_ijlk, **_):
        # Set at twice the source radius for now
        return np.zeros_like(dw_ijlk) + D_src_il[:, na, :, na]


class FugaYawDeficit(FugaDeficit):
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'dw_ijlk', 'hcw_ijlk', 'dh_ijlk', 'h_il', 'ct_ilk', 'D_src_il', 'yaw_ilk']

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/',
                 remove_wriggles=False, method='linear'):
        """
        Parameters
        ----------
        LUT_path : str
            Path to folder containing 'CaseData.bin', input parameter file (*.par) and loop-up tables
        remove_wriggles : bool
            The current Fuga loop-up tables have significan wriggles.
            If True, all deficit values after the first zero crossing (when going from the center line
            and out in the lateral direction) is set to zero.
            This means that all speed-up regions are also removed
        """

        FugaUtils.__init__(self, LUT_path, on_mismatch='input_par')
        self.remove_wriggles = remove_wriggles
        x, y, z, dUL = self.load()

        mdUT = self.load_luts(['UT'])[0]
        dUT = np.array(mdUT, dtype=np.float32) * self.zeta0_factor()
        dU = np.concatenate([dUL[:, :, :, na], dUT[:, :, :, na]], 3)
        err_msg = "Method must be 'linear' or 'spline'. Spline is supports only height level only"
        assert method == 'linear' or (method == 'spline' and len(z) == 1), err_msg

        if method == 'linear':
            self.lut_interpolator = LUTInterpolator(x, y, z, dU)
        else:
            UL_interpolator = RectBivariateSpline(x, y, dU[0, :, :, 0].T)
            UT_interpolator = RectBivariateSpline(x, y, dU[0, :, :, 1].T)

            def interp(xyz):
                x, y, z = xyz
                assert np.all(z == self.z[0]), f'LUT table contains z={self.z} only'
                return np.moveaxis([UL_interpolator.ev(x, y), UT_interpolator.ev(x, y)], 0, -1)
            self.lut_interpolator = interp

    def calc_deficit_downwind(self, WS_ilk, WS_eff_ilk, dw_ijlk, hcw_ijlk,
                              dh_ijlk, h_il, ct_ilk, D_src_il, yaw_ilk, **_):

        mdUL_ijlk, mdUT_ijlk = np.moveaxis(self.interpolate(
            dw_ijlk, np.abs(hcw_ijlk), (h_il[:, na, :, na] + dh_ijlk)), -1, 0)
        mdUT_ijlk[hcw_ijlk < 0] *= -1  # UT is antisymmetric
        theta_ilk = np.deg2rad(yaw_ilk)
        mdu_ijlk = (mdUL_ijlk * np.cos(theta_ilk)[:, na] - mdUT_ijlk * np.sin(theta_ilk)[:, na])
        mdu_ijlk *= ~((dw_ijlk == 0) & (hcw_ijlk <= D_src_il[:, na, :, na]))  # avoid wake on itself

        return mdu_ijlk * (ct_ilk * WS_eff_ilk**2 / WS_ilk)[:, na]


class LUTInterpolator(object):
    # Faster than scipy.interpolate.interpolate.RegularGridInterpolator
    def __init__(self, x, y, z, V):
        self.x = x
        self.y = y
        self.z = z
        self.V = V
        self.nx = nx = len(x)
        self.ny = ny = len(y)
        self.nz = nz = len(z)
        assert V.shape[:3] == (nz, ny, nx)
        self.dx, self.dy = [xy[1] - xy[0] for xy in [x, y]]

        self.x0 = x[0]
        self.y0 = y[0]

        Ve = np.concatenate((V, V[-1:]), 0)
        Ve = np.concatenate((Ve, Ve[:, -1:]), 1)
        Ve = np.concatenate((Ve, Ve[:, :, -1:]), 2)

        self.V000 = np.array([V,
                              Ve[:-1, :-1, 1:],
                              Ve[:-1, 1:, :-1],
                              Ve[:-1, 1:, 1:],
                              Ve[1:, :-1, :-1],
                              Ve[1:, :-1, 1:],
                              Ve[1:, 1:, :-1],
                              Ve[1:, 1:, 1:]])
        if V.shape == (nz, ny, nx, 2):
            # Both UL and UT
            self.V000 = self.V000.reshape((8, nz * ny * nx, 2))
        else:
            self.V000 = self.V000.reshape((8, nz * ny * nx))

    def __call__(self, xyz):
        xp, yp, zp = xyz
        xp = np.maximum(np.minimum(xp, self.x[-1]), self.x[0])
        yp = np.maximum(np.minimum(yp, self.y[-1]), self.y[0])
        # zp = np.maximum(np.minimum(zp, self.z[-1]), self.z[0])

        def i0f(_i):
            _i0 = np.asarray(_i).astype(int)
            _if = _i - _i0
            return _i0, _if

        xi0, xif = i0f((xp - self.x0) / self.dx)
        yi0, yif = i0f((yp - self.y0) / self.dy)

        zi0, zif = i0f(np.interp(zp, self.z, np.arange(self.nz)))

        nx, ny = self.nx, self.ny

        v000, v001, v010, v011, v100, v101, v110, v111 = self.V000[:, zi0 * nx * ny + yi0 * nx + xi0]
        if len(self.V000.shape) == 3:
            # Both UL and UT
            xif = xif[:, :, :, :, na]
            yif = yif[:, :, :, :, na]
            zif = zif[:, :, :, :, na]
        v_00 = v000 + (v100 - v000) * zif
        v_01 = v001 + (v101 - v001) * zif
        v_10 = v010 + (v110 - v010) * zif
        v_11 = v011 + (v111 - v011) * zif
        v__0 = v_00 + (v_10 - v_00) * yif
        v__1 = v_01 + (v_11 - v_01) * yif

        return (v__0 + (v__1 - v__0) * xif)
#         # Slightly slower
#         xif1, yif1, zif1 = 1 - xif, 1 - yif, 1 - zif
#         w = np.array([xif1 * yif1 * zif1,
#                       xif * yif1 * zif1,
#                       xif1 * yif * zif1,
#                       xif * yif * zif1,
#                       xif1 * yif1 * zif,
#                       xif * yif1 * zif,
#                       xif1 * yif * zif,
#                       xif * yif * zif])
#
#         return np.sum(w * self.V01[:, zi0, yi0, xi0], 0)


class Fuga(PropagateDownwind):
    def __init__(self, LUT_path, site, windTurbines,
                 rotorAvgModel=RotorCenter(), deflectionModel=None, turbulenceModel=None, remove_wriggles=False):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            Defaults to RotorCenter that uses the rotor center wind speed (i.e. one point) only
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=FugaDeficit(LUT_path, remove_wriggles=remove_wriggles),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)


class FugaBlockage(All2AllIterative):
    def __init__(self, LUT_path, site, windTurbines,
                 rotorAvgModel=RotorCenter(),
                 deflectionModel=None, turbulenceModel=None, convergence_tolerance=1e-6, remove_wriggles=False):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            Defaults to RotorCenter that uses the rotor center wind speed (i.e. one point) only
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        fuga_deficit = FugaDeficit(LUT_path, remove_wriggles=remove_wriggles)
        All2AllIterative.__init__(self, site, windTurbines, wake_deficitModel=fuga_deficit,
                                  rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                  deflectionModel=deflectionModel, blockage_deficitModel=fuga_deficit,
                                  turbulenceModel=turbulenceModel, convergence_tolerance=convergence_tolerance)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'

        for wf_model in [Fuga(path, site, windTurbines),
                         FugaBlockage(path, site, windTurbines)]:
            plt.figure()
            print(wf_model)

            # run wind farm simulation
            sim_res = wf_model(x, y)

            # calculate AEP
            aep = sim_res.aep().sum()

            # plot wake map
            flow_map = sim_res.flow_map(wd=30, ws=9.8)
            flow_map.plot_wake_map()
            flow_map.plot_windturbines()
            plt.title('AEP: %.2f GWh' % aep)
            plt.show()
        plt.show()


main()
