import os
import struct
from numpy import newaxis as na
import numpy as np
from py_wake.deficit_models.deficit_model import DeficitModel
from py_wake.superposition_models import LinearSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.rotor_avg_models.rotor_avg_model import RotorCenter
from py_wake.tests.test_files import tfp
from py_wake.utils.grid_interpolator import GridInterpolator


class FugaDeficit(DeficitModel):
    ams = 5
    invL = 0
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'dw_ijlk', 'hcw_ijlk', 'dh_ijl', 'h_il', 'ct_ilk', 'D_src_il']

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'):
        self.lut_interpolator = LUTInterpolator(*self.load(LUT_path))

    def load(self, path):

        with open(path + 'CaseData.bin', 'rb') as fid:
            case_name = struct.unpack('127s', fid.read(127))[0]  # @UnusedVariable
            r = struct.unpack('d', fid.read(8))[0]  # @UnusedVariable
            zhub = struct.unpack('d', fid.read(8))[0]
            lo_level = struct.unpack('I', fid.read(4))[0]  # @UnusedVariable
            hi_level = struct.unpack('I', fid.read(4))[0]  # @UnusedVariable
            z0 = struct.unpack('d', fid.read(8))[0]
            zi = struct.unpack('d', fid.read(8))[0]  # @UnusedVariable
            ds = struct.unpack('d', fid.read(8))[0]
            closure = struct.unpack('I', fid.read(4))[0]  # @UnusedVariable
            if os.path.getsize(path + 'CaseData.bin') == 187:
                zeta0 = struct.unpack('d', fid.read(8))[0]
            else:
                #                 with open(path + 'CaseData.bin', 'rb') as fid2:
                #                     info = fid2.read(127).decode()
                #                 zeta0 = float(info[info.index('Zeta0'):].replace("Zeta0=", ""))
                zeta0 = float(path[path.index('Zeta0'):].replace("Zeta0=", "").replace("/", ""))

        def psim(zeta):
            return self.ams * zeta

        if not zeta0 >= 0:  # pragma: no cover
            # See Colonel.u2b.psim
            raise NotImplementedError
        factor = 1 / (1 - (psim(zhub * self.invL) - psim(zeta0)) / np.log(zhub / z0))

        f = [f for f in os.listdir(path) if f.endswith("input.par") or f.endswith('inputfile.par')][0]
        # z0_zi_zeta0 = os.path.split(os.path.dirname(path))[1]
        # z0, zi, zeta0 = re.match('Z0=(\d+.\d+)Zi=(\d+)Zeta0=(\d+.\d+E\+\d+)', z0_zi_zeta0).groups()

        with open(path + f) as fid:
            lines = fid.readlines()
        prefix = lines[0].strip()
        nxW, nyW = map(int, lines[2:4])
        dx, dy, sigmax, sigmay = map(float, lines[4:8])  # @UnusedVariable
        lo_level, hi_level = map(int, lines[11:13])
        dsAll = ds

        zlevels = np.arange(lo_level, hi_level + 1)
        mdu = [np.fromfile(path + prefix + '%04dUL.dat' % j, np.dtype('<f'), -1)
               for j in zlevels]

        du = -np.array(mdu, dtype=np.float32).reshape((len(mdu), nyW // 2, nxW)) * factor
        z0 = z0
        x0 = nxW // 4
        dx = dx
        x = np.arange(-x0, nxW * 3 / 4) * dx
        y = np.arange(nyW // 2) * dy
        dy = dy
        if lo_level == hi_level == 9999:
            z = [zhub]
        else:
            z = z0 * np.exp(zlevels * dsAll)
        # self.grid_interplator = GridInterpolator([self.z, self.y, self.x], self.du)
        return x, y, z, du

    def interpolate(self, x, y, z):
        # self.grid_interplator(np.array([zyx.flatten() for zyx in [z, y, x]]).T, check_bounds=False).reshape(x.shape)
        return self.lut_interpolator((x, y, z))

    def _calc_layout_terms(self, dw_ijlk, hcw_ijlk, h_il, dh_ijl, D_src_il, **_):

        self.mdu_ijlk = self.interpolate(dw_ijlk, np.abs(hcw_ijlk), (h_il[:, na] + dh_ijl)[:, :, :, na]) * \
            ~((dw_ijlk == 0) & (hcw_ijlk <= D_src_il[:, na, :, na])  # avoid wake on itself
              )

    def calc_deficit(self, WS_ilk, WS_eff_ilk, dw_ijlk, hcw_ijlk, dh_ijl, h_il, ct_ilk, D_src_il, **kwargs):
        if not self.deficit_initalized:
            self._calc_layout_terms(dw_ijlk, hcw_ijlk, h_il, dh_ijl, D_src_il, **kwargs)
        return self.mdu_ijlk * (ct_ilk * WS_eff_ilk**2 / WS_ilk)[:, na]

    def wake_radius(self, D_src_il, dw_ijlk, **_):
        # Set at twice the source radius for now
        return np.zeros_like(dw_ijlk) + D_src_il[:, na, :, na]


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
        assert V.shape == (nz, ny, nx)
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
                              Ve[1:, 1:, 1:]]).reshape((8, nz * ny * nx))

    def __call__(self, xyz):
        xp, yp, zp = xyz
        xp = np.maximum(np.minimum(xp, self.x[-1]), self.x[0])
        yp = np.maximum(np.minimum(yp, self.y[-1]), self.y[0])
        zp = np.maximum(np.minimum(zp, self.z[-1]), self.z[0])

        def i0f(_i):
            _i0 = np.asarray(_i).astype(np.int)
            _if = _i - _i0
            return _i0, _if

        xi0, xif = i0f((xp - self.x0) / self.dx)
        yi0, yif = i0f((yp - self.y0) / self.dy)

        zi0, zif = i0f(np.interp(zp, self.z, np.arange(self.nz)))

        nx, ny = self.nx, self.ny

        v000, v001, v010, v011, v100, v101, v110, v111 = self.V000[:, zi0 * nx * ny + yi0 * nx + xi0]

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
                 rotorAvgModel=RotorCenter(), deflectionModel=None, turbulenceModel=None):
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
                                   wake_deficitModel=FugaDeficit(LUT_path),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)


class FugaBlockage(All2AllIterative):
    def __init__(self, LUT_path, site, windTurbines,
                 rotorAvgModel=RotorCenter(),
                 deflectionModel=None, turbulenceModel=None, convergence_tolerance=1e-6):
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
        fuga_deficit = FugaDeficit(LUT_path)
        All2AllIterative.__init__(self, site, windTurbines, wake_deficitModel=fuga_deficit,
                                  rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                  blockage_deficitModel=fuga_deficit, turbulenceModel=turbulenceModel,
                                  convergence_tolerance=convergence_tolerance)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'

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
