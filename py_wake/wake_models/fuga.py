import os
import struct
import numpy as np
from py_wake.wake_model import LinearSum, WakeModel
from numpy import newaxis as na


class Fuga(LinearSum, WakeModel):
    ams = 5
    invL = 0
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'dw_ijl', 'hcw_ijl', 'dh_ijl', 'h_il', 'ct_ilk']

    def __init__(self, LUT_path, site, windTurbines, **kwargs):
        WakeModel.__init__(self, site, windTurbines, **kwargs)
        self.load(LUT_path)

    def load(self, path):

        with open(path + 'CaseData.bin', 'rb') as fid:
            case_name = struct.unpack('127s', fid.read(127))[0]
            r = struct.unpack('d', fid.read(8))[0]
            zhub = struct.unpack('d', fid.read(8))[0]
            lo_level = struct.unpack('I', fid.read(4))[0]
            hi_level = struct.unpack('I', fid.read(4))[0]
            z0 = struct.unpack('d', fid.read(8))[0]
            zi = struct.unpack('d', fid.read(8))[0]
            ds = struct.unpack('d', fid.read(8))[0]
            closure = struct.unpack('I', fid.read(4))[0]
            if os.path.getsize(path + 'CaseData.bin') == 187:
                zeta0 = struct.unpack('d', fid.read(8))[0]
            else:
                with open(path + 'CaseData.bin') as fid2:
                    info = fid2.read(127)
                zeta0 = float(info[info.index('Zeta0'):].replace("Zeta0=", ""))
                # zeta0 = float(path[path.index('Zeta0'):].replace("Zeta0=", "").replace("/", ""))

        def psim(zeta):
            return self.ams * zeta

        if not zeta0 >= 0:
            raise NotImplementedError  # See Colonel.u2b.psim
        factor = 1 / (1 - (psim(zhub * self.invL) - psim(zeta0)) / np.log(zhub / z0))

        f = [f for f in os.listdir(path) if f.endswith("input.par") or f.endswith('inputfile.par')][0]
        # z0_zi_zeta0 = os.path.split(os.path.dirname(path))[1]
        # z0, zi, zeta0 = re.match('Z0=(\d+.\d+)Zi=(\d+)Zeta0=(\d+.\d+E\+\d+)', z0_zi_zeta0).groups()

        with open(path + f) as fid:
            lines = fid.readlines()
        prefix = lines[0].strip()
        nxW, nyW = map(int, lines[2:4])
        dx, dy, sigmax, sigmay = map(float, lines[4:8])
        self.lo_level, self.hi_level = map(int, lines[11:13])
        self.dsAll = ds

        zlevels = np.arange(self.lo_level, self.hi_level + 1)
        mdu = [np.fromfile(path + prefix + '%04dUL.dat' % j, np.dtype('<f'), -1)
               for j in zlevels]

        self.du = -np.array(mdu, dtype=np.float32).reshape((len(mdu), nyW // 2, nxW)) * factor
        self.z0 = z0
        self.x0 = nxW // 4
        self.dx = dx
        self.x = np.arange(-self.x0, nxW * 3 / 4) * dx
        self.y = np.arange(nyW // 2) * dy
        self.dy = dy
        if self.lo_level == self.hi_level == 9999:
            self.z = [zhub]
        else:
            self.z = z0 * np.exp(zlevels * self.dsAll)

        self.lut_interpolator = LUTInterpolator(self.x, self.y, self.z, self.du)

    def interpolate(self, x, y, z):
        x = np.maximum(np.minimum(x, self.x[-1]), self.x[0])
        y = np.maximum(np.minimum(y, self.y[-1]), self.y[0])
        z = np.maximum(np.minimum(z, self.z[-1]), self.z[0])
        return self.lut_interpolator((x, y, z))

    def _calc_layout_terms(self, dw_ijl, hcw_ijl, h_il, dh_ijl, **_):
        self.mdu_ijl = self.interpolate(dw_ijl, np.abs(hcw_ijl), h_il[:, na] + dh_ijl)

    def calc_deficit(self, WS_ilk, WS_eff_ilk, dw_ijl, hcw_ijl, dh_ijl, h_il, ct_ilk, **_):
        if not self.deficit_initalized:
            self._calc_layout_terms(dw_ijl, hcw_ijl, h_il, dh_ijl)
        return self.mdu_ijl[..., na] * (ct_ilk * WS_eff_ilk**2 / WS_ilk)[:, na]


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


def main():
    if __name__ == '__main__':
        from py_wake.aep_calculator import AEPCalculator
        from py_wake.examples.data.iea37 import iea37_path
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines

        # setup site, turbines and wakemodel
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

        from py_wake.tests.test_files import tfp
        path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'
        wake_model = Fuga(path, site, windTurbines)

        # calculate AEP
        aep_calculator = AEPCalculator(wake_model)
        aep = aep_calculator.calculate_AEP(x, y)[0].sum()

        # plot wake map
        import matplotlib.pyplot as plt
        aep_calculator.plot_wake_map(wt_x=x, wt_y=y, wd=[0], ws=[9])
        plt.title('AEP: %.2f GWh' % aep)
        windTurbines.plot(x, y)
        plt.show()


main()
