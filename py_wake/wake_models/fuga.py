import os
import struct

from scipy.interpolate.interpolate import RegularGridInterpolator

import numpy as np
from py_wake.wake_model import LinearSum, WakeModel
from numpy import newaxis as na
import re


class FugaWakeModel(WakeModel, LinearSum):
    ams = 5
    invL = 0
    args4deficit = ['WS_lk', 'WS_eff_lk', 'dw_jl', 'cw_jl', 'ct_lk']

    def __init__(self, LUT_path, windTurbines):
        WakeModel.__init__(self, windTurbines)
        self.load(LUT_path)

    def load(self, path):

        with open(path + 'CaseData.bin', 'rb') as fid:
            case_name = struct.unpack('127s', fid.read(127))[0]
            r = struct.unpack('d', fid.read(8))[0]
            zhub = struct.unpack('d', fid.read(8))[0]
            lo_level = struct.unpack('L', fid.read(4))[0]
            hi_level = struct.unpack('L', fid.read(4))[0]
            z0 = struct.unpack('d', fid.read(8))[0]
            zi = struct.unpack('d', fid.read(8))[0]
            ds = struct.unpack('d', fid.read(8))[0]
            closure = struct.unpack('L', fid.read(4))[0]
            zeta0 = struct.unpack('d', fid.read(8))[0]

        def psim(zeta):
            return self.ams * zeta

        if not zeta0 >= 0:
            raise NotImplementedError  # See Colonel.u2b.psim
        factor = 1 / (1 - (psim(zhub * self.invL) - psim(zeta0)) / np.log(zhub / z0))

        f = [f for f in os.listdir(path) if f.endswith("input.par")][0]
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

        self.du = np.array(mdu, dtype=np.float32).reshape((len(mdu), nyW // 2, nxW)) * factor
        self.z0 = z0
        self.x0 = nxW // 4
        self.dx = dx
        self.x = np.arange(-self.x0, nxW * 3 / 4) * dx
        self.y = np.arange(nyW // 2) * dy
        self.dy = dy
        self.z = z0 * np.exp(zlevels * self.dsAll)
        self.regularGridInterpolator = RegularGridInterpolator((self.z, self.y, self.x), self.du)

    def interpolate(self, x, y, z):
        x = np.maximum(np.minimum(x, self.x[-1]), self.x[0])
        y = np.maximum(np.minimum(y, self.y[-1]), self.y[0])
        z = np.maximum(np.minimum(z, self.z[-1]), self.z[0])
        return self.regularGridInterpolator((z, y, x))

    def calc_deficit(self, WS_lk, WS_eff_lk, dw_jl, cw_jl, ct_lk):
        mdu_jl = self.interpolate(dw_jl, cw_jl, 70)
        deficit_jlk = mdu_jl[:, :, na] * ct_lk[na] * WS_eff_lk**2 / WS_lk
        return -deficit_jlk


def main():
    if __name__ == '__main__':
        from py_wake.aep._aep import AEP
        from py_wake.examples.data.iea37 import iea37_path
        from py_wake.examples.data.iea37.iea37_reader import read_iea37_windrose,\
            read_iea37_windfarm
        from py_wake.site._site import UniformSite
        from py_wake.wind_turbines.iea37_wind_turbine import IEA37_WindTurbines

        _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
        n_wt = 16
        x, y, _ = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)

        site = UniformSite(freq, ti=0.75)
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

        import matplotlib.pyplot as plt
        x_j = np.linspace(-1500, 1500, 500)
        y_j = np.linspace(-1500, 1500, 300)
        from py_wake.tests.test_files import tfp
        path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'
        wake_model = FugaWakeModel(path, windTurbines)
        aep = AEP(site, windTurbines, wake_model)
        X, Y, Z = aep.wake_map(x_j, y_j, 110, x, y, wd=[0], ws=[9])
        plt.figure()
        c = plt.contourf(X, Y, Z, 100)
        plt.colorbar(c)

        plt.plot(x, y, '2k')
        for i, (x_, y_) in enumerate(zip(x, y)):
            plt.annotate(i, (x_, y_))
        plt.axis('equal')

        plt.show()


main()
