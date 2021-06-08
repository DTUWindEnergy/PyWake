import os
import struct
import numpy as np
from pathlib import Path


class FugaUtils():
    def __init__(self, path, on_mismatch='raise'):
        """
        Parameters
        ----------
        path : string
            Path to folder containing 'CaseData.bin', input parameter file (*.par) and loop-up tables
        on_mismatch : {'raise', 'casedata','input_par'}
            Determines how to handle mismatch between info from CaseData.in and input.par.
            If 'raise' a ValueError exception is raised in case of mismatch\n
            If 'casedata', the values from CaseData.bin is used\n
            If 'input_par' the values from the input parameter file (*.par) is used
        """
        self.path = Path(path)
        if (self.path / 'CaseData.bin').exists():
            with open(self.path / 'CaseData.bin', 'rb') as fid:
                case_name = struct.unpack('127s', fid.read(127))[0]  # @UnusedVariable
                self.r = struct.unpack('d', fid.read(8))[0]  # @UnusedVariable
                self.zHub = struct.unpack('d', fid.read(8))[0]
                self.low_level = struct.unpack('I', fid.read(4))[0]
                self.high_level = struct.unpack('I', fid.read(4))[0]
                self.z0 = struct.unpack('d', fid.read(8))[0]
                zi = struct.unpack('d', fid.read(8))[0]  # @UnusedVariable
                self.ds = struct.unpack('d', fid.read(8))[0]
                closure = struct.unpack('I', fid.read(4))[0]  # @UnusedVariable
#                 if os.path.getsize(self.path / 'CaseData.bin') == 187:
#                     self.zeta0 = struct.unpack('d', fid.read(8))[0]
#                 else:
#                                    with open(path + 'CaseData.bin', 'rb') as fid2:
#                                        info = fid2.read(127).decode()
#                                    zeta0 = float(info[info.index('Zeta0'):].replace("Zeta0=", ""))

        else:
            with open(self.path / 'WTdata.bin', 'rb') as fid:
                case_name = struct.unpack('127s', fid.read(127))[0]  # @UnusedVariable
                prelut_name = struct.unpack('127s', fid.read(127))[0]  # @UnusedVariable
                self.r = struct.unpack('d', fid.read(8))[0]  # @UnusedVariable
                self.zHub = struct.unpack('d', fid.read(8))[0]
            on_mismatch = 'input_par'
        if not hasattr(self, 'zeta0') and 'Zeta0' in self.path.name:
            self.zeta0 = float(self.path.name[self.path.name.index('Zeta0'):].replace("Zeta0=", "").replace("/", ""))
        f = [f for f in os.listdir(self.path) if f.endswith('.par')][0]
        lines = (self.path / f).read_text().split("\n")

        self.prefix = lines[0].strip()
        self.nx, self.ny = map(int, lines[2:4])
        self.dx, self.dy = map(float, lines[4:6])  # @UnusedVariable
        self.sigmax, self.sigmay = map(float, lines[6:8])  # @UnusedVariable

        def set_Value(n, v):
            if on_mismatch == 'raise' and getattr(self, n) != v:
                raise ValueError("Mismatch between CaseData.bin and %s: %s %s!=%s" %
                                 (f, n, getattr(self, n), v))  # pragma: no cover
            elif on_mismatch == 'input_par':
                setattr(self, n, v)

        set_Value('low_level', int(lines[11]))
        set_Value('high_level', int(lines[12]))
        set_Value('z0', float(lines[8]))  # roughness level
        set_Value('zHub', float(lines[10]))  # hub height
        self.nx0 = self.nx // 4
        self.ny0 = self.ny // 2

        self.x = np.arange(-self.nx0, self.nx * 3 / 4) * self.dx  # rotor is located 1/4 downstream
        self.y = np.arange(self.ny // 2) * self.dy
        self.zlevels = np.arange(self.low_level, self.high_level + 1)

        if self.low_level == self.high_level == 9999:
            self.z = [self.zHub]
        else:
            if not hasattr(self, 'ds'):
                with open(self.path / "levels.txt") as fid:
                    level, z = fid.readline().split()
                self.ds = np.round(np.log(float(z) / self.z0) / int(level), 5)

            self.z = self.z0 * np.exp(self.zlevels * self.ds)

    def mirror(self, x, anti_symmetric=False):
        x = np.asarray(x)
        return np.concatenate([((1, -1)[anti_symmetric]) * x[::-1], x[1:]])

    def lut_exists(self, zlevels=None):
        return {uvwp_lt for uvwp_lt in ['UL', 'UT', 'VL', 'VT', 'WL', 'WT', 'PL', 'PT']
                if np.all([(self.path / (self.prefix + '%04d%s.dat' % (j, uvwp_lt))).exists()
                           for j in (zlevels or self.zlevels)])}

    def load_luts(self, UVLT=['UL', 'UT', 'VL', 'VT'], zlevels=None):
        luts = np.array([[np.fromfile(str(self.path / (self.prefix + '%04d%s.dat' % (j, uvlt))), np.dtype('<f'), -1)
                          for j in (zlevels or self.zlevels)] for uvlt in UVLT]).astype(float)
        return luts.reshape((len(UVLT), len(zlevels or self.zlevels), self.ny // 2, self.nx))
