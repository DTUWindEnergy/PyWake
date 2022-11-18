import os
import struct
from py_wake import np
import xarray as xr
from pathlib import Path
import warnings
from numpy import newaxis as na
from py_wake.utils import gradients


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
        if os.path.isdir(path):
            warnings.warn(f"""Using fuga with a folder of *.dat files is deprecated.
Please convert the data to the new netcdf format:

from py_wake.utils.fuga_utils import dat2netcdf
dat2netcdf(folder='{Path(path).as_posix()}')

After that pass the new netcdf file to Fuga instead of the old folder""",
                          DeprecationWarning, stacklevel=2)

            self.path = Path(path)
            if (self.path / 'CaseData.bin').exists():
                with open(self.path / 'CaseData.bin', 'rb') as fid:
                    case_name = struct.unpack('127s', fid.read(127))[0]  # @UnusedVariable
                    self.r = struct.unpack('d', fid.read(8))[0]  # @UnusedVariable
                    self.zHub = struct.unpack('d', fid.read(8))[0]
                    self.low_level = struct.unpack('I', fid.read(4))[0]
                    self.high_level = struct.unpack('I', fid.read(4))[0]
                    self.z0 = struct.unpack('d', fid.read(8))[0]
                    self.zi = struct.unpack('d', fid.read(8))[0]  # @UnusedVariable
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
                self.zeta0 = float(self.path.name[self.path.name.index(
                    'Zeta0'):].replace("Zeta0=", "").replace("/", ""))
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
            set_Value('zi', float(lines[9]))  # inversion height
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
        else:
            ds = xr.open_dataset(path)
            self.dataset_path = path
            self.x, self.y, self.z = ds.x.values, ds.y.values, ds.z.values
            self.dx, self.dy = np.diff(self.x[:2]), np.diff(self.y[:2])
            self.zeta0, self.zHub, self.z0 = ds.zeta0.item(), ds.hubheight.item(), ds.z0.item()
            self.ds = ds.ds.item()

    def mirror(self, x, anti_symmetric=False):
        x = np.asarray(x)
        return np.concatenate([((1, -1)[anti_symmetric]) * x[::-1], x[1:]])

    def lut_exists(self, zlevels=None):
        if hasattr(self, 'dataset_path'):
            return [k for k in ['UL', 'UT', 'VL', 'VT', 'WL', 'WT', 'PL', 'PT']
                    if k in xr.open_dataset(self.dataset_path)]
        else:
            return {uvwp_lt for uvwp_lt in ['UL', 'UT', 'VL', 'VT', 'WL', 'WT', 'PL', 'PT']
                    if np.all([(self.path / (self.prefix + '%04d%s.dat' % (j, uvwp_lt))).exists()
                               for j in (zlevels or self.zlevels)])}

    def load_luts(self, UVLT=['UL', 'UT', 'VL', 'VT'], zlevels=None):
        if hasattr(self, 'dataset_path'):
            dataset = xr.load_dataset(self.dataset_path)
            return np.array([dataset[k].load().transpose('z', 'y', 'x').values for k in UVLT])
        else:
            luts = np.array([[np.fromfile(str(self.path / (self.prefix + '%04d%s.dat' % (j, uvlt))), np.dtype('<f'), -1)
                              for j in (zlevels or self.zlevels)] for uvlt in UVLT])
            return luts.reshape((len(UVLT), len(zlevels or self.zlevels), self.ny // 2, self.nx))

    def zeta0_factor(self, zhub):
        ams = 5
        invL = 0

        def psim(zeta):
            return ams * zeta

        if not self.zeta0 >= 0:  # pragma: no cover
            # See Colonel.u2b.psim
            raise NotImplementedError
        return 1 / (1 - (psim(zhub * invL) - psim(self.zeta0)) / np.log(zhub / self.z0))

    def init_lut(self, lut, zhub, smooth2zero_x=None, smooth2zero_y=None, remove_wriggles=False):
        """initialize lut (negate, remove wriggles and smooth edges to zero)

        Parameters
        ----------
        lut : array_like
            Look-up data table
        smooth2zero_x : int or None:
            if None, default, smooth2zero_x is set to 1/8 of the box length
            if 0, no correction is applied.
            if >0, the first and last <smooth2zero_x> points are linearly faded to zero
        smooth2zero_y : int or None:
            if None, default, smooth2zero_y is set to 1/8 of the box width (i.e. center line to the side)
            if 0, no correction is applied.
            if >0, the <smooth2zero_x> points farthest away from the centerline are linearly faded to zero
        remove_wriggles : boolean
            if True, the lut is traversed from the centerline to the side and all values after the
            first zero crossing is set to zero. This means that all wriggles as well and speed-ups are removed

        Returns
        -------
        lut : array_like
            resulting lut

        """
        lut = -lut * self.zeta0_factor(zhub)  # minus to get deficit

        if remove_wriggles:
            # remove all positive and negative deficits after first zero crossing in lateral direction
            lut *= (np.cumsum(lut < 0, 1) == 0)

        # smooth edges to zero
        if smooth2zero_x is None:
            smooth2zero_x = lut.shape[2] // 8
        if smooth2zero_x:
            n = smooth2zero_x
            lut[:, :, :n] = lut[:, :, n][:, :, na] * np.arange(n) / n
            lut[:, :, -n:] = lut[:, :, -n][:, :, na] * np.arange(n)[::-1] / n
        if smooth2zero_y is None:
            smooth2zero_y = lut.shape[1] // 8
        if smooth2zero_y:
            n = smooth2zero_y
            lut[:, -n:, :] = lut[:, -n, :][:, na, :] * np.arange(n)[::-1][na, :, na] / n
        return lut

    @property
    def TI(self):
        """Streamwise Turbulence intensity"""
        return ti(self.z0, self.zHub)


def dat2netcdf(folder):
    """Convert Fuga LUT from old folder-of-dat-files format to the new netcdf

    Parameters
    ----------
    folder : str or path
        Folder containing dat files

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        fu = FugaUtils(folder)

    lut_vars = [k for k in ['UL', 'UT', 'VL', 'VT', 'WL', 'WT', 'PL', 'PT']
                if (fu.path / (fu.prefix + '%04d%s.dat' % (fu.zlevels[0], k))).exists()]
    luts_dict = {k: (('z', 'y', 'x'), fu.load_luts([k])[0].astype(np.float32)) for k in lut_vars}
    ds = xr.Dataset({'diameter': fu.r * 2, 'hubheight': fu.zHub, 'z0': fu.z0,
                     **luts_dict},
                    coords={'x': fu.x, 'y': fu.y, 'z': fu.z},
                    attrs=dict(ds=getattr(fu, 'ds', 0.05), zeta0=fu.zeta0, zi=fu.zi))

    L_vars = [v[0] for v in lut_vars if v[1] == 'L']
    T_vars = [v[0] for v in lut_vars if v[1] == 'T']
    lut_vars_id = ""
    if L_vars:
        lut_vars_id += f"_{''.join(L_vars)}L"
    if T_vars:
        lut_vars_id += f"_{''.join(T_vars)}T"

    if len(fu.z) == 1:
        z_id = f"z{fu.zHub:.1f}"
    else:
        z_id = f"z{fu.z[0]:.1f}-{fu.z[-1]:.1f}"

    preluts_id = f'Zeta0={fu.zeta0:3.2f}'
    fluts_id = f'_D{fu.r*2}_zhub{fu.zHub}_zi{fu.zi}_z0={fu.z0}_{z_id}{lut_vars_id}'
    luts_id = f'_nx{fu.nx}_ny{fu.ny}_dx{fu.dx}_dy{fu.dy}'
    filename = Path(folder).parent / (preluts_id + fluts_id + luts_id + ".nc")
    ds.to_netcdf(filename)
    ds.attrs['filename'] = filename
    print(f"""LUTs from '{folder}' converted to new netcdf format and
saved as '{filename}'""")
    return ds


def ti(z0, zhub):
    return 1 / np.log(np.asarray(zhub) / np.asarray(z0))


def z0(ti, zhub):
    return np.asarray(zhub) / np.exp(1 / np.asarray(ti))


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

        xif, xi0 = gradients.modf((xp - self.x0) / self.dx)
        yif, yi0 = gradients.modf((yp - self.y0) / self.dy)

        zif, zi0 = gradients.modf(gradients.interp(zp, self.z, np.arange(self.nz)))

        nx, ny = self.nx, self.ny
        idx = zi0 * nx * ny + yi0 * nx + xi0
        v000, v001, v010, v011, v100, v101, v110, v111 = self.V000[:, idx]
        if len(self.V000.shape) == 3:
            # Both UL and UT
            xif = xif[..., na]
            yif = yif[..., na]
            zif = zif[..., na]
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
