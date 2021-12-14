from py_wake.utils.grid_interpolator import GridInterpolator
import numpy as np
from numpy import newaxis as na
import xarray as xr


class VectorField3D():
    def __init__(self, da):
        da = da[:, :-1]
        self.da = da
        self.interpolator = GridInterpolator(
            [da.wd.values, da.x.values, da.y.values, da.h.values], da.values, method='linear')
    vy = property(lambda self: self.da.sel(v_xyz=0))
    vx = property(lambda self: self.da.sel(v_xyz=1))
    vw = property(lambda self: self.da.sel(v_xyz=2))
    x = property(lambda self: self.da.x.values)
    y = property(lambda self: self.da.y.values)
    z = property(lambda self: self.da.y.values)

    def __call__(self, wd, x, y, h):
        return self.interpolator(np.array([np.atleast_1d(v) for v in [wd, x, y, h]]).T, bounds='limit')

    @staticmethod
    def from_WaspGridSite(site):
        ds = site.ds
        alpha = np.deg2rad(270 - (ds.wd + ds.Turning))
        beta = np.deg2rad(ds.flow_inc)

        da = xr.concat([np.cos(alpha), np.sin(alpha), np.tan(beta)], 'v_xyz') * ds.Speedup
        da.assign_coords(v_xyz=[0, 1, 2])
        return VectorField3D(da.transpose('wd', 'x', 'y', 'h', 'v_xyz'))

    def stream_lines(self, wd, start_points, dw_stop, step_size=20):
        stream_lines = [start_points]
        m = np.arange(len(wd))
        co, si = np.cos(np.deg2rad(270 - wd)), np.sin(np.deg2rad(270 - wd))
        for _ in range(1000):
            p = stream_lines[-1].copy()
            v = self(wd[m], p[m, 0], p[m, 1], p[m, 2])
            v = v / (np.sqrt(np.sum(v**2, -1)) / step_size)[:, na]  # normalize vector distance to step_size
            p[m] += v
            p[m, 2] = np.maximum(p[m, 2], 0)  # avoid underground flow
            stream_lines.append(p)

            # calculate downwind distance and update m (mask of streamlines to continue)
            dist = (p[m, :2] - start_points[m, :2])
            dw = dist[:, 0] * co[m] + dist[:, 1] * si[m]
            m = m[dw < dw_stop[m]]
            if len(m) == 0:
                break
        return np.moveaxis(stream_lines, 0, 1).reshape((len(wd), -1, 3))
