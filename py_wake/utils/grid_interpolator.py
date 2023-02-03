from py_wake import np
from numpy import newaxis as na
from py_wake.utils import gradients
from autograd.numpy.numpy_boxes import ArrayBox


class GridInterpolator(object):
    # Faster than scipy.interpolate.interpolate.RegularGridInterpolator
    def __init__(self, x, V, method='linear', bounds='check'):
        """
        Parameters
        ----------
        x : list of array_like
            Interpolation coordinates
        V : array_like
            Interpolation data (more dimensions than coordinates is possible)
        method : {'linear' or 'nearest} or None
            Overrides self.method
        bounds : {'check', 'limit', 'ignore'}
            Specifies how bounds is handled:\n
            - 'check': bounds check is performed. An error is raised if interpolation point outside area
            - 'limit': interpolation points are forced inside the area
            - 'ignore': Faster option with no check. Use this option if data is guaranteed to be inside the area
        """
        self.x = x.copy()
        self.V = V
        self.bounds = bounds
        self.method = method
        self.n = np.array([len(x) for x in x], dtype=int)
        self.x0 = np.array([x[0] for x in x])
        dx = np.array([x[min(1, len(x) - 1)] - x[0] for x in x])
        self.dx = np.where(dx == 0, 1, dx)
        self.irregular_axes = np.array([np.allclose(np.diff(x), dx) is False for dx, x in zip(self.dx, x)])
        self.irregular_axes_indexes = np.where(self.irregular_axes)[0]
        for i in self.irregular_axes_indexes:
            self.x[i] = np.r_[self.x[i], self.x[i][-1] + 1]
        self.V = np.asarray(V)
        if not np.all(self.V.shape[:len(self.n)] == self.n):
            raise ValueError("Lengths of x does not match shape of V")
        ui = np.array([[0], [1]])
        for _ in range(len(x) - 1):
            ui = np.array([(np.r_[ui, 0], np.r_[ui, 1]) for ui in ui])
            ui = ui.reshape((ui.shape[0] * ui.shape[1], ui.shape[2]))
        ui[:, dx == 0] = 0
        self.ui = ui.astype(int)

    def __call__(self, xp, method=None, bounds=None, deg=False):
        """Interpolate points

        Parameters
        ----------
        xp : array_like
            Interpolation points, shape=(n_points, interpolation_dimensions)
        method : {'linear' or 'nearest} or None
            Overrides self.method if not None
        bounds : {'check', 'limit', 'ignore'} or None
            Overrides self.bounds if not None
        """
        if len(np.atleast_1d(xp)) == 0:
            return np.array([])
        method = np.atleast_1d(method or self.method)
        assert np.all([m in ['linear', 'nearest'] for m in method]), 'method must be "linear" or "nearest"'
        assert len(method) in [1, len(self.x)]
        linear = [method[min(len(method) - 1, i)] == 'linear' for i in range(len(self.x))]
        bounds = bounds or self.bounds
        assert bounds in ['check', 'limit', 'ignore'], 'bounds must be "check", "limit" or "ignore"'
        xp = np.atleast_2d(xp)
        xp_shape = xp.shape
        assert xp_shape[-1] == len(self.x), xp_shape
        xp = np.reshape(xp, (-1, xp_shape[-1]))
        if len(self.irregular_axes_indexes):
            xpi0 = np.array([np.clip(np.searchsorted(x, xp, side='right') - 1, 0, n - 2, dtype=int)
                             for x, xp, n in zip(self.x, xp.T, self.n)], dtype=int)
            xp0 = np.array([np.asarray(x)[xpi0] for x, xpi0 in zip(self.x, xpi0)])
            xp1 = np.array([np.asarray(x)[xpi0 + 1] for x, xpi0 in zip(self.x, xpi0)])
            xp_dx = xp1 - xp0
            xpi = xpi0.T + (xp - xp0.T) / xp_dx.T
        else:
            xpi = (xp - self.x0) / self.dx

        if bounds == 'check' and (np.any(xpi < 0) or np.any(xpi + 1 > self.n[na])):
            if -xpi.min() > (xpi + 1 - self.n[na]).max():
                point, dimension = np.unravel_index(xpi.argmin(), np.atleast_2d(xpi).shape)
            else:
                point, dimension = np.unravel_index(((xpi + 1 - self.n[na])).argmax(), np.atleast_2d(xpi).shape)
            raise ValueError("Point %d, dimension %d with value %f is outside range %f-%f" %
                             (point, dimension, np.atleast_2d(xp)[point, dimension], self.x[dimension][0], self.x[dimension][-1]))
        if bounds == 'limit':
            xpi = np.minimum(np.maximum(xpi, 0), self.n - 1)

        if 'nearest' in method:
            # round x.5 down to match old results
            xpi = np.where(linear, xpi, np.round(xpi - .1 * (gradients.mod(xpi, 2) == 1.5)))
        xpif, xpi0 = gradients.modf(xpi)

        int_box_axes = [[0, (0, 1)][l] for l in linear]
        ui = np.moveaxis(np.meshgrid(*int_box_axes, indexing='ij'), 0, -1)

        ui = ui.reshape((-1, len(self.x)))
        indexes = (ui.T[:, :, na] + xpi0.T[:, na])

        indexes = np.minimum(indexes, (self.n - 1)[:, na, na], dtype=int)
        v = np.moveaxis(self.V[tuple(indexes)], [0, 1], [-2, -1])
        if deg:
            v = (v + 180) % 360 - 180  # -180..180 > 0-360

        for i, x in enumerate(zip((1 - xpif).T, xpif.T)):
            if linear[i]:
                if i == 0:
                    w = np.array(x)
                else:
                    w = w[..., na, :] * np.expand_dims(np.array(x), tuple(range(i)))
            else:
                if i == 0:
                    w = x[0]
                else:
                    w = w * x[0]  # np.expand_dims(x[0], tuple(range(i)))
        w = np.reshape(w, (-1, xpif.shape[0]))

        # w = np.prod(np.array([xpif1, xpif])[ui, :, range(len(self.x))], 1) # slower
        # w = np.product(np.take_along_axis(xpif10[:, :, na], self.ui[na, na], 0).squeeze(), 2)  # even slower

        res = np.moveaxis((w * v).sum(-2), -1, 0)
        if deg:
            res = gradients.mod(res, 360)
        return np.reshape(res, xp_shape[:-1] + self.V.shape[len(self.x):])


class EqDistRegGrid2DInterpolator():
    def __init__(self, x, y, Z):
        self.x = x
        self.y = y
        self.Z = Z
        self.dx, self.dy = [xy[1] - xy[0] for xy in [x, y]]
        assert all(np.diff(x) == self.dx), "x is not equidistant"
        assert all(np.diff(y) == self.dy), "y is not equidistant"
        self.x0 = x[0]
        self.y0 = y[0]
        xi_valid = np.where(np.any(~np.isnan(self.Z), 1))[0]
        yi_valid = np.where(np.any(~np.isnan(self.Z), 0))[0]
        self.xi_valid_min, self.xi_valid_max = xi_valid[0], xi_valid[-1]
        self.yi_valid_min, self.yi_valid_max = yi_valid[0], yi_valid[-1]

    def __call__(self, x, y, mode='valid'):
        xp, yp = x, y

        xif, xi0 = gradients.modf((xp - self.x0) / self.dx)
        yif, yi0 = gradients.modf((yp - self.y0) / self.dy)

        if mode == 'extrapolate':
            xif[xi0 < self.xi_valid_min] = 0
            xif[xi0 > self.xi_valid_max - 2] = 1
            yif[yi0 < self.yi_valid_min] = 0
            yif[yi0 > self.yi_valid_max - 2] = 1
            xi0 = np.minimum(np.maximum(xi0, self.xi_valid_min), self.xi_valid_max - 2)
            yi0 = np.minimum(np.maximum(yi0, self.yi_valid_min), self.yi_valid_max - 2)
        xi1 = xi0 + (xif > 0)
        yi1 = yi0 + (yif > 0)
        if isinstance(xp, ArrayBox) or isinstance(yp, ArrayBox):
            valid = slice(None)
        else:
            valid = (xif >= 0) & (yif >= 0) & (xi1 < len(self.x)) & (yi1 < len(self.y))
        xi0, xi1, xif, yi0, yi1, yif = [v[valid] for v in [xi0, xi1, xif, yi0, yi1, yif]]
        z00 = self.Z[xi0, yi0]
        z10 = self.Z[xi1, yi0]
        z01 = self.Z[xi0, yi1]
        z11 = self.Z[xi1, yi1]
        z0 = z00 + (z10 - z00) * xif
        z1 = z01 + (z11 - z01) * xif
        if isinstance(xp, ArrayBox) or isinstance(yp, ArrayBox):
            z = z0 + (z1 - z0) * yif
        else:
            z = np.full(xp.shape, np.nan, dtype=xp.dtype)
            z[valid] = z0 + (z1 - z0) * yif
        return z
