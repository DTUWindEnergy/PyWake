import numpy as np
from numpy import newaxis as na
from py_wake.tests import npt


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
        self.x = x
        self.V = V
        self.bounds = bounds
        self.method = method
        self.n = np.array([len(x) for x in x])
        self.x0 = np.array([x[0] for x in x])
        dx = np.array([x[min(1, len(x) - 1)] - x[0] for x in x])
        self.dx = np.where(dx == 0, 1, dx)
        self.irregular_axes = np.where([np.allclose(np.diff(x), dx) is False for dx, x in zip(self.dx, x)])[0]
        for i in self.irregular_axes:
            self.x[i] = np.r_[self.x[i], self.x[i][-1] + 1]
        self.V = np.asarray(V)
        if not np.all(self.V.shape[:len(self.n)] == self.n):
            raise ValueError("Lengths of x does not match shape of V")
        ui = np.array([[0], [1]])
        for _ in range(len(x) - 1):
            ui = np.array([(np.r_[ui, 0], np.r_[ui, 1]) for ui in ui])
            ui = ui.reshape((ui.shape[0] * ui.shape[1], ui.shape[2]))
        ui[:, dx == 0] = 0
        self.ui = ui

    def __call__(self, xp, method=None, bounds=None):
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
        method = method or self.method
        bounds = bounds or self.bounds
        assert method in ['linear', 'nearest'], 'method must be "linear" or "nearest"'
        assert bounds in ['check', 'limit', 'ignore'], 'bounds must be "check", "limit" or "ignore"'
        xp = np.asarray(xp)
        xpi = (xp - self.x0) / self.dx
        if len(self.irregular_axes):
            irreg_i = np.array([np.searchsorted(self.x[i], xp[:, i], side='right') - 1
                                for i in self.irregular_axes])
            irreg_x0 = np.array([np.asarray(self.x[i])[irreg_i] for i, irreg_i in zip(self.irregular_axes, irreg_i)])
            irreg_x1 = np.array([np.asarray(self.x[i])[irreg_i + 1]
                                 for i, irreg_i in zip(self.irregular_axes, irreg_i)])
            irreg_dx = irreg_x1 - irreg_x0
            xpi[:, self.irregular_axes] = irreg_i.T + (xp[:, self.irregular_axes] - irreg_x0.T) / irreg_dx.T

        if bounds == 'check' and (np.any(xpi < 0) or np.any(xpi + 1 > self.n[na])):
            if -xpi.min() > (xpi + 1 - self.n[na]).max():
                point, dimension = np.unravel_index(xpi.argmin(), np.atleast_2d(xpi).shape)
            else:
                point, dimension = np.unravel_index(((xpi + 1 - self.n[na])).argmax(), np.atleast_2d(xpi).shape)
            raise ValueError("Point %d, dimension %d with value %f is outside range %f-%f" %
                             (point, dimension, np.atleast_2d(xp)[point, dimension], self.x[dimension][0], self.x[dimension][-1]))
        if bounds == 'limit':
            xpi = np.minimum(np.maximum(xpi, 0), self.n - 1)
        xpi0 = xpi.astype(int)
        xpif = xpi - xpi0
        if method == 'nearest':
            xpif = np.round(xpif)

        indexes = (self.ui.T[:, :, na] + xpi0.T[:, na])

        indexes = np.minimum(indexes, (self.n - 1)[:, na, na])
        v = np.moveaxis(self.V[tuple(indexes)], [0, 1], [-2, -1])

        xpif1 = 1 - xpif
        # w = np.product([xpif10_.T[ui] for xpif10_, ui in zip(np.array([xpif1, xpif]).T, self.ui.T)], 0).T # slower
        # w = np.product(np.take_along_axis(xpif10[:, :, na], self.ui[na, na], 0).squeeze(), 2)  # even slower

        def mul_weight(weights, i):
            if i == xpif.shape[1]:
                return weights
            else:
                return np.r_[mul_weight(weights * xpif1[:, i], i + 1), mul_weight(weights * xpif[:, i], i + 1)]

        w = mul_weight(1, 0).reshape(-1, xpif.shape[0])

        return np.moveaxis((w * v).sum(-2), -1, 0)
