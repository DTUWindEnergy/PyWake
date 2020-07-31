import numpy as np
from numpy import newaxis as na
from py_wake.tests import npt


class GridInterpolator(object):
    # Faster than scipy.interpolate.interpolate.RegularGridInterpolator
    def __init__(self, x, V, method='linear'):
        self.x = x
        self.n = np.array([len(x) for x in x])
        self.x0 = np.array([x[0] for x in x])
        self.dx = np.array([x[1] - x[0] for x in x])
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
        self.ui = ui
        self.method = method

    def __call__(self, xp, method=None, check_bounds=True):
        method = method or self.method
        if method not in ['linear', 'nearest']:
            raise ValueError('Method must be "linear" or "nearest"')
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

        if check_bounds and (np.any(xpi < 0) or np.any(xpi + 1 > self.n[na])):
            raise ValueError("Outside data area")
        xpi0 = xpi.astype(int)
        xpif = xpi - xpi0
        if method == 'nearest':
            xpif = np.round(xpif)

        indexes = (self.ui.T[:, :, na] + xpi0.T[:, na])

        indexes = np.minimum(indexes, (self.n - 1)[:, na, na])
        v = self.V[tuple(indexes)].T

        xpif1 = 1 - xpif
        # w = np.product([xpif10_.T[ui] for xpif10_, ui in zip(np.array([xpif1, xpif]).T, self.ui.T)], 0).T # slower
        # w = np.product(np.take_along_axis(xpif10[:, :, na], self.ui[na, na], 0).squeeze(), 2)  # even slower

        def mul_weight(weights, i):
            if i == xpif.shape[1]:
                return weights
            else:
                return np.r_[mul_weight(weights * xpif1[:, i], i + 1), mul_weight(weights * xpif[:, i], i + 1)]

        w = mul_weight(1, 0).reshape(-1, xpif.shape[0]).T

        return (w * v).sum(-1)
