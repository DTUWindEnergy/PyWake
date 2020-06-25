import numpy as np
from numpy import newaxis as na


class EqDistRegGridInterpolator(object):
    # Faster than scipy.interpolate.interpolate.RegularGridInterpolator
    def __init__(self, x, V, method='linear'):
        self.x = x
        self.n = np.array([len(x) for x in x])
        self.x0 = np.array([x[0] for x in x])
        self.dx = np.array([x[1] - x[0] for x in x])
        if not np.all([np.allclose(np.diff(x), dx) for dx, x in zip(self.dx, x)]):
            raise ValueError("Axes must be equidistant")
        self.V = np.asarray(V)
        if not np.all(self.V.shape[:len(self.n)] == self.n):
            raise ValueError("Lengths of x does not match shape of V")
        ui = np.array([[0], [1]])
        for _ in range(len(x) - 1):
            ui = np.array([(np.r_[ui, 0], np.r_[ui, 1]) for ui in ui])
            ui = ui.reshape((ui.shape[0] * ui.shape[1], ui.shape[2]))
        self.ui = ui
        self.method = method

    def __call__(self, xp, method=None):
        method = method or self.method
        if method not in ['linear', 'nearest']:
            raise ValueError('Method must be "linear" or "nearest"')
        xp = (xp - self.x0) / self.dx
        xi0 = xp.astype(int)
        xif = xp - xi0
        if method == 'nearest':
            xif = np.round(xif)

        indexes = np.minimum((self.ui.T[:, :, na] + xi0.T[:, na]), (self.n - 1)[:, na, na])
        v = self.V[tuple(indexes)].T

        w = np.product([np.choose(self.ui, [xif1, xif]) for xif, xif1 in zip(xif, 1 - xif)], 2)

        return (w * v).sum(-1)
