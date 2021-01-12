import numpy as np
from numpy import newaxis as na


class RotorAvgModel():
    """Wrap a DeficitModel.
    The RotorAvgModel
    - add an extra dimension (one or more points covering the downstream rotors)
    - Call the wrapped DeficitModel to calculate the deficit at all points
    - Compute a (weighted) mean of the deficit values covering the downstream rotors
    """
    args4rotor_avg_deficit = ['hcw_ijlk', 'dh_ijlk', 'D_dst_ijl']

    def __init__(self):
        pass

    def calc_deficit_convection(self, deficitModel, D_dst_ijl, **kwargs):
        self.deficitModel = deficitModel
        return self.deficitModel.calc_deficit_convection(D_dst_ijl=D_dst_ijl, **kwargs)

    def __call__(self, func, D_dst_ijl, **kwargs):
        # add extra dimension, p, with 40 points distributed over the destination rotors
        kwargs = self._update_kwargs(D_dst_ijl=D_dst_ijl, **kwargs)

        values_ijlkp = func(**kwargs)
        # Calculate weighted sum of deficit over the destination rotors
        if self.nodes_weight is None:
            return np.mean(values_ijlkp, -1)
        return np.sum(self.nodes_weight[na, na, na, na, :] * values_ijlkp, -1)


class RotorCenter(RotorAvgModel):
    args4rotor_avg_deficit = ['D_dst_ijl']
    nodes_x = [0]
    nodes_y = [0]
    nodes_weight = [1]

    def __call__(self, func, **kwargs):
        return func(**kwargs)

    def _calc_layout_terms(self, deficitModel, **kwargs):
        deficitModel._calc_layout_terms(**kwargs)


class GridRotorAvg(RotorAvgModel):
    nodes_weight = None

    def __init__(self, nodes_x, nodes_y, nodes_weight=None):
        self.nodes_x = np.asarray(nodes_x)
        self.nodes_y = np.asarray(nodes_y)
        if nodes_weight is not None:
            self.nodes_weight = np.asarray(nodes_weight)

    def _update_kwargs(self, hcw_ijlk, dh_ijlk, D_dst_ijl, **kwargs):
        # add extra dimension, p, with 40 points distributed over the destination rotors
        R_dst_ijl = D_dst_ijl / 2
        hcw_ijlkp = hcw_ijlk[..., na] + R_dst_ijl[:, :, :, na, na] * self.nodes_x[na, na, na, na, :]
        dh_ijlkp = dh_ijlk[..., na] + R_dst_ijl[:, :, :, na, na] * self.nodes_y[na, na, na, na, :]
        new_kwargs = {'dh_ijlk': dh_ijlkp, 'hcw_ijlk': hcw_ijlkp, 'D_dst_ijl': D_dst_ijl[..., na]}

        new_kwargs['cw_ijlk'] = np.sqrt(hcw_ijlkp**2 + dh_ijlkp**2)
        new_kwargs['D_dst_ijl'] = D_dst_ijl

        new_kwargs.update({k: v[..., na] for k, v in kwargs.items() if k not in new_kwargs})
        return new_kwargs

    def _calc_layout_terms(self, deficitModel, **kwargs):
        self.deficitModel = deficitModel
        self.deficitModel._calc_layout_terms(**self._update_kwargs(**kwargs))


class EqGridRotorAvg(GridRotorAvg):
    def __init__(self, n):
        X, Y = np.meshgrid(np.linspace(-1, 1, n + 2)[1:-1], np.linspace(-1, 1, n + 2)[1:-1])
        m = (X**2 + Y**2) < 1
        GridRotorAvg.__init__(self,
                              nodes_x=X[m].flatten(),
                              nodes_y=Y[m].flatten())


class GQGridRotorAvg(GridRotorAvg):
    """Gauss Quadrature grid rotor average model"""

    def __init__(self, n_x, n_y):
        x, y, w = gauss_quadrature(n_x, n_y)
        m = (x**2 + y**2) < 1
        w = w[m]
        w /= w.sum()
        GridRotorAvg.__init__(self, nodes_x=x[m], nodes_y=y[m], nodes_weight=w)


class PolarGridRotorAvg(GridRotorAvg):
    def __init__(self, nodes_r, nodes_theta, nodes_weight):
        self.nodes_x = nodes_r * np.cos(-nodes_theta - np.pi / 2)
        self.nodes_y = nodes_r * np.sin(-nodes_theta - np.pi / 2)
        self.nodes_weight = nodes_weight


class CGIRotorAvg(GridRotorAvg):
    """Circular Gauss Integration"""

    def __init__(self, n=7):
        """Circular Gauss Integration

        Parameters
        ----------
        n : {4, 7, 9, 21}
            Number of points.
        """
        pm = np.array([[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])
        nodes_x, nodes_y, nodes_weight = {
            # 1: np.array([[0, 0, .5], [-1, 0, 1 / 8], [1, 0, 1 / 8], [0, -1, 1 / 8], [0, 1, 1 / 8]]),
            4: pm * [0.5, 0.5, 1 / 4],
            # 3: np.r_[[[0, 0, 1 / 2], [-1, 0, 1 / 12], [1, 0, 1 / 12]], pm * [1 / 2, np.sqrt(3) / 2, 1 / 12]],
            7: np.r_[[[0, 0, 1 / 4], [-np.sqrt(2 / 3), 0, 1 / 8], [np.sqrt(2 / 3), 0, 1 / 8]],
                     pm * [np.sqrt(1 / 6), np.sqrt(1 / 2), 1 / 8]],
            9: np.r_[[[0, 0, 1 / 6], [-1, 0, 1 / 24], [1, 0, 1 / 24], [0, -1, 1 / 24], [0, 1, 1 / 24]],
                     pm * [1 / 2, 1 / 2, 1 / 6]],
            21: np.r_[[[0, 0, 1 / 9]],
                      [[np.sqrt((6 - np.sqrt(6)) / 10) * np.cos(2 * np.pi * k / 10),
                        np.sqrt((6 - np.sqrt(6)) / 10) * np.sin(2 * np.pi * k / 10),
                        (16 + np.sqrt(6)) / 360] for k in range(1, 11)],
                      [[np.sqrt((6 + np.sqrt(6)) / 10) * np.cos(2 * np.pi * k / 10),
                        np.sqrt((6 + np.sqrt(6)) / 10) * np.sin(2 * np.pi * k / 10),
                        (16 - np.sqrt(6)) / 360] for k in range(1, 11)]]
        }[n].T
        GridRotorAvg.__init__(self, nodes_x, nodes_y, nodes_weight=nodes_weight)


def gauss_quadrature(n_x, n_y):
    nodes_x, nodes_x_weight = np.polynomial.legendre.leggauss(n_x)
    nodes_y, nodes_y_weight = np.polynomial.legendre.leggauss(n_y)
    X, Y = np.meshgrid(nodes_x, nodes_y)
    weights = np.prod(np.meshgrid(nodes_x_weight, nodes_y_weight), 0) / 4
    return X.flatten(), Y.flatten(), weights.flatten()


def polar_gauss_quadrature(n_r, n_theta):
    x, y, w = gauss_quadrature(n_r, n_theta)
    return (x + 1) / 2, (y + 1) * np.pi, w
