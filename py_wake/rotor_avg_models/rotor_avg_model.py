from py_wake.deficit_models.deficit_model import DeficitModel
import numpy as np
from abc import abstractmethod
from numpy import newaxis as na


class RotorAvgModel(DeficitModel):
    args4rotor_avg_deficit = ['hcw_ijlk', 'dh_ijl', 'D_dst_ijl']

    def set_wake_deficitModel(self, deficitModel):
        self.deficitModel = deficitModel
        self.args4deficit = deficitModel.args4deficit + \
            [a for a in self.args4rotor_avg_deficit if a not in deficitModel.args4deficit]

    def calc_deficit(self, D_dst_ijl, **kwargs):
        if D_dst_ijl is None:
            return self.deficitModel.calc_deficit(D_dst_ijl=D_dst_ijl, **kwargs)
        else:
            return self._calc_rotor_avg_deficit(D_dst_ijl=D_dst_ijl, **kwargs)

    @abstractmethod
    def _calc_rotor_avg_deficit(self):
        """Similar to calc_deficit, but with an extra point dimension to calculate the
        rotor average wind speed as a weighted average of a number of points instead of
        only the rotor center"""


class RotorCenter(RotorAvgModel):
    args4rotor_avg_deficit = ['D_dst_ijl']

    def _calc_rotor_avg_deficit(self, **kwargs):
        return self.deficitModel.calc_deficit(**kwargs)

    def _calc_layout_terms(self, **kwargs):
        self.deficitModel._calc_layout_terms(**kwargs)


class GridRotorAvgModel(RotorAvgModel):
    def __init__(self, nodes_x, nodes_y, nodes_weight=None):
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.nodes_weight = nodes_weight

    def _update_kwargs(self, hcw_ijlk, dh_ijl, D_dst_ijl, **kwargs):
        # add extra dimension, p, with 40 points distributed over the destination rotors
        R_dst_ijl = D_dst_ijl / 2
        hcw_ijlkp = hcw_ijlk[..., na] + R_dst_ijl[:, :, :, na, na] * self.nodes_x[na, na, na, na, :]
        dh_ijlp = dh_ijl[..., na] + R_dst_ijl[:, :, :, na] * self.nodes_y[na, na, na, :]
        new_kwargs = {'dh_ijl': dh_ijlp, 'hcw_ijlk': hcw_ijlkp}
        if 'cw_ijlk' in self.args4deficit:
            cw_ijlkp = np.sqrt(hcw_ijlkp**2 + dh_ijlp[:, :, :, na]**2)
            new_kwargs['cw_ijlk'] = cw_ijlkp

        new_kwargs.update({k: v[..., na] for k, v in kwargs.items() if k not in new_kwargs})
        return new_kwargs

    def _calc_rotor_avg_deficit(self, **kwargs):

        # add extra dimension, p, with 40 points distributed over the destination rotors
        kwargs = self._update_kwargs(**kwargs)
        deficit_ijlkp = self.deficitModel.calc_deficit(**kwargs)
        # Calculate weighted sum of deficit over the destination rotors
        if self.nodes_weight is None:
            return np.mean(deficit_ijlkp, -1)
        return np.sum(self.nodes_weight[na, na, na, na, :] * deficit_ijlkp, -1)

    def _calc_layout_terms(self, **kwargs):
        self.deficitModel._calc_layout_terms(**self._update_kwargs(**kwargs))


class EqGridRotorAvgModel(GridRotorAvgModel):
    def __init__(self, n):
        X, Y = np.meshgrid(np.linspace(-1, 1, n + 2)[1:-1], np.linspace(-1, 1, n + 2)[1:-1])
        m = (X**2 + Y**2) < 1
        GridRotorAvgModel.__init__(self,
                                   nodes_x=X[m].flatten(),
                                   nodes_y=Y[m].flatten())


class GQGridRotorAvgModel(GridRotorAvgModel):
    """Gauss Quadrature grid rotor average model"""

    def __init__(self, n_x, n_y):
        x, y, w = gauss_quadrature(n_x, n_y)
        m = (x**2 + y**2) < 1
        w = w[m]
        w /= w.sum()
        GridRotorAvgModel.__init__(self, nodes_x=x[m], nodes_y=y[m], nodes_weight=w)


class PolarGridRotorAvgModel(GridRotorAvgModel):
    def __init__(self, nodes_r, nodes_theta, nodes_weight):
        self.nodes_x = nodes_r * np.cos(nodes_theta + np.pi / 2)
        self.nodes_y = nodes_r * np.sin(nodes_theta + np.pi / 2)
        self.nodes_weight = nodes_weight


def gauss_quadrature(n_x, n_y):
    nodes_x, nodes_x_weight = np.polynomial.legendre.leggauss(n_x)
    nodes_y, nodes_y_weight = np.polynomial.legendre.leggauss(n_y)
    X, Y = np.meshgrid(nodes_x, nodes_y)
    weights = np.prod(np.meshgrid(nodes_x_weight, nodes_y_weight), 0) / 4
    return X.flatten(), Y.flatten(), weights.flatten()


def polar_gauss_quadrature(n_r, n_theta):
    x, y, w = gauss_quadrature(n_r, n_theta)
    return (x + 1) / 2, (y + 1) * np.pi, w
