from py_wake.wake_model import WakeModel, SquaredSum
import numpy as np
from numpy import newaxis as na


class IEA37SimpleBastankhahGaussian(WakeModel, SquaredSum):

    def __init__(self, windTurbines):
        WakeModel.__init__(self, windTurbines)
        self.k = 0.0324555

    def calc_deficit(self, WS_lk, D_src_l, D_dst_j, dw_jl, cw_jl, ct_lk):

        # Calculate the wake loss using
        # simplified Bastankhah Gaussian wake model
        sigma_jl = self.k * dw_jl + D_src_l[na, :] / np.sqrt(8.)

        exponent_jl = -0.5 * (cw_jl / sigma_jl)**2
        radical_jlk = (1. - ct_lk[na, :, :] / (8. * sigma_jl[:, :, na]**2 / D_src_l[na, :, na]**2))
        deficit_jlk = (WS_lk[na, :, :] * (1. - np.sqrt(radical_jlk)) * np.exp(exponent_jl[:, :, na]))

        return deficit_jlk
