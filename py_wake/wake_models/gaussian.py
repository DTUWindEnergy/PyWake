from py_wake.wake_model import WakeModel, SquaredSum
import numpy as np
from numpy import newaxis as na


class BastankhahGaussian(SquaredSum, WakeModel):
    """Implemented according to
    Bastankhah M and Porté-Agel F.
    A new analytical model for wind-turbine wakes.
    J. Renew. Energy. 2014;70:116-23.
    """
    args4deficit = ['WS_lk', 'D_src_l', 'dw_jl', 'cw_jl', 'ct_lk']

    def __init__(self, windTurbines):
        WakeModel.__init__(self, windTurbines)
        self.k = 0.0324555

    def calc_deficit(self, WS_lk, D_src_l, dw_jl, cw_jl, ct_lk):
        sqrt1ct = np.sqrt(1 - ct_lk)
        beta_lk = 1 / 2 * (1 + sqrt1ct) / sqrt1ct
        sigma_sqr_jlk = (self.k * dw_jl[:, :, na] / D_src_l[na, :, na] + .2 * np.sqrt(beta_lk)[na])**2
        exponent_jlk = -1 / (2 * sigma_sqr_jlk) * (cw_jl**2 / D_src_l[na, :]**2)[:, :, na]
        # maximum added to avoid sqrt of negative number
        radical_jlk = np.maximum(0, (1. - ct_lk[na, :, :] / (8. * sigma_sqr_jlk)))
        deficit_jlk = (WS_lk[na, :, :] * (1. - np.sqrt(radical_jlk)) * np.exp(exponent_jlk))
        return deficit_jlk


class IEA37SimpleBastankhahGaussian(SquaredSum, WakeModel):
    """Implemented according to
    https://github.com/byuflowlab/iea37-wflo-casestudies/blob/master/iea37-wakemodel.pdf

    Equivalent to BastankhahGaussian for beta=1/sqrt(8) ~ ct=0.9637188
    """
    args4deficit = ['WS_lk', 'D_src_l', 'dw_jl', 'cw_jl', 'ct_lk']

    def __init__(self, windTurbines):
        WakeModel.__init__(self, windTurbines)
        self.k = 0.0324555

    def calc_deficit(self, WS_lk, D_src_l, dw_jl, cw_jl, ct_lk):
        sigma_jl = self.k * dw_jl + D_src_l[na, :] / np.sqrt(8.)

        exponent_jl = -0.5 * (cw_jl / sigma_jl)**2
        radical_jlk = (1. - ct_lk[na, :, :] / (8. * sigma_jl[:, :, na]**2 / D_src_l[na, :, na]**2))
        deficit_jlk = (WS_lk[na, :, :] * (1. - np.sqrt(radical_jlk)) * np.exp(exponent_jl[:, :, na]))

        return deficit_jlk


def main():
    if __name__ == '__main__':
        from py_wake.aep_calculator import AEPCalculator
        from py_wake.examples.data.iea37 import iea37_path
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines

        # setup site, turbines and wakemodel
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

        wake_model = IEA37SimpleBastankhahGaussian(windTurbines)

        # calculate AEP
        aep_calculator = AEPCalculator(site, windTurbines, wake_model)
        aep = aep_calculator.calculate_AEP(x, y)[0].sum()

        # plot wake mape
        import matplotlib.pyplot as plt
        aep_calculator.plot_wake_map(wt_x=x, wt_y=y, wd=[0], ws=[9])
        plt.title('AEP: %.2f GWh' % aep)
        windTurbines.plot(x, y)
        plt.show()


main()
