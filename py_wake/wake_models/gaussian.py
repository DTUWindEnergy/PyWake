from py_wake.wake_model import WakeModel, SquaredSum
import numpy as np
from numpy import newaxis as na


class BastankhahGaussian(SquaredSum, WakeModel):
    """Implemented according to
    Bastankhah M and Porté-Agel F.
    A new analytical model for wind-turbine wakes.
    J. Renew. Energy. 2014;70:116-23.
    """
    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijl', 'cw_ijl', 'ct_ilk']

    def __init__(self, site, windTurbines, k=0.0324555, **kwargs):
        WakeModel.__init__(self, site, windTurbines, **kwargs)
        self.k = k

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijl, cw_ijl, ct_ilk):
        sqrt1ct_ilk = np.sqrt(1 - ct_ilk)
        beta_ilk = 1 / 2 * (1 + sqrt1ct_ilk) / sqrt1ct_ilk
        sigma_sqr_ijlk = (self.k * dw_ijl[..., na] / D_src_il[:, na, :, na] + .2 * np.sqrt(beta_ilk)[:, na])**2
        exponent_ijlk = -1 / (2 * sigma_sqr_ijlk) * (cw_ijl**2 / D_src_il[:, na, :]**2)[..., na]
        # maximum added to avoid sqrt of negative number
        radical_ijlk = np.maximum(0, (1. - ct_ilk[:, na] / (8. * sigma_sqr_ijlk)))
        deficit_ijlk = (WS_ilk[:, na] * (1. - np.sqrt(radical_ijlk)) * np.exp(exponent_ijlk))
        return deficit_ijlk


class IEA37SimpleBastankhahGaussian(SquaredSum, WakeModel):
    """Implemented according to
    https://github.com/byuflowlab/iea37-wflo-casestudies/blob/master/iea37-wakemodel.pdf

    Equivalent to BastankhahGaussian for beta=1/sqrt(8) ~ ct=0.9637188
    """
    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijl', 'cw_ijl', 'ct_ilk']
    args4update = ['ct_ilk']

    def __init__(self, site, windTurbines, **kwargs):
        WakeModel.__init__(self, site, windTurbines, **kwargs)
        self.k = 0.0324555

    def _calc_layout_terms(self, WS_ilk, D_src_il, dw_ijl, cw_ijl, **_):
        sigma_ijl = self.k * dw_ijl * (dw_ijl > 0) + (D_src_il / np.sqrt(8.))[:, na]
        self.layout_factor_ijlk = WS_ilk[:, na] * (dw_ijl > 0)[..., na] * \
            np.exp(-0.5 * (cw_ijl / sigma_ijl)**2)[..., na]
        self.denominator_ijlk = 8. * (sigma_ijl[..., na] / D_src_il[:, na, :, na])**2

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijl, cw_ijl, ct_ilk, **_):
        if not self.deficit_initalized:
            self._calc_layout_terms(WS_ilk, D_src_il, dw_ijl, cw_ijl)
        ct_factor_ijlk = (1. - ct_ilk[:, na] / self.denominator_ijlk)
        return self.layout_factor_ijlk * (1 - np.sqrt(ct_factor_ijlk))  # deficit_ijlk


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

        wake_model = IEA37SimpleBastankhahGaussian(site, windTurbines)

        # calculate AEP
        aep_calculator = AEPCalculator(wake_model)
        aep = aep_calculator.calculate_AEP(x, y)[0].sum()

        # plot wake mape
        import matplotlib.pyplot as plt
        aep_calculator.plot_wake_map(wt_x=x, wt_y=y, wd=[0], ws=[9])
        plt.title('AEP: %.2f GWh' % aep)
        windTurbines.plot(x, y)
        plt.show()


main()
