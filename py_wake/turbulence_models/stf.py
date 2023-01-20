import warnings

from numpy import newaxis as na

from py_wake import np
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.utils.gradients import hypot, cabs


class FrandsenWeight():
    """compute and apply a bell-shaped weight according to S. T. Frandsen's thesis
    https://orbit.dtu.dk/en/publications/turbulence-and-turbulence-generated-structural-loading-in-wind-tu

    The weight is given by the exponential term in Eq 3.16 and accounts
    for the lateral offset between the wake and the affected turbine.
    """

    def apply_weight(self, dw_ijlk, cw_ijlk, D_src_il, TI_ilk, TI_add_ijlk):
        s_ijlk = dw_ijlk / D_src_il[:, na, :, na]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')
            warnings.filterwarnings('ignore', r'divide by zero encountered in divide')
            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings('ignore', r'invalid value encountered in divide')

            # Theta_w is the characteristic view angle defined in Eq. (3.18)
            theta_w = (180.0 / np.pi * np.arctan(1 / s_ijlk) + 10) / 2

            # thetq denotes the acutally view angles
            theta = np.where(dw_ijlk > 0, np.arctan(cw_ijlk / dw_ijlk) * 180.0 / np.pi, 0)
        weights_ijlk = np.where(theta < theta_w, np.exp(-(theta / theta_w)**2), 0) * (dw_ijlk > 1e-10)

        # In Frandsens thesis, the weight is multiplied to I0 * alpha:
        # eq 3.16: I = I0 + I0 * alpha * weight
        # I0 is added in the LinearSum TurbulenceSuperpositionModel
        # so we need to multiply the weight to I0 * alpha
        # 3.17: I0 * alpha = sqrt(Iadd^2 + I0^2) - I0
        return weights_ijlk * (hypot(TI_add_ijlk, TI_ilk[:, na]) - TI_ilk[:, na])


class IECWeight():
    def __init__(self, distance_limit=10):
        self.dist_limit = distance_limit

    def apply_weight(self, dw_ijlk, cw_ijlk, D_src_il, TI_add_ijlk, **_):
        # In IEC 61400-1, 2005 Annex D and IEC 61400-1, 2017 Annex E the effective
        # turbulence intensity formula contains the term p_w * sigma_hat_T,
        # where p_w is 6% and sigma_hat_T is representative value of the maximum center-wake,
        # hub-height turbulence standard deviation.
        # This term is added to the representative ambient turbulence standard deviation
        # taking into account the wohler exponent of the considered material.
        # Note that this is a load-representative turbulence estimate.
        # Based on this formula, we assume that the increased turbulence level should
        # be added in a downwind angle of 360deg*6% = 21.6deg

        angleSpread = 21.6 / 2  # half angle
        r = np.tan(angleSpread * np.pi / 180.0) * dw_ijlk
        weights_ijlk = ((cabs(cw_ijlk) < cabs(r)) & (dw_ijlk > -1e-10) &
                        (hypot(dw_ijlk, cw_ijlk) < (self.dist_limit * D_src_il)[:, na, :, na]))
        return TI_add_ijlk * weights_ijlk


class STF2017TurbulenceModel(TurbulenceModel):
    """Steen Frandsen model implemented according to IEC61400-1, 2017"""

    def __init__(self, c=[1.5, 0.8], addedTurbulenceSuperpositionModel=LinearSum(),
                 weight_function=FrandsenWeight(), rotorAvgModel=None, groundModel=None):
        TurbulenceModel.__init__(self, addedTurbulenceSuperpositionModel,
                                 rotorAvgModel=rotorAvgModel, groundModel=groundModel)
        self.c = c
        self.apply_weight = weight_function.apply_weight

    def max_centre_wake_turbulence(self, dw_ijlk, cw_ijlk, D_src_il, ct_ilk, **_):
        dist_ijlk = hypot(dw_ijlk, cw_ijlk) / D_src_il[:, na, :, na]
        # In the standard (see page 103), the maximal added TI is calculated as
        # TI_add = 1/(1.5 + 0.8*d/sqrt(Ct))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')
            warnings.filterwarnings('ignore', r'divide by zero encountered in divide')
            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings('ignore', r'invalid value encountered in divide')
            return 1 / (self.c[0] + self.c[1] * dist_ijlk / np.sqrt(ct_ilk)[:, na])

    def calc_added_turbulence(self, WS_ilk, dw_ijlk, cw_ijlk, D_src_il, TI_ilk, ct_ilk, **kwargs):
        """ Calculate the added turbulence intensity at locations specified by
        downstream distances (dw_jl) and crosswind distances (cw_jl)
        caused by the wake of a turbine (diameter: D_src_l, thrust coefficient: Ct_lk).

        Returns
        -------
        TI_eff_ijlk: array:float
            Effective turbulence intensity [-]
        """
        # In the standard (see page 103), the maximal added TI is calculated as
        # TI_add = 1/(1.5 + 0.8*d/sqrt(Ct))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')
            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            TI_add_ijlk = self.max_centre_wake_turbulence(WS_ilk=WS_ilk, dw_ijlk=dw_ijlk, cw_ijlk=cw_ijlk,
                                                          D_src_il=D_src_il, ct_ilk=ct_ilk, **kwargs)
        # TI_add_ijlk[np.isnan(TI_add_ijlk)] = 0
        TI_add_ijlk = np.where(np.isnan(TI_add_ijlk), 0, TI_add_ijlk)

        return self.apply_weight(dw_ijlk, cw_ijlk, D_src_il, TI_ilk=TI_ilk, TI_add_ijlk=TI_add_ijlk)


class STF2005TurbulenceModel(STF2017TurbulenceModel):
    """Steen Frandsen model implemented according to IEC61400-1, 2005"""

    def max_centre_wake_turbulence(self, WS_ilk, dw_ijlk, D_src_il, **_):
        # In the standard (see page 73), the maximal added TI is calculated as
        # TI_add = 0.9/(1.5 + 0.3*d*sqrt(V_hub/c))
        return 0.9 / (1.5 + 0.3 * (dw_ijlk / D_src_il[:, na, :, na]) * np.sqrt(WS_ilk)[:, na])


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake import NOJ
        # setup site, turbines and flow model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        import matplotlib.pyplot as plt
        _, (ax1, ax2) = plt.subplots(1, 2)
        for ax, wf_model, lbl in [(ax1, NOJ(site, windTurbines, turbulenceModel=STF2005TurbulenceModel()), 'STF2005'),
                                  (ax2, NOJ(site, windTurbines, turbulenceModel=STF2017TurbulenceModel()), 'STF2017')]:

            sim_res = wf_model(x, y)
            print(sim_res.TI_eff_ilk[:, 0])
            # plot wake map
            flow_map = sim_res.flow_map(wd=0, ws=9.8)
            flow_map.plot_ti_map(ax=ax)
            ax.set_title('Turbulence intensity calculated by %s' % lbl)
        plt.show()


main()
