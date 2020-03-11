from numpy import newaxis as na
import numpy as np
from py_wake.turbulence_models.turbulence_model import TurbulenceModel, MaxSum


class STF2017TurbulenceModel(MaxSum, TurbulenceModel):
    """Steen Frandsen model implemented according to IEC61400-1, 2017"""

    args4addturb = ['dw_ijlk', 'cw_ijlk', 'D_src_il', 'ct_ilk', 'TI_ilk']

    def weight(self, dw_ijlk, cw_ijlk, D_src_il):
        s_ijlk = dw_ijlk / D_src_il[:, na, :, na]
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')

            # Theta_w is the characteristic view angle defined in Eq. (3.18) of
            # ST Frandsen's thesis
            theta_w = (180.0 / np.pi * np.arctan2(1, s_ijlk) + 10) / 2

            # thetq denotes the acutally view angles
            theta = np.arctan2(cw_ijlk, dw_ijlk) * 180.0 / np.pi

        # weights_jl = np.where(theta < 3 * theta_w, np.exp(-(theta / theta_w)**2), 0)
        weights_ijlk = np.where(theta < theta_w, np.exp(-(theta / theta_w)**2), 0)
        return weights_ijlk

    def calc_added_turbulence(self, dw_ijlk, cw_ijlk, D_src_il, ct_ilk, TI_ilk, **_):
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

        TI_add_ijlk = 1 / (1.5 + 0.8 * (dw_ijlk / D_src_il[:, na, :, na]) / np.sqrt(ct_ilk)[:, na])
        weights_ijlk = self.weight(dw_ijlk, cw_ijlk, D_src_il)
        # the way effective added TI is calculated is derived from Eqs. (3.16-18)
        # in ST Frandsen's thesis
        TI_add_ijlk = weights_ijlk * (np.hypot(TI_add_ijlk, TI_ilk[:, na]) - TI_ilk[:, na])
        return TI_add_ijlk


class STF2005TurbulenceModel(STF2017TurbulenceModel):
    """Steen Frandsen model implemented according to IEC61400-1, 2005"""

    args4addturb = ['dw_ijlk', 'cw_ijlk', 'D_src_il', 'WS_ilk', 'TI_ilk']

    def calc_added_turbulence(self, dw_ijlk, cw_ijlk, D_src_il, WS_ilk, TI_ilk, **_):
        """ Calculate the added turbulence intensity at locations specified by
        downstream distances (dw_jl) and crosswind distances (cw_jl)
        caused by the wake of a turbine (diameter: D_src_l, thrust coefficient: Ct_lk).

        Returns
        -------
        TI_eff_jlk: array:float
            Effective turbulence intensity [-]
        """

        # In the standard (see page 74), the maximal added TI is calculated as
        # TI_add = 0.9/(1.5 + 0.3*d*sqrt(V_hub/c))

        TI_add_ijlk = 0.9 / (1.5 + 0.3 * (dw_ijlk / D_src_il[:, na, :, na]) * np.sqrt(WS_ilk)[:, na])

        weights_ijlk = self.weight(dw_ijlk, cw_ijlk, D_src_il)

        # the way effective added TI is calculated is derived from Eqs. (3.16-18)
        # in ST Frandsen's thesis
        TI_add_ijlk = weights_ijlk * (np.hypot(TI_add_ijlk, TI_ilk[:, na]) - TI_ilk[:, na])
        return TI_add_ijlk


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
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for ax, wf_model, lbl in [(ax1, NOJ(site, windTurbines, turbulenceModel=STF2005TurbulenceModel()), 'STF2005'),
                                  (ax2, NOJ(site, windTurbines, turbulenceModel=STF2017TurbulenceModel()), 'STF2017')]:

            sim_res = wf_model(x, y)
            print(sim_res.TI_eff_ilk[:, 0])
            # plot wake map
            flow_map = sim_res.flow_map(wd=0, ws=9.8)
            c = flow_map.plot_ti_map(ax=ax, plot_colorbar=False)
            ax.set_title('Turbulence intensity calculated by %s' % lbl)
        cbaxes = fig.add_axes([.92, 0.1, 0.01, 0.8])
        plt.colorbar(c, cax=cbaxes, label='turbulence intensity [m/s]')
        plt.show()


main()
