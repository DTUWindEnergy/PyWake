from py_wake.turbulence_model import TurbulenceModel, MaxSum
import numpy as np
from numpy import newaxis as na
from py_wake.wake_models.noj import NOJ
from matplotlib.font_manager import weight_dict


class STF2017TurbulenceModel(MaxSum, TurbulenceModel):
    """Steen Frandsen model implemented according to IEC61400-1, 2017"""

    args4addturb = ['dw_ijl', 'cw_ijl', 'D_src_il', 'ct_ilk', 'TI_ilk']

    def weight(self, dw_ijl, cw_ijl, D_src_il):
        s_ijl = dw_ijl / D_src_il[:, na]
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')

            # Theta_w is the characteristic view angle defined in Eq. (3.18) of
            # ST Frandsen's thesis
            theta_w = (180.0 / np.pi * np.arctan2(1, s_ijl) + 10) / 2

            # thetq denotes the acutally view angles
            theta = np.arctan2(cw_ijl, dw_ijl) * 180.0 / np.pi

        # weights_jl = np.where(theta < 3 * theta_w, np.exp(-(theta / theta_w)**2), 0)
        weights_ijl = np.where(theta < theta_w, np.exp(-(theta / theta_w)**2), 0)
        return weights_ijl

    def calc_added_turbulence(self, dw_ijl, cw_ijl, D_src_il, ct_ilk, TI_ilk):
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

        TI_add_ijlk = 1 / (1.5 + 0.8 * (dw_ijl / D_src_il[:, na])[..., na] / np.sqrt(ct_ilk)[:, na])
        weights_ijl = self.weight(dw_ijl, cw_ijl, D_src_il)
        # the way effective added TI is calculated is derived from Eqs. (3.16-18)
        # in ST Frandsen's thesis
        TI_add_ijlk = weights_ijl[..., na] * (np.hypot(TI_add_ijlk, TI_ilk[:, na]) - TI_ilk[:, na])
        return TI_add_ijlk


class STF2005TurbulenceModel(STF2017TurbulenceModel):
    """Steen Frandsen model implemented according to IEC61400-1, 2005"""

    args4addturb = ['dw_ijl', 'cw_ijl', 'D_src_il', 'WS_ilk', 'TI_ilk']

    def calc_added_turbulence(self, dw_ijl, cw_ijl, D_src_il, WS_ilk, TI_ilk):
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

        TI_add_ijlk = 0.9 / (1.5 + 0.3 * (dw_ijl / D_src_il[:, na])[..., na] * np.sqrt(WS_ilk)[:, na])

        weights_ijl = self.weight(dw_ijl, cw_ijl, D_src_il)

        # the way effective added TI is calculated is derived from Eqs. (3.16-18)
        # in ST Frandsen's thesis
        TI_add_ijlk = weights_ijl[..., na] * (np.hypot(TI_add_ijlk, TI_ilk[:, na]) - TI_ilk[:, na])
        return TI_add_ijlk


class NOJ_STF2005(NOJ, STF2005TurbulenceModel):
    pass


class NOJ_STF2017(NOJ, STF2017TurbulenceModel):
    pass


def main():
    if __name__ == '__main__':
        from py_wake.aep_calculator import AEPCalculator
        from py_wake.examples.data.iea37 import iea37_path
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.wake_models.noj import NOJ
        # setup site, turbines and wakemodel
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for ax, wake_model, lbl in [  # (ax1, NOJ_STF2005(site, windTurbines), 'STF2005'),
                (ax2, NOJ_STF2017(site, windTurbines), 'STF2017')]:

            aep_calculator = AEPCalculator(wake_model)
            aep_calculator.calculate_AEP(x, y)
            print(aep_calculator.TI_eff_ilk[:, 0])
            # plot wake map
            X, Y, Z = aep_calculator.ti_map(wt_x=x, wt_y=y, wd=[0], ws=[9.8])
            c = ax.contourf(X, Y, Z, levels=np.arange(0.075, .7, .01), cmap='Blues')

            ax.set_title('Turbulence intensity calculated by %s' % lbl)
            windTurbines.plot(x, y, ax=ax)
        cbaxes = fig.add_axes([.92, 0.1, 0.01, 0.8])
        plt.colorbar(c, cax=cbaxes, label='turbulence intensity [m/s]')
        plt.show()


main()
