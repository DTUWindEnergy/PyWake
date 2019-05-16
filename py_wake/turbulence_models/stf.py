from py_wake.turbulence_model import TurbulenceModel, MaxSum
import numpy as np
from numpy import newaxis as na
from py_wake.wake_models.noj import NOJ


class STFTurbulenceModel(MaxSum, TurbulenceModel):
    args4addturb = ['dw_jl', 'cw_jl', 'D_src_l', 'ct_lk', 'TI_lk']

    def calc_added_turbulence(self, dw_jl, cw_jl, D_src_l, ct_lk, TI_lk):
        """ Calculate the added turbulence intensity at locations specified by
        downstream distances (dw_jl) and crosswind distances (cw_jl)
        caused by the wake of a turbine (diameter: D_src_l, thrust coefficient: Ct_lk).

        Returns
        -------
        TI_eff_jlk: array:float
            Effective turbulence intensity [-]
        """
        s_jl = dw_jl / D_src_l
        # In the standard (see page 78), the maximal added TI is calculated as
        # TI_add = 0.9/(1.5 + 0.3*d*sqrt(V_hub/c))
        # where d is the downwind distance normalised by rotor diameter, c=1.0m/s
        # Here it is assumed Ct = 7/V_hub (see Eq. (3.12) of ST Frandsen's thesis)
        # thus, when using the acutal Ct, the function will be tranformed into:
        # TI_add = 0.9/(1.5 + 0.8*d/sqrt(Ct))

        TI_add_jlk = 0.9 / (1.5 + 0.8 * (dw_jl / D_src_l[na])[:, :, na] / np.sqrt(ct_lk)[na])

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')

            # Theta_w is the characteristic view angle defined in Eq. (3.18) of
            # ST Frandsen's thesis
            theta_w = (180.0 / np.pi * np.arctan(1 / s_jl) + 10) / 2

            # thetq denotes the acutally view angles
            theta = np.arctan(cw_jl / dw_jl) * 180.0 / np.pi

        # weights_jl = np.where(theta < 3 * theta_w, np.exp(-(theta / theta_w)**2), 0)
        weights_jl = np.where(theta < theta_w, np.exp(-(theta / theta_w)**2), 0)

        # the way effective added TI is calculated is derived from Eqs. (3.16-18)
        # in ST Frandsen's thesis
        TI_add_jlk = weights_jl[:, :, na] * (np.sqrt(TI_add_jlk**2 + TI_lk[na]**2) - TI_lk[na])
        return TI_add_jlk


class NOJ_STF(NOJ, STFTurbulenceModel):
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

        class NOJ_STF(NOJ, STFTurbulenceModel):
            pass

        wake_model = NOJ_STF(windTurbines)

        # calculate AEP
        aep_calculator = AEPCalculator(site, windTurbines, wake_model)
        aep = aep_calculator.calculate_AEP(x, y)[0].sum()

        # plot wake mape
        import matplotlib.pyplot as plt

        X, Y, Z = aep_calculator.ti_map(wt_x=x, wt_y=y, wd=[0], ws=[9])
        c = plt.contourf(X, Y, Z, levels=100, cmap='Blues')
        plt.colorbar(c, label='turbulence intensity [m/s]')
        plt.title('AEP: %.2f GWh' % aep)
        windTurbines.plot(x, y)
        plt.show()


main()
