import numpy as np
from numpy import newaxis as na


class AEPCalculator():

    def __init__(self, site, windTurbines, wake_model):
        """
        site: f(turbine_positions, wd, ws) -> WD[nWT,nWdir,nWsp], WS[nWT,nWdir,nWsp], TI[nWT,nWdir,nWsp), Weight[nWdir,nWsp]
        wake_model: f(turbine_positions, WD[nWT,nWdir,nWsp], WS[nWT,nWdir,nWsp], TI[nWT,nWdir,nWsp) -> power[nWdir,nWsp] (W)
        """
        self.site = site
        self.wake_model = wake_model
        self.windTurbines = windTurbines

    def _get_defaults(self, x_i, h_i, type_i, wd, ws):
        if type_i is None:
            type_i = np.zeros_like(x_i)
        if h_i is None:
            h_i = self.windTurbines.hub_height(type_i)
        if wd is None:
            wd = self.site.default_wd
        if ws is None:
            ws = self.site.default_ws
        return h_i, type_i, wd, ws

    def _run_wake_model(self, x_i, y_i, h_i=None, type_i=None, wd=None, ws=None):
        h_i, type_i, wd, ws = self._get_defaults(x_i, h_i, type_i, wd, ws)
        # Find local wind speed, wind direction, turbulence intensity and probability
        self.WD_ilk, self.WS_ilk, self.TI_ilk, self.P_lk = self.site.local_wind(x_i=x_i, y_i=y_i, wd=wd, ws=ws)

        # Calculate down-wind and cross-wind distances
        dw_iil, cw_iil, dh_iil, dw_order_l = self.site.wt2wt_distances(x_i, y_i, h_i, self.WD_ilk.mean(2))

        self.WS_eff_ilk, self.TI_eff_ilk, self.power_ilk, self.ct_ilk =\
            self.wake_model.calc_wake(self.WS_ilk, self.TI_ilk, dw_iil, cw_iil, dh_iil, dw_order_l, type_i)

    def calculate_AEP(self, x_i, y_i, h_i=None, type_i=None, wd=None, ws=None):
        self._run_wake_model(x_i=x_i, y_i=y_i, h_i=h_i, type_i=type_i, wd=wd, ws=ws)
        AEP_GWh_ilk = self.power_ilk * self.P_lk[na, :, :] * 24 * 365 * 1e-9
        return AEP_GWh_ilk

    def calculate_AEP_no_wake_loss(self, x_i, y_i, h_i=None, type_i=None, wd=None, ws=None):
        h_i, type_i, wd, ws = self._get_defaults(x_i, h_i, type_i, wd=wd, ws=ws)

        # Find local wind speed, wind direction, turbulence intensity and probability
        self.WD_ilk, self.WS_ilk, self.TI_ilk, self.P_lk = self.site.local_wind(x_i=x_i, y_i=y_i, wd=wd, ws=ws)

        type_ilk = np.zeros(self.WS_ilk.shape, dtype=np.int) + type_i[:, np.newaxis, np.newaxis]
        _ct_ilk, self.power_ilk = self.windTurbines.ct_power(self.WS_ilk, type_ilk)
        AEP_GWh_ilk = self.power_ilk * self.P_lk[na, :, :] * 24 * 365 * 1e-9
        return AEP_GWh_ilk

    def WS_eff_map(self, x_j, y_j, h, x_i, y_i, type_i, h_i, wd=None, ws=None):
        h_i, type_i, wd, ws = self._get_defaults(x_i, h_i, type_i, wd, ws)

        def f(x, N=500, ext=.2):
            ext *= (max(x) - min(x))
            return np.linspace(min(x) - ext, max(x) + ext, N)

        if x_j is None:
            x_j = f(x_i)
        if y_j is None:
            y_j = f(y_i)
        if h is None:
            h = np.mean(h_i)

        X_j, Y_j = np.meshgrid(x_j, y_j)
        x_j, y_j = X_j.flatten(), Y_j.flatten()
        if len(x_i) == 0:
            _, WS_jlk, _, P_lk = self.site.local_wind(x_i=x_j, y_i=y_j, wd=wd, ws=ws, wd_bin_size=1)
            return X_j, Y_j, WS_jlk, P_lk

        self._run_wake_model(x_i, y_i, h_i, type_i, wd, ws)

        h_j = np.zeros_like(x_j) + h
        _, WS_jlk, _, P_lk = self.site.local_wind(x_i=x_j, y_i=y_j, wd=wd, ws=ws)
        dw_ijl, cw_ijl, dh_ijl, _ = self.site.distances(x_i, y_i, h_i, x_j, y_j, h_j, self.WD_ilk.mean(2))
        WS_eff_jlk = self.wake_model.wake_map(self.WS_ilk, self.WS_eff_ilk, dw_ijl,
                                              cw_ijl, dh_ijl, self.ct_ilk, type_i, WS_jlk)

        return X_j, Y_j, WS_eff_jlk, P_lk

    def wake_map(self, x_j=None, y_j=None, h=None, wt_x=[], wt_y=[], wt_type=None, wt_height=None, wd=None, ws=None):
        X_j, Y_j, WS_eff_jlk, P_lk = self.WS_eff_map(x_j, y_j, h, wt_x, wt_y, wt_type, wt_height, wd, ws)
        return X_j, Y_j, (WS_eff_jlk * P_lk[na, :, :] / P_lk.sum()).sum((1, 2)).reshape(X_j.shape)

    def plot_wake_map(self, x_j=None, y_j=None, h=None, wt_x=[], wt_y=[], wt_type=None, wt_height=None, wd=None, ws=None, ax=None, levels=100):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        X, Y, Z = self.wake_map(x_j, y_j, h, wt_x, wt_y, wt_type, wt_height, wd, ws)
        c = ax.contourf(X, Y, Z, levels, cmap='Blues_r')
        plt.colorbar(c, label='wind speed [m/s]')

    def aep_map(self, x_j, y_j, type_j, x_i, y_i, type_i=None, h_i=None, wd=None, ws=None):
        h = self.windTurbines.hub_height(type_j)
        X_j, Y_j, WS_eff_jlk, P_lk = self.WS_eff_map(x_j, y_j, h, x_i, y_i, type_i, h_i, wd, ws)
        P_lk /= P_lk.sum()  # AEP if wind only comes from specified wd and ws

        # power_jlk = self.windTurbines.power_func(WS_eff_jlk, type_j)
        # aep_jlk = power_jlk * P_lk[na, :, :] * 24 * 365 * 1e-9
        # return X_j, Y_j, aep_jlk.sum((1, 2)).reshape(X_j.shape)

        # same as above but requires less memory
        return X_j, Y_j, ((self.windTurbines.power(WS_eff_jlk, type_j) * P_lk[na, :, :]).sum((1, 2)) * 24 * 365 * 1e-9).reshape(X_j.shape)
