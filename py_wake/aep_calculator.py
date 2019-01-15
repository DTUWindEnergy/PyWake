import numpy as np
from numpy import newaxis as na


class AEPCalculator():

    def __init__(self, site, windTurbines, wake_model):
        """Initialize AEPCalculator

        Parameters
        ----------
        site : py_wake.site.Site
        windTurbines : WindTurbines
        wake_model : WakeModel
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
        """Calculate AEP

        In addition effective wind speed, turbulence intensity, and the
        power, ct and probability is calculated

        Parameters
        ----------
        x_i : array_like
            X position of wind turbines
        y_i : array_like
            Y position of wind turbines
        h_i : array_like or None, optional
            Hub height of wind turbines\n
            If None, default, the standard hub height is used
        type_i array_like or None, optional
            Wind turbine types\n
            If None, default, the first type is used (type=0)
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws

        Returns
        -------
        AEP_GWh_ilk : array_like
            AEP in GWh
        """
        self._run_wake_model(x_i=x_i, y_i=y_i, h_i=h_i, type_i=type_i, wd=wd, ws=ws)
        AEP_GWh_ilk = self.power_ilk * self.P_lk[na, :, :] * 24 * 365 * 1e-9
        return AEP_GWh_ilk

    def calculate_AEP_no_wake_loss(self, x_i, y_i, h_i=None, type_i=None, wd=None, ws=None):
        """Calculate AEP without wake loss

        In addition local wind speed, turbulence intensity, and the
        power, ct and probability is calculated

        Parameters
        ----------
        x_i : array_like
            X position of wind turbines
        y_i : array_like
            Y position of wind turbines
        h_i : array_like or None, optional
            Hub height of wind turbines\n
            If None, default, the standard hub height is used
        type_i array_like or None, optional
            Wind turbine types\n
            If None, default, the first type is used (type=0)
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws

        Returns
        -------
        AEP_GWh_ilk : array_like
            AEP (without wake loss) in GWh
        """
        h_i, type_i, wd, ws = self._get_defaults(x_i, h_i, type_i, wd=wd, ws=ws)

        # Find local wind speed, wind direction, turbulence intensity and probability
        self.WD_ilk, self.WS_ilk, self.TI_ilk, self.P_lk = self.site.local_wind(x_i=x_i, y_i=y_i, wd=wd, ws=ws)

        type_ilk = np.zeros(self.WS_ilk.shape, dtype=np.int) + type_i[:, np.newaxis, np.newaxis]
        self.power_ilk = self.windTurbines.power(self.WS_ilk, type_ilk)
        AEP_GWh_ilk = self.power_ilk * self.P_lk[na, :, :] * 24 * 365 * 1e-9
        return AEP_GWh_ilk

    def _WS_eff_map(self, x_j, y_j, h, x_i, y_i, type_i, h_i, wd=None, ws=None):
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
        """Calculate wake(effective wind speed) map

        In addition local wind speed, turbulence intensity, and the
        power, ct and probability is calculated

        Parameters
        ----------
        x_j : array_like or None, optional
            X position map points
        y_j : array_like
            Y position of map points
        h : int, float or None, optional
            Height of wake map\n
            If None, default, the mean hub height is used
        wt_x : array_like, optional
            X position of wind turbines
        wt_y : array_like, optional
            Y position of wind turbines
        wt_type : array_like or None, optional
            Type of the wind turbines
        wt_height : array_like or None, optional
            Hub height of the wind turbines\n
            If None, default, the standard hub height is used
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws

        Returns
        -------
        X_j : array_like
            2d array of map x positions
        Y_j : array_like
            2d array of map y positions
        WS_eff_avg : array_like
            2d array of average effective local wind speed taking into account
            the probability of wind direction and speed

        See Also
        --------
        plot_wake_map
        """
        X_j, Y_j, WS_eff_jlk, P_lk = self._WS_eff_map(x_j, y_j, h, wt_x, wt_y, wt_type, wt_height, wd, ws)
        return X_j, Y_j, (WS_eff_jlk * P_lk[na, :, :] / P_lk.sum()).sum((1, 2)).reshape(X_j.shape)

    def plot_wake_map(self, x_j=None, y_j=None, h=None, wt_x=[], wt_y=[], wt_type=None, wt_height=None, wd=None, ws=None, ax=None, levels=100):
        """Plot wake(effective wind speed) map

        Parameters
        ----------
        x_j : array_like or None, optional
            X position map points
        y_j : array_like
            Y position of map points
        h : int, float or None, optional
            Height of wake map\n
            If None, default, the mean hub height is used
        wt_x : array_like, optional
            X position of wind turbines
        wt_y : array_like, optional
            Y position of wind turbines
        wt_type : array_like or None, optional
            Type of the wind turbines
        wt_height : array_like or None, optional
            Hub height of the wind turbines\n
            If None, default, the standard hub height is used
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws
        ax : pyplot or matplotlib axes object, default None
            Axes to plot on
        levels : int or array_like
            levels for pyplot.contourf
        """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        X, Y, Z = self.wake_map(x_j, y_j, h, wt_x, wt_y, wt_type, wt_height, wd, ws)
        c = ax.contourf(X, Y, Z, levels, cmap='Blues_r')
        plt.colorbar(c, label='wind speed [m/s]')

    def aep_map(self, x_j=None, y_j=None, type_j=None, wt_x=[], wt_y=[], wt_type=None, wt_height=None, wd=None, ws=None):
        """Calculate AEP map

        The map represents the of AEP produced by a new turbine at the specified positions

        Parameters
        ----------
        x_j : array_like or None, optional
            X position map points (potential turbine positions)
        y_j : array_like
            Y position of map points (potential turbine positions)
        type_j : int, float or None, optional
            Type of potential turbine positions\n
            If None, default, first turbine type(0) is used
        wt_x : array_like, optional
            X position of the current wind turbines
        wt_y : array_like, optional
            Y position of the current wind turbines
        wt_type : array_like or None, optional
            Type of the current wind turbines
        wt_height : array_like or None, optional
            Hub height of the current wind turbines\n
            If None, default, the standard hub height is used
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws

        Returns
        -------
        X_j : array_like
            2d array of map x positions
        Y_j : array_like
            2d array of map y positions
        WS_eff_avg : array_like
            2d array of average effective local wind speed taking into account
            the probability of wind direction and speed
        """
        h_j = self.windTurbines.hub_height(type_j)
        X_j, Y_j, WS_eff_jlk, P_lk = self._WS_eff_map(x_j, y_j, h_j, wt_x, wt_y, wt_type, wt_height, wd, ws)
        P_lk /= P_lk.sum()  # AEP if wind only comes from specified wd and ws

        # power_jlk = self.windTurbines.power_func(WS_eff_jlk, type_j)
        # aep_jlk = power_jlk * P_lk[na, :, :] * 24 * 365 * 1e-9
        # return X_j, Y_j, aep_jlk.sum((1, 2)).reshape(X_j.shape)

        # same as above but requires less memory
        return X_j, Y_j, ((self.windTurbines.power(WS_eff_jlk, type_j) * P_lk[na, :, :]).sum((1, 2)) * 24 * 365 * 1e-9).reshape(X_j.shape)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37 import iea37_path
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.wake_models import NOJ

        # setup site, turbines and wakemodel
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

        wake_model = NOJ(windTurbines)

        # calculate AEP
        aep_calculator = AEPCalculator(site, windTurbines, wake_model)
        aep = aep_calculator.calculate_AEP(x, y)[0].sum()

        print(aep_calculator.WS_eff_ilk.shape)

        # plot wake map
        import matplotlib.pyplot as plt
        aep_calculator.plot_wake_map(wt_x=x, wt_y=y, wd=[0], ws=[9])
        plt.title('AEP: %.2f GWh' % aep)
        windTurbines.plot(x, y)
        plt.show()


main()
