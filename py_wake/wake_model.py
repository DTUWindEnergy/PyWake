from abc import abstractmethod, ABC
from numpy import newaxis as na
import numpy as np
from py_wake.site._site import Site
from py_wake.wind_turbines import WindTurbines
from py_wake.tests import npt


class WakeModel(ABC):
    """
    Base class for wake models

    Make a subclass and implement calc_deficit and calc_effective_WS.

    >>> class MyWakeModel(WakeModel):
    >>>    args4deficit = ['WS_lk', 'dw_jl'] # specify arguments required by calc_deficit
    >>>
    >>>    def calc_deficit(self, WS_lk, dw_jl):
    >>>        deficit_jlk = ...
    >>>        return deficit_jlk
    >>>
    >>>    def calc_effective_WS(self, WS_lk, deficit_jlk):
    >>>        deficit_sum_lk = ...
    >>>        return WS_lk - deficit_sum_lk

    Implementations of linear and squared sum method for calc_effective_WS are
    available through inherritance:

    >>> class MySquaredSumWakeModel(SquaredSum, WakeModel):
    >>>     ...

    >>> class MySquaredSumWakeModel(LinearSum, WakeModel):
    >>>     ...


    Suffixes:

    - d: turbines down wind order
    - i: turbines ordered by id
    - j: downstream points/turbines
    - k: wind speeds
    - l: wind directions
    - m: turbines and wind directions (il.flatten())
    - n: from_turbines, to_turbines and wind directions (iil.flatten())

    Arguments available for calc_deficit (specifiy in args4deficit):

    - WS_ilk: Local wind speed without wake effects
    - TI_ilk: local turbulence intensity without wake effects
    - WS_eff_ilk: Local wind speed with wake effects
    - TI_eff_ilk: local turbulence intensity with wake effects
    - D_src_il: Diameter of source turbine
    - D_dst_ijl: Diameter of destination turbine
    - dw_ijl: Downwind distance from turbine i to point/turbine j
    - hcw_ijl: Horizontal cross wind distance from turbine i to point/turbine j
    - cw_ijl: Cross wind(horizontal and vertical) distance from turbine i to point/turbine j
    - ct_ilk: Thrust coefficient

    """

    args4deficit = ['WS_lk']

    def __init__(self, site, windTurbines, wec=1):
        """Initialize WakeModel

        Parameters
        ----------
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        wec : float, optional
            Wake expansion continuation (default 1). For details, see Thomas and Ning
        (2018), "A Method for Reducing Multi-Modality in the Wind Farm Layout Optimization
        Problem".

        """
        assert isinstance(site, Site)
        assert isinstance(windTurbines, WindTurbines)
        self.site = site
        self.windTurbines = windTurbines
        self.wec = wec  # wake expansion continuation see
        # Thomas, J. J. and Ning, A., “A Method for Reducing Multi-Modality in the Wind Farm Layout Optimization Problem,”
        # Journal of Physics: Conference Series, Vol. 1037, The Science of Making
        # Torque from Wind, Milano, Italy, jun 2018, p. 10.
        self.deficit_initalized = False

    def init_deficit(self, **kwargs):
        self._calc_layout_terms(**kwargs)
        self.deficit_initalized = True

    def _calc_layout_terms(self, **_):
        pass

    def calc_wake(self, x_i, y_i, h_i=None, type_i=None, wd=None, ws=None):
        """Calculate wake effects

        Calculate effective wind speed, turbulence intensity (not
        implemented yet), power and thrust coefficient, and local
        site parameters

        Parameters
        ----------
        x_i : array_like
            X position of wind turbines
        y_i : array_like
            Y position of wind turbines
        h_i : array_like or None, optional
            Hub height of wind turbines\n
            If None, default, the standard hub height is used
        type_i : array_like or None, optional
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
        WS_eff_ilk : array_like
            Effective wind speeds [m/s]
        TI_eff_ilk : array_like
            Turbulence intensities. Should be effective, but not implemented yet
        power_ilk : array_like
            Power productions [w]
        ct_ilk : array_like
            Thrust coefficients
        WD_ilk : array_like
            Wind direction(s)
        WS_ilk : array_like
            Wind speed(s)
        TI_ilk : array_like
            Ambient turbulence intensitie(s)
        P_ilk : array_like
            Probability
        """

        type_i, h_i, D_i = self.windTurbines.get_defaults(len(x_i), type_i, h_i)
        wd, ws = self.site.get_defaults(wd, ws)

        # Find local wind speed, wind direction, turbulence intensity and probability
        WD_ilk, WS_ilk, TI_ilk, P_ilk = self.site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i, wd=wd, ws=ws)

        # Calculate down-wind and cross-wind distances
        dw_iil, hcw_iil, dh_iil, dw_order_indices_dl = self.site.wt2wt_distances(x_i, y_i, h_i, WD_ilk.mean(2))
        self._validate_input(dw_iil, hcw_iil)

        I, L = dw_iil.shape[1:]
        K = WS_ilk.shape[2]

        deficit_nk = np.zeros((I * I * L, K))

        from py_wake.turbulence_model import TurbulenceModel
        calc_ti = isinstance(self, TurbulenceModel)

        if calc_ti:
            add_turb_nk = np.zeros((I * I * L, K))

        indices = np.arange(I * I * L).reshape((I, I, L))
        WS_mk = WS_ilk.astype(np.float).reshape((I * L, K))
        WS_eff_mk = WS_mk.copy()
        TI_mk = TI_ilk.astype(np.float).reshape((I * L, K))
        TI_eff_mk = TI_mk.copy()
        dw_n = dw_iil.flatten()
        hcw_n = hcw_iil.flatten()
        if self.wec != 1:
            hcw_n = hcw_n / self.wec
        if 'cw_ijl' in self.args4deficit:
            cw_n = np.hypot(hcw_iil, dh_iil).flatten()
        dh_n = dh_iil.flatten()
        power_ilk = np.zeros((I, L, K))
        ct_ilk = np.zeros((I, L, K))
        i_wd_l = np.arange(L)

        # Iterate over turbines in down wind order
        for j in range(I):
            i_wt_l = dw_order_indices_dl[:, j]
            m = i_wt_l * L + i_wd_l  # current wt (j'th most upstream wts for all wdirs)

            # generate indexes of up wind(n_uw) and down wind(n_dw) turbines
            n_uw = indices[:, i_wt_l, i_wd_l][dw_order_indices_dl[:, :j].T, np.arange(L)]
            n_dw = indices[i_wt_l, :, i_wd_l][np.arange(L), dw_order_indices_dl[:, j + 1:].T]

            # Calculate effectiv wind speed at current turbines(all wind directions and wind speeds) and
            # look up power and thrust coefficient
            if j == 0:  # Most upstream turbines (no wake)
                WS_eff_lk = WS_mk[m]
            else:  # 2..n most upstream turbines (wake)
                WS_eff_lk = self.calc_effective_WS(WS_mk[m], deficit_nk[n_uw])
                WS_eff_mk[m] = WS_eff_lk
                if calc_ti:
                    TI_eff_mk[m] = self.calc_effective_TI(TI_mk[m], add_turb_nk[n_uw])

            ct_lk, power_lk = self.windTurbines._ct_power(WS_eff_lk, type_i[i_wt_l])

            power_ilk[i_wt_l, i_wd_l] = power_lk
            ct_ilk[i_wt_l, i_wd_l, :] = ct_lk

            if j < I - 1:
                # Calculate required args4deficit parameters
                arg_funcs = {'WS_ilk': lambda: WS_mk[m][na],
                             'WS_eff_ilk': lambda: WS_eff_mk[m][na],
                             'TI_ilk': lambda: TI_mk[m][na],
                             'TI_eff_ilk': lambda: TI_eff_mk[m][na],
                             'D_src_il': lambda: D_i[i_wt_l][na],
                             'D_dst_ijl': lambda: D_i[dw_order_indices_dl[:, j + 1:]].T[na],
                             'dw_ijl': lambda: dw_n[n_dw][na],
                             'cw_ijl': lambda: cw_n[n_dw][na],
                             'hcw_ijl': lambda: hcw_n[n_dw][na],
                             'dh_ijl': lambda: dh_n[n_dw][na],
                             'h_il': lambda: h_i[i_wt_l][na],
                             'ct_ilk': lambda: ct_ilk.reshape((I * L, K))[m][na]}
                args = {k: arg_funcs[k]() for k in self.args4deficit}

                # Calcualte deficit
                deficit_nk[n_dw] = self.calc_deficit(**args)
                if calc_ti:
                    # Calculate required args4deficit parameters and calculate added turbulence
                    args = {k: arg_funcs[k]() for k in self.args4addturb}
                    add_turb_nk[n_dw] = self.calc_added_turbulence(**args)

        WS_eff_ilk = WS_eff_mk.reshape((I, L, K))
        TI_eff_ilk = TI_eff_mk.reshape((I, L, K))

        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, WD_ilk, WS_ilk, TI_ilk, P_ilk

    def _map(self, args4func, func,
             x_j, y_j, h, wt_x_i, wt_y_i, wt_type_i, wt_h_i, wd=None, ws=None):
        wt_type_i, wt_h_i, wt_d_i = self.windTurbines.get_defaults(len(wt_x_i), wt_type_i, wt_h_i)
        wd, ws = self.site.get_defaults(wd, ws)

        # setup X,Y grid
        def f(x, N=500, ext=.2):
            ext *= (max(x) - min(x))
            return np.linspace(min(x) - ext, max(x) + ext, N)

        if x_j is None:
            x_j = f(wt_x_i)
        if y_j is None:
            y_j = f(wt_y_i)
        if h is None:
            h = np.mean(wt_h_i)

        X_j, Y_j = np.meshgrid(x_j, y_j)
        x_j, y_j = X_j.flatten(), Y_j.flatten()

        # calculate local wind at map points x_j, y_j, h
        h_j = np.zeros_like(x_j) + h
        _, WS_jlk, TI_jlk, P_jlk = self.site.local_wind(x_i=x_j, y_i=y_j, h_i=h_j, wd=wd, ws=ws)

        if len(wt_x_i) == 0:
            # If not turbines just return local wind
            return X_j, Y_j, np.zeros((0, 1, 1, 1)), WS_jlk, TI_jlk, P_jlk

        # Calculate ct for all turbines
        WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, WD_ilk, WS_ilk, TI_ilk, P_ilk = \
            self.calc_wake(x_i=wt_x_i, y_i=wt_y_i, h_i=wt_h_i, type_i=wt_type_i, wd=wd, ws=ws)

        # calculate distances
        dw_ijl, hcw_ijl, dh_ijl, _ = self.site.distances(wt_x_i, wt_y_i, wt_h_i, x_j, y_j, h_j, WD_ilk.mean(2))
        if 'cw_ijl' in self.args4deficit:
            cw_ijl = np.hypot(hcw_ijl, dh_ijl)

        if self.wec != 1:
            hcw_ijl = hcw_ijl / self.wec
        I, J, L = dw_ijl.shape
        K = WS_ilk.shape[2]

        deficit_ijlk = []
        for i in range(I):
            deficit_jlk = np.zeros((J, L, K))

            for l in range(L):
                m = dw_ijl[i, :, l] > 0

                arg_funcs = {'WS_ilk': lambda: WS_ilk[i, l][na, na],
                             'WS_eff_ilk': lambda: WS_eff_ilk[i, l][na, na],
                             'TI_ilk': lambda: TI_ilk[i, l][na, na],
                             'TI_eff_ilk': lambda: TI_eff_ilk[i, l][na, na],
                             'D_src_il': lambda: wt_d_i[i][na, na],
                             'D_dst_ijl': lambda: None,
                             'dw_ijl': lambda: dw_ijl[i, :, l][m][na, :, na],
                             'cw_ijl': lambda: cw_ijl[i, :, l][m][na, :, na],
                             'hcw_ijl': lambda: hcw_ijl[i, :, l][m][na, :, na],
                             'dh_ijl': lambda: dh_ijl[i, :, l][m][na, :, na],
                             'h_il': lambda: wt_h_i[i][na, na],
                             'ct_ilk': lambda: ct_ilk[i, l][na, na]}

                args = {k: arg_funcs[k]() for k in args4func}
                deficit_jlk[:, l][m] = func(**args)[0, :, 0]

            deficit_ijlk.append(deficit_jlk)
        deficit_ijlk = np.array(deficit_ijlk)
        return X_j, Y_j, deficit_ijlk, WS_jlk, TI_jlk, P_jlk

    def ws_map(self, x_j, y_j, h, wt_x_i, wt_y_i, wt_type_i, wt_h_i, wd, ws):
        """Calculate a wake (effecitve wind speed) map

        Parameters
        ----------
        x_j : array_like or None, optional
            X position map points
        y_j : array_like
            Y position of map points
        h : int, float or None, optional
            Height of wake map\n
            If None, default, the mean hub height is used
        wt_x_i : array_like, optional
            X position of wind turbines
        wt_y_i : array_like, optional
            Y position of wind turbines
        wt_type_i : array_like or None, optional
            Type of the wind turbines
        wt_h_i : array_like or None, optional
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
        WS_eff_jlk : array_like
            Local effective wind speed [m/s] for all map points(j),
            wind direction(l) and wind speed(k)
        WS_jlk : array_like
            Local wind speed without wake effects
        P_ilk : array_like
            Probability

        """
        X_j, Y_j, deficit_ijlk, WS_jlk, Ti_jlk, P_ilk = \
            self._map(args4func=self.args4deficit, func=self.calc_deficit,
                      x_j=x_j, y_j=y_j, h=h,
                      wt_x_i=wt_x_i, wt_y_i=wt_y_i, wt_type_i=wt_type_i, wt_h_i=wt_h_i,
                      wd=wd, ws=ws)
        WS_eff_jlk = self.calc_effective_WS(WS_jlk, deficit_ijlk)
        return X_j, Y_j, WS_eff_jlk, WS_jlk, P_ilk

    @abstractmethod
    def calc_deficit(self):
        """Calculate wake deficit caused by the x'th most upstream wind turbines
        for all wind directions(l) and wind speeds(k) on a set of points(j)

        This method must be overridden by subclass

        Arguments required by this method must be added to the class list
        args4deficit

        See class documentation for examples and available arguments

        Returns
        -------
        deficit_jlk : array_like
        """

    @abstractmethod
    def calc_effective_WS(self, WS_lk, deficit_ilk):
        """Calculate effective wind speed

        This method must be overridden by subclass or by adding SquaredSum or
        LinearSum as base class, see examples in WakeModel documentation

        Parameters
        ----------
        WS_lk : array_like
            Local wind speed at turbine/site(j) for all wind
            directions(l) and wind speeds(k)
        deficit_ilk : array_like
            deficit caused by upstream turbines(i) on all downstream turbines/points (j) for all wind directions(l)
            and wind speeds(k)

        Returns
        -------
        WS_eff_lk : array_like
            Effective wind speed at the x'th most upstream turbines for all wind
            directions(l) and wind speeds(k)

        See Also
        --------
        WakeModel

        """

    def _validate_input(self, dw_iil, hcw_iil):
        I_ = dw_iil.shape[0]
        i1, i2, _ = np.where((np.abs(dw_iil) + np.abs(hcw_iil) + np.eye(I_)[:, :, na]) == 0)
        if len(i1):
            msg = "\n".join(["Turbines %d and %d are at the same position" %
                             (i1[i], i2[i]) for i in range(len(i1))])
            raise ValueError(msg)


class SquaredSum():
    def calc_effective_WS(self, WS_lk, deficit_ilk):
        return WS_lk - np.sqrt(np.sum(deficit_ilk**2, 0))


class LinearSum():
    def calc_effective_WS(self, WS_lk, deficit_ilk):
        return WS_lk - np.sum(deficit_ilk, 0)


class MaxSum():
    def calc_effective_WS(self, WS_lk, deficit_ilk):
        return WS_lk - np.max(deficit_ilk, 0)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37 import iea37_path
        from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm,\
            read_iea37_windrose
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.site._site import UniformSite
        from py_wake.aep_calculator import AEPCalculator

        class MyWakeModel(SquaredSum, WakeModel):
            args4deficit = ['WS_ilk', 'dw_ijl']

            def calc_deficit(self, WS_ilk, dw_ijl):
                # 10% deficit downstream
                return (WS_ilk * .1)[:, na] * (dw_ijl > 0)[..., na]

        _, _, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
        n_wt = 16
        x, y, _ = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)

        site = UniformSite(freq, ti=0.075)
        windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')

        wake_model = MyWakeModel(site, windTurbines)
        aep_calculator = AEPCalculator(wake_model)

        import matplotlib.pyplot as plt
        aep_calculator.plot_wake_map(wt_x=x, wt_y=y, wd=[0, 30], ws=[9], levels=np.linspace(5, 9, 100))
        windTurbines.plot(x, y)

        plt.show()


main()
