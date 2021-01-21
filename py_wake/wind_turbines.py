import numpy as np
import xml.etree.ElementTree as ET
from scipy.interpolate.fitpack2 import UnivariateSpline
from autograd.core import defvjp, primitive
from matplotlib.patches import Ellipse
from inspect import signature


class WindTurbines():
    """Set of multiple type wind turbines"""

    def __init__(self, names, diameters, hub_heights, ct_funcs, power_funcs, power_unit):
        """Initialize WindTurbines

        Parameters
        ----------
        names : array_like
            Wind turbine names
        diameters : array_like
            Diameter of wind turbines
        hub_heights : array_like
            Hub height of wind turbines
        ct_funcs : list of functions
            Wind turbine ct functions; func(ws) -> ct
        power_funcs : list of functions
            Wind turbine power functions; func(ws) -> power
        power_unit : {'W', 'kW', 'MW', 'GW'}
            Unit of power_func output (case insensitive)
        """
        self._names = np.array(names)
        self._diameters = np.array(diameters)
        self._hub_heights = np.array(hub_heights)

        def add_yaw_model(func_lst, yaw_model):
            return [(f, yaw_model(f))[len(signature(f).parameters) == 1] for f in func_lst]
        self.ct_funcs = add_yaw_model(ct_funcs, CTYawModel)
        power_funcs = add_yaw_model(power_funcs, YawModel)

        assert len(names) == len(diameters) == len(hub_heights) == len(ct_funcs) == len(power_funcs)
        power_scale = {'w': 1, 'kw': 1e3, 'mw': 1e6, 'gw': 1e9}[power_unit.lower()]
        if power_scale != 1:
            self.power_funcs = list([PowerScaler(f, power_scale) for f in power_funcs])
        else:
            self.power_funcs = power_funcs

    def _info(self, var, types):
        return var[np.asarray(types, int)]

    def hub_height(self, types=0):
        """Hub height of the specified type(s) of wind turbines
        """
        return self._info(self._hub_heights, types)

    def diameter(self, types=0):
        """Rotor diameter of the specified type(s) of wind turbines
        """
        return self._info(self._diameters, types)

    def name(self, types=0):
        """Name of the specified type(s) of wind turbines
        """
        return self._info(self._names, types)

    def power(self, ws_i, type_i=0, yaw_i=0):
        """Power in watt

        Parameters
        ----------
        ws_i : array_like, shape (i,...)
            Wind speed
        type_i : int or array_like, shape (i,)
            wind turbine type


        Returns
        -------
        power : array_like
            Power production for the specified wind turbine type(s) and wind speed
        """
        return self._ct_power(np.atleast_1d(ws_i), type_i, yaw_i=yaw_i)[1]

    def ct(self, ws_i, type_i=0, yaw_i=0):
        """Thrust coefficient

        Parameters
        ----------
        ws_i : array_like, shape (i,...)
            Wind speed
        type_i : int or array_like, shape (i,)
            wind turbine type

        Returns
        -------
        ct : array_like
            Thrust coefficient for the specified wind turbine type(s) and wind speed
        """
        return self._ct_power(ws_i, type_i, yaw_i=yaw_i)[0]

    def types(self):
        return np.arange(len(self._names))

    def get_defaults(self, N, type_i=0, h_i=None, d_i=None):
        """
        Parameters
        ----------
        N : int
            number of turbines
        type_i : array_like or None, optional
            Turbine type. If None, all turbines is type 0
        h_i : array_like or None, optional
            hub heights. If None: default hub heights (set in WindTurbines)
        d_i : array_lie or None, optional
            Rotor diameter. If None: default diameter (set in WindTurbines)
        """
        type_i = np.zeros(N, dtype=int) + type_i
        if h_i is None:
            h_i = self.hub_height(type_i)
        elif isinstance(h_i, (int, float)):
            h_i = np.zeros(N) + h_i
        if d_i is None:
            d_i = self.diameter(type_i)
        elif isinstance(d_i, (int, float)):
            d_i = np.zeros(N) + d_i
        return np.asarray(type_i), np.asarray(h_i), np.asarray(d_i)

    def _ct_power(self, ws_i, type_i=0, yaw_i=0):
        ws_i = np.asarray(ws_i)
        t = np.unique(type_i)  # .astype(int)
        if len(t) > 1:
            if type_i.shape != ws_i.shape:
                type_i = (np.zeros(ws_i.shape[0]) + type_i)
            if np.asarray(yaw_i).shape != ws_i.shape:
                yaw_i = (np.zeros(ws_i.shape[0]) + yaw_i)
            type_i = type_i.astype(int)
            CT = np.array([self.ct_funcs[t](ws, yaw) for t, ws, yaw in zip(type_i, ws_i, yaw_i)])
            P = np.array([self.power_funcs[t](ws, yaw) for t, ws, yaw in zip(type_i, ws_i, yaw_i)])
            return CT, P
        else:
            return self.ct_funcs[int(t[0])](ws_i, yaw_i), self.power_funcs[int(t[0])](ws_i, yaw_i)

    def set_gradient_funcs(self, power_grad_funcs, ct_grad_funcs):
        def add_grad(f_lst, df_lst):
            for i, f in enumerate(f_lst):
                @primitive
                def wrap(wsp, yaw, f=f):
                    return f(wsp, yaw)

                defvjp(wrap, lambda ans, wsp, yaw, df_lst=df_lst, i=i:
                       lambda g, df_lst=df_lst, i=i: g * df_lst[i](wsp))
                f_lst[i] = wrap

        add_grad(self.power_funcs, power_grad_funcs)
        add_grad(self.ct_funcs, ct_grad_funcs)

    def spline_ct_power(self, err_tol_factor=1e-2):
        def get_spline(func, err_tol_factor=1e-2):
            """Generate a spline of a ws dependent curve (power/ct)

            Parameters
            ----------
            func : function
                curve function (power/ct)
            err_tol_factor : float, default is 0.01
                the number of data points used by the spline is increased until the relative
                sum of errors is less than err_tol_factor.
            """
            # make curve tabular
            ws = np.arange(0, 100, .001)
            curve = func(ws, yaw=0)

            # smoothen curve to avoid spline oscillations around steps (especially around cut out)
            n, e = 99, 3
            lp_filter = ((np.cos(np.linspace(-np.pi, np.pi, n)) + 1) / 2)**e
            lp_filter /= lp_filter.sum()
            curve = np.convolve(curve, lp_filter, 'same')

            # make spline
            return UnivariateSpline(ws, curve, s=(curve.max() * err_tol_factor)**2)

        self.power_splines = [get_spline(p, err_tol_factor) for p in self.power_funcs]
        self.ct_splines = [get_spline(ct, err_tol_factor) for ct in self.ct_funcs]
        self.org_power_funcs = self.power_funcs.copy()
        self.org_ct_funcs = self.ct_funcs.copy()

        def add_grad(spline_lst):
            funcgrad_lst = []
            for spline in spline_lst:

                @primitive
                def f(ws, yaw=0):
                    assert np.all(yaw == 0), "Gradients of yaw not implemented"
                    return spline(ws)

                def f_vjp(ans, ws, yaw=0):
                    def gr(g):
                        return g * spline.derivative()(ws)
                    return gr

                defvjp(f, f_vjp)

                funcgrad_lst.append(f)
            return funcgrad_lst

        # replace power anc ct funcs
        self.power_funcs = add_grad(self.power_splines)
        self.ct_funcs = add_grad(self.ct_splines)

    def plot_xy(self, x, y, types=None, wd=None, yaw=0, normalize_with=1, ax=None):
        """Plot wind farm layout including type name and diameter

        Parameters
        ----------
        x : array_like
            x position of wind turbines
        y : array_like
            y position of wind turbines
        types : int or array_like
            type of the wind turbines
        wd : int, float, array_like or None
            - if int, float or array_like: wd is assumed to be the wind direction(s) and a line\
            indicating the perpendicular rotor is plotted.
            - if None: An circle indicating the rotor diameter is plotted
        ax : pyplot or matplotlib axes object, default None

        """
        import matplotlib.pyplot as plt
        if types is None:
            types = np.zeros_like(x)
        if ax is None:
            ax = plt.gca()
        markers = np.array(list("213v^<>o48spP*hH+xXDd|_"))
        colors = ['gray', 'r', 'g', 'k'] * 5

        from matplotlib.patches import Circle
        assert len(x) == len(y)
        types = (np.zeros_like(x) + types).astype(int)  # ensure same length as x
        yaw = np.zeros_like(x) + yaw

        x, y, D = [np.asarray(v) / normalize_with for v in [x, y, self.diameter(types)]]

        for i, (x_, y_, d, t, yaw_) in enumerate(zip(x, y, D, types, yaw)):
            if wd is None or len(np.atleast_1d(wd)) > 1:
                circle = Circle((x_, y_), d / 2, ec=colors[t], fc="None")
                ax.add_artist(circle)
                ax.plot(x_, y_, 'None', )
            else:
                for wd_ in np.atleast_1d(wd):
                    c, s = np.cos(np.deg2rad(90 + wd_ - yaw_)), np.sin(np.deg2rad(90 + wd_ - yaw_))
                    ax.plot([x_ - s * d / 2, x_ + s * d / 2], [y_ - c * d / 2, y_ + c * d / 2], lw=1, color=colors[t])

        for t, m, c in zip(np.unique(types), markers, colors):
            # ax.plot(np.asarray(x)[types == t], np.asarray(y)[types == t], '%sk' % m, label=self._names[int(t)])
            ax.plot([], [], '2', color=colors[t], label=self._names[int(t)])

        for i, (x_, y_, d) in enumerate(zip(x, y, D)):
            ax.annotate(i, (x_ + d / 2, y_ + d / 2), fontsize=7)
        ax.legend(loc=1)
        ax.axis('equal')

    def plot_yz(self, y, z=None, h=None, types=None, wd=270, yaw=0, normalize_with=1, ax=None):
        """Plot wind farm layout in yz-plane including type name and diameter

        Parameters
        ----------
        y : array_like
            y position of wind turbines
        types : int or array_like
            type of the wind turbines
        wd : int, float, array_like or None
            - if int, float or array_like: wd is assumed to be the wind direction(s) and a line\
            indicating the perpendicular rotor is plotted.
            - if None: An circle indicating the rotor diameter is plotted
        ax : pyplot or matplotlib axes object, default None

        """
        import matplotlib.pyplot as plt
        if z is None:
            z = np.zeros_like(y)
        if types is None:
            types = np.zeros_like(y).astype(int)
        else:
            types = (np.zeros_like(y) + types).astype(int)  # ensure same length as x
        if h is None:
            h = np.zeros_like(y) + self.hub_height(types)
        else:
            h = np.zeros_like(y) + h

        if ax is None:
            ax = plt.gca()
        markers = np.array(list("213v^<>o48spP*hH+xXDd|_"))
        colors = ['gray', 'k', 'r', 'g', 'k'] * 5

        from matplotlib.patches import Circle

        yaw = np.zeros_like(y) + yaw
        y, z, h, D = [v / normalize_with for v in [y, z, h, self.diameter(types)]]
        for i, (y_, z_, h_, d, t, yaw_) in enumerate(
                zip(y, z, h, D, types, yaw)):
            circle = Ellipse((y_, h_ + z_), d * np.sin(np.deg2rad(wd - yaw_)), d, ec=colors[t], fc="None")
            ax.add_artist(circle)
            ax.plot([y_, y_], [z_, z_ + h_], 'k')
            ax.plot(y_, h_, 'None')

        for t, m, c in zip(np.unique(types), markers, colors):
            ax.plot([], [], '2', color=c, label=self._names[int(t)])

        for i, (y_, z_, h_, d) in enumerate(zip(y, z, h, D)):
            ax.annotate(i, (y_ + d / 2, z_ + h_ + d / 2), fontsize=7)
        ax.legend(loc=1)
        ax.axis('equal')

    def plot(self, x, y, types=None, wd=None, yaw=0, normalize_with=1, ax=None):
        return self.plot_xy(x, y, types, wd, yaw, normalize_with, ax)

    @staticmethod
    def from_WindTurbines(wt_lst):
        """Generate a WindTurbines object from a list of (Onetype)WindTurbines

        Parameters
        ----------
        wt_lst : array_like
            list of (OneType)WindTurbines
        """
        def get(att):
            lst = []
            for wt in wt_lst:
                lst.extend(getattr(wt, att))
            return lst
        return WindTurbines(*[get(n) for n in ['_names', '_diameters', '_hub_heights',
                                               'ct_funcs', 'power_funcs']],
                            power_unit='w')

    @staticmethod
    def from_WAsP_wtg(wtg_file, mode=0, power_unit='W'):
        """ Parse the one/multiple .wtg file(s) (xml) to initilize an
        WindTurbines object.

        Parameters
        ----------
        wtg_file : string or a list of string
            A string denoting the .wtg file, which is exported from WAsP.

        Returns
        -------
        an object of WindTurbines.

        Note: it is assumed that the power_unit inside multiple .wtg files is the same, i.e., power_unit.
        """
        if not isinstance(wtg_file, list):
            wtg_file_list = [wtg_file]
        else:
            wtg_file_list = wtg_file

        names = []
        diameters = []
        hub_heights = []
        ct_funcs = []
        power_funcs = []
        densities = []
        mode = np.zeros(len(wtg_file_list), dtype=int) + mode
        char_data_tables = []
        cut_ins = []
        cut_outs = []
        for wtg_file, m in zip(wtg_file_list, mode):
            tree = ET.parse(wtg_file)
            root = tree.getroot()
            # Reading data from wtg_file
            name = root.attrib['Description']
            diameter = np.float(root.attrib['RotorDiameter'])
            hub_height = np.float(root.find('SuggestedHeights').find('Height').text)

            perftab = list(root.iter('PerformanceTable'))[m]
            density = np.float(perftab.attrib['AirDensity'])
            ws_cutin = np.float(perftab.find('StartStopStrategy').attrib['LowSpeedCutIn'])
            ws_cutout = np.float(perftab.find('StartStopStrategy').attrib['HighSpeedCutOut'])
            cut_ins.append(ws_cutin)
            cut_outs.append(ws_cutout)

            i_point = 0
            for DataPoint in perftab.iter('DataPoint'):
                i_point = i_point + 1
                ws = np.float(DataPoint.attrib['WindSpeed'])
                Ct = np.float(DataPoint.attrib['ThrustCoEfficient'])
                power = np.float(DataPoint.attrib['PowerOutput'])
                if i_point == 1:
                    dt = np.array([[ws, Ct, power]])
                else:
                    dt = np.append(dt, np.array([[ws, Ct, power]]), axis=0)
            ct_idle = dt[-1, 1]
            eps = 1e-8

            ws = np.r_[0, ws_cutin - eps, dt[:, 0], ws_cutout + eps, 100]
            ct = np.r_[ct_idle, ct_idle, dt[:, 1], ct_idle, ct_idle]
            power = np.r_[0, 0, dt[:, 2], 0, 0]

            names.append(name)
            diameters.append(diameter)
            hub_heights.append(hub_height)

            ct_funcs.append(Interp(ws, ct))
            power_funcs.append(Interp(ws, power))
            char_data_tables.append(np.array([ws, ct, power]).T)

        wts = WindTurbines(names=names, diameters=diameters,
                           hub_heights=hub_heights, ct_funcs=ct_funcs,
                           power_funcs=power_funcs, power_unit=power_unit)
        wts.upct_tables = char_data_tables
        wts.cut_in = cut_ins
        wts.cut_out = cut_outs
        return wts


class OneTypeWindTurbines(WindTurbines):
    """Set of wind turbines (one type, i.e. all wind turbines have same name, diameter, power curve etc"""

    def __init__(self, name, diameter, hub_height, ct_func, power_func, power_unit):
        """Initialize OneTypeWindTurbine

        Parameters
        ----------
        name : str
            Wind turbine name
        diameter : int or float
            Diameter of wind turbine
        hub_height : int or float
            Hub height of wind turbine
        ct_func : function
            Wind turbine ct function; func(ws) -> ct
        power_func : function
            Wind turbine power function; func(ws) -> power
        power_unit : {'W', 'kW', 'MW', 'GW'}
            Unit of power_func output (case insensitive)
        """
        WindTurbines.__init__(self, [name], [diameter], [hub_height],
                              [ct_func],
                              [power_func],
                              power_unit)

    @staticmethod
    def from_tabular(name, diameter, hub_height, ws, power, ct, power_unit):
        def power_func(u):
            return np.interp(u, ws, power)

        def ct_func(u):
            return np.interp(u, ws, ct)
        return OneTypeWindTurbines(name=name, diameter=diameter, hub_height=hub_height,
                                   ct_func=ct_func,
                                   power_func=power_func,
                                   power_unit=power_unit)

    def set_gradient_funcs(self, power_grad_funcs, ct_grad_funcs):
        WindTurbines.set_gradient_funcs(self, [power_grad_funcs], [ct_grad_funcs])


class PowerScaler():
    def __init__(self, f, power_scale):
        self.f = f
        self.power_scale = power_scale

    def __call__(self, ws, yaw):
        return self.f(ws, yaw) * self.power_scale


class YawModel():
    def __init__(self, func):
        self.func = func

    def __call__(self, ws, yaw=0):
        return self.func(np.cos(yaw) * np.asarray(ws))


class CTYawModel(YawModel):
    def __call__(self, ws, yaw=0):
        # ct_n = ct_curve(cos(yaw)*ws)*cos^2(yaw)
        # mapping to downwind deficit, i.e. ct_x = ct_n*cos(yaw) = ct_curve(cos(yaw)*ws)*cos^3(yaw),
        # handled in deficit model
        co = np.cos(yaw)
        return self.func(co * np.asarray(ws)) * co**2


class Interp(object):
    """ Decorate numpy.interp in a class that can be used like
    scipy.interpolate.interp1d: initialized once and interpolate latter."""

    def __init__(self, xp, fp, left=None, right=None, period=None):
        self.xp = xp
        self.fp = fp
        self.left = left
        self.right = right
        self.period = period

    def __call__(self, x):
        return np.interp(x, self.xp, self.fp, left=self.left,
                         right=self.right,
                         period=self.period)


def cube_power(ws_cut_in=3, ws_cut_out=25, ws_rated=12, power_rated=5000):
    def power_func(ws):
        ws = np.asarray(ws)
        power = np.zeros_like(ws, dtype=np.float)
        m = (ws >= ws_cut_in) & (ws < ws_rated)
        power[m] = power_rated * ((ws[m] - ws_cut_in) / (ws_rated - ws_cut_in))**3
        power[(ws >= ws_rated) & (ws <= ws_cut_out)] = power_rated
        return power
    return power_func


def dummy_thrust(ws_cut_in=3, ws_cut_out=25, ws_rated=12, ct_rated=8 / 9):
    # temporary thrust curve fix
    def ct_func(ws):
        ws = np.asarray(ws)
        ct = np.zeros_like(ws, dtype=np.float)
        if ct_rated > 0:
            # ct = np.ones_like(ct)*ct_rated
            m = (ws >= ws_cut_in) & (ws < ws_rated)
            ct[m] = ct_rated
            idx = (ws >= ws_rated) & (ws <= ws_cut_out)
            # second order polynomial fit for above rated
            ct[idx] = np.polyval(np.polyfit([ws_rated, (ws_rated + ws_cut_out) / 2,
                                             ws_cut_out], [ct_rated, 0.4, 0.03], 2), ws[idx])
        return ct
    return ct_func


def main():
    if __name__ == '__main__':
        import os.path
        import matplotlib.pyplot as plt
        from py_wake.examples.data import wtg_path

        wts = WindTurbines(names=['tb1', 'tb2'],
                           diameters=[80, 120],
                           hub_heights=[70, 110],
                           ct_funcs=[lambda ws: ws * 0 + 8 / 9,
                                     dummy_thrust()],
                           power_funcs=[cube_power(ws_cut_in=3, ws_cut_out=25, ws_rated=12, power_rated=2000),
                                        cube_power(ws_cut_in=3, ws_cut_out=25, ws_rated=12, power_rated=3000)],
                           power_unit='kW')

        ws = np.arange(25)
        plt.figure()
        plt.plot(ws, wts.power(ws, 0), label=wts.name(0))
        plt.plot(ws, wts.power(ws, 1), label=wts.name(1))
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(ws, wts.ct(ws, 0), label=wts.name(0))
        plt.plot(ws, wts.ct(ws, 1), label=wts.name(1))
        plt.legend()
        plt.show()

        plt.figure()
        wts.plot([0, 100], [0, 100], [0, 1])
        plt.xlim([-50, 150])
        plt.ylim([-50, 150])
        plt.show()

        # Exmaple using two wtg files to initialize a wind turbine
#        vestas_v80_wtg = './examples/data/Vestas-V80.wtg'
#        NEG_2750_wtg = './examples/data/NEG-Micon-2750.wtg'

#        data_folder = Path('./examples/data/')
#        vestas_v80_wtg = data_folder / 'Vestas-V80.wtg'
#        NEG_2750_wtg = data_folder / 'NEG-Micon-2750.wtg'
        vestas_v80_wtg = os.path.join(wtg_path, 'Vestas-V80.wtg')
        NEG_2750_wtg = os.path.join(wtg_path, 'NEG-Micon-2750.wtg')
        wts_wtg = WindTurbines.from_WAsP_wtg([vestas_v80_wtg, NEG_2750_wtg])

        ws = np.arange(30)

        plt.figure()
        plt.plot(ws, wts_wtg.power(ws, 0), label=wts_wtg.name(0))
        plt.plot(ws, wts_wtg.power(ws, 1), label=wts_wtg.name(1))
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(ws, wts_wtg.ct(ws, 0), label=wts_wtg.name(0))
        plt.plot(ws, wts_wtg.ct(ws, 1), label=wts_wtg.name(1))
        plt.legend()
        plt.show()


main()
