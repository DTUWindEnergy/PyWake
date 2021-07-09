import numpy as np
import xml.etree.ElementTree as ET
from matplotlib.patches import Ellipse
import warnings
import inspect
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular, SimpleYawModel, CubePowerSimpleCt
import xarray as xr
from numpy import newaxis as na


class WindTurbines():
    """Set of multiple type wind turbines"""

    def __new__(cls, *args, **kwargs):
        from py_wake.wind_turbines.wind_turbines_deprecated import DeprecatedWindTurbines
        if cls != WindTurbines:
            return super(WindTurbines, cls).__new__(cls)
        try:
            inspect.getcallargs(DeprecatedWindTurbines.__init__, None, *args, **kwargs)
            warnings.warn("""WindTurbines(names, diameters, hub_heights, ct_funcs, power_funcs, power_unit=None) is deprecated.
Use WindTurbines(names, diameters, hub_heights, power_ct_funcs) instead""", DeprecationWarning, stacklevel=2)
            return DeprecatedWindTurbines(*args, **kwargs)
        except TypeError:
            return super(WindTurbines, cls).__new__(cls)

    def __init__(self, names, diameters, hub_heights, powerCtFunctions, loadFunctions=None):
        """Initialize WindTurbines

        Parameters
        ----------
        names : array_like
            Wind turbine names
        diameters : array_like
            Diameter of wind turbines
        hub_heights : array_like
            Hub height of wind turbines
        powerCtFunctions : list of powerCtFunction objects
            Wind turbine ct functions; func(ws) -> ct
        """
        self._names = np.array(names)
        self._diameters = np.array(diameters)
        self._hub_heights = np.array(hub_heights)
        assert len(names) == len(diameters) == len(hub_heights) == len(powerCtFunctions)
        self.powerCtFunction = PowerCtFunctionList('type', powerCtFunctions)
#         if loadFunctions:
#             self.loadfunction =

    @property
    def function_inputs(self):
        ri, oi = self.powerCtFunction.required_inputs, self.powerCtFunction.optional_inputs
        if hasattr(self, 'loadFunction'):
            ri += self.loadFunction.required_inputs
            oi += self.loadFunction.optional_inputs
        return ri, oi

    def _info(self, var, type):
        return var[np.asarray(type, int)]

    def hub_height(self, type=0):
        """Hub height of the specified type(s) of wind turbines"""
        return self._info(self._hub_heights, type)

    def diameter(self, type=0):
        """Rotor diameter of the specified type(s) of wind turbines"""
        return self._info(self._diameters, type)

    def name(self, type=0):
        """Name of the specified type(s) of wind turbines"""
        return self._info(self._names, type)

    def power(self, ws, **kwargs):
        """Power in watt

        Parameters
        ----------
        ws : array_like
            Wind speed
        kwargs : keyword arguments
            required and optional inputs
        """
        return self.powerCtFunction(ws, run_only=0, **kwargs)

    def ct(self, ws, **kwargs):
        """Thrust coefficient

        Parameters
        ----------
        ws : array_like
            Wind speed
        kwargs : keyword arguments
            required and optional inputs
        """
        return self.powerCtFunction(ws, run_only=1, **kwargs)

    def power_ct(self, ws, **kwargs):
        return [self.power(ws, **kwargs), self.ct(ws, **kwargs)]

    def loads(self, ws, **kwargs):
        return self.loadFunction(ws, **kwargs)

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
        return np.asarray(h_i), np.asarray(d_i)

    def enable_autograd(self):
        self.powerCtFunction.enable_autograd()

    def plot_xy(self, x, y, types=None, wd=None, yaw=0, tilt=0, normalize_with=1, ax=None):
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
        tilt = np.zeros_like(x) + tilt

        x, y, D = [np.asarray(v) / normalize_with for v in [x, y, self.diameter(types)]]
        R = D / 2
        for i, (x_, y_, r, t, yaw_, tilt_) in enumerate(zip(x, y, R, types, yaw, tilt)):
            if wd is None or len(np.atleast_1d(wd)) > 1:
                circle = Circle((x_, y_), r, ec=colors[t], fc="None")
                ax.add_artist(circle)
                ax.plot(x_, y_, 'None', )
            else:
                for wd_ in np.atleast_1d(wd):
                    circle = Ellipse((x_, y_), 2 * r * np.sin(np.deg2rad(tilt_)), 2 * r,
                                     angle=90 - wd_ + yaw_, ec=colors[t], fc="None")
                    ax.add_artist(circle)

        for t, m, c in zip(np.unique(types), markers, colors):
            # ax.plot(np.asarray(x)[types == t], np.asarray(y)[types == t], '%sk' % m, label=self._names[int(t)])
            ax.plot([], [], '2', color=colors[t], label=self._names[int(t)])

        for i, (x_, y_, r) in enumerate(zip(x, y, R)):
            ax.annotate(i, (x_ + r, y_ + r), fontsize=7)
        ax.legend(loc=1)
        ax.axis('equal')

    def plot_yz(self, y, z=None, h=None, types=None, wd=270, yaw=0, tilt=0, normalize_with=1, ax=None):
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
        tilt = np.zeros_like(y) + tilt
        y, z, h, D = [v / normalize_with for v in [y, z, h, self.diameter(types)]]
        for i, (y_, z_, h_, d, t, yaw_, tilt_) in enumerate(
                zip(y, z, h, D, types, yaw, tilt)):
            circle = Ellipse((y_, h_ + z_), d * np.sin(np.deg2rad(wd - yaw_)),
                             d, angle=-tilt_, ec=colors[t], fc="None")
            ax.add_artist(circle)
            ax.plot([y_, y_], [z_, z_ + h_], 'k')
            ax.plot(y_, h_, 'None')

        for t, m, c in zip(np.unique(types), markers, colors):
            ax.plot([], [], '2', color=c, label=self._names[int(t)])

        for i, (y_, z_, h_, d) in enumerate(zip(y, z, h, D)):
            ax.annotate(i, (y_ + d / 2, z_ + h_ + d / 2), fontsize=7)
        ax.legend(loc=1)
        ax.axis('equal')

    def plot(self, x, y, type=None, wd=None, yaw=0, tilt=0, normalize_with=1, ax=None):
        return self.plot_xy(x, y, type, wd, yaw, tilt, normalize_with, ax)

    @staticmethod
    def from_WindTurbine_lst(wt_lst):
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
        return WindTurbines(*[get(n) for n in ['_names', '_diameters', '_hub_heights']] +
                            [[getattr(wt, 'powerCtFunction') for wt in wt_lst]])

    @staticmethod
    def from_WindTurbines(wt_lst):
        from py_wake.wind_turbines.wind_turbines_deprecated import DeprecatedWindTurbines
        assert not any([isinstance(wt, DeprecatedWindTurbines) for wt in wt_lst]
                       ), "from_WindTurbines no longer supports DeprecatedWindTurbines"
        warnings.simplefilter('default', DeprecationWarning)
        warnings.warn("""WindTurbines.from_WindTurbines is deprecated. Use WindTurbines.from_WindTurbine_lst instead""",
                      DeprecationWarning, stacklevel=2)

        return WindTurbines.from_WindTurbine_lst(wt_lst)

    @staticmethod
    def from_WAsP_wtg(wtg_file, default_mode=0, power_unit='W'):
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
        if isinstance(wtg_file, (list, tuple)):
            return WindTurbine.from_WindTurbine_lst([WindTurbines.from_WAsP_wtg(f) for f in wtg_file])

        cut_ins = []
        cut_outs = []

        tree = ET.parse(wtg_file)
        root = tree.getroot()
        # Reading data from wtg_file
        name = root.attrib['Description']
        diameter = float(root.attrib['RotorDiameter'])
        hub_height = float(root.find('SuggestedHeights').find('Height').text)

        performance_tables = list(root.iter('PerformanceTable'))

        def fmt(v):
            try:
                return int(v)
            except (ValueError, TypeError):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return v

        wt_data = [{k: fmt(perftab.attrib.get(k, None)) for k in performance_tables[0].attrib}
                   for perftab in performance_tables]

        for i, perftab in enumerate(performance_tables):
            wt_data[i].update({k: float(perftab.find('StartStopStrategy').attrib.get(k, None))
                               for k in perftab.find('StartStopStrategy').attrib})
            wt_data[i].update({k: np.array([dp.attrib.get(k, np.nan) for dp in perftab.iter('DataPoint')], dtype=float)
                               for k in list(perftab.iter('DataPoint'))[0].attrib})
            wt_data[i]['ct_idle'] = wt_data[i]['ThrustCoEfficient'][-1]

        power_ct_funcs = PowerCtFunctionList(
            'mode', [PowerCtTabular(wt['WindSpeed'], wt['PowerOutput'], power_unit, wt['ThrustCoEfficient'],
                                    ws_cutin=wt['LowSpeedCutIn'], ws_cutout=wt['HighSpeedCutOut'],
                                    ct_idle=wt['ct_idle'], additional_models=[]) for wt in wt_data],
            default_value=default_mode, additional_models=[SimpleYawModel()])

        char_data_tables = [np.array([pct.ws_tab, pct.power_ct_tab[0], pct.power_ct_tab[1]]).T
                            for pct in power_ct_funcs.windTurbineFunction_lst]

        wts = WindTurbine(name=name, diameter=diameter, hub_height=hub_height,
                          powerCtFunction=power_ct_funcs)
        wts.wt_data = wt_data
        wts.upct_tables = char_data_tables
        wts.cut_in = cut_ins
        wts.cut_out = cut_outs
        return wts


class WindTurbine(WindTurbines):
    """Set of wind turbines (one type, i.e. all wind turbines have same name, diameter, power curve etc"""

    def __init__(self, name, diameter, hub_height, powerCtFunction, **windTurbineFunctions):
        """Initialize OneTypeWindTurbine

        Parameters
        ----------
        name : str
            Wind turbine name
        diameter : int or float
            Diameter of wind turbine
        hub_height : int or float
            Hub height of wind turbine
        powerCtFunction : PowerCtFunction object
            Wind turbine powerCtFunction
        """
        self._names = np.array([name])
        self._diameters = np.array([diameter])
        self._hub_heights = np.array([hub_height])
        self.powerCtFunction = powerCtFunction
        for k, v in windTurbineFunctions.items():
            setattr(self, k, v)


def main():
    if __name__ == '__main__':
        import os.path
        import matplotlib.pyplot as plt
        from py_wake.examples.data import wtg_path

        wts = WindTurbines(names=['tb1', 'tb2'],
                           diameters=[80, 120],
                           hub_heights=[70, 110],
                           powerCtFunctions=[
                               CubePowerSimpleCt(ws_cutin=3, ws_cutout=25, ws_rated=12,
                                                 power_rated=2000, power_unit='kW',
                                                 ct=8 / 9, additional_models=[]),
                               CubePowerSimpleCt(ws_cutin=3, ws_cutout=25, ws_rated=12,
                                                 power_rated=3000, power_unit='kW',
                                                 ct=8 / 9, additional_models=[]),
        ])

        ws = np.arange(25)
        plt.figure()
        plt.plot(ws, wts.power(ws, type=0), label=wts.name(0))
        plt.plot(ws, wts.power(ws, type=1), label=wts.name(1))
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(ws, wts.ct(ws, type=0), label=wts.name(0))
        plt.plot(ws, wts.ct(ws, type=1), label=wts.name(1))
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
        plt.plot(ws, wts_wtg.power(ws, type=0), label=wts_wtg.name(0))
        plt.plot(ws, wts_wtg.power(ws, type=1), label=wts_wtg.name(1))
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(ws, wts_wtg.ct(ws, type=0), label=wts_wtg.name(0))
        plt.plot(ws, wts_wtg.ct(ws, type=1), label=wts_wtg.name(1))
        plt.legend()
        plt.show()


main()
