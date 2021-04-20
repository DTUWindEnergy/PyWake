import numpy as np
from scipy.interpolate.fitpack2 import UnivariateSpline
from autograd.core import defvjp, primitive
from inspect import signature
from py_wake.wind_turbines._wind_turbines import WindTurbines
from py_wake.wind_turbines.wind_turbine_functions import WindTurbineFunction


class DeprecatedWindTurbines(WindTurbines):
    """Set of multiple type wind turbines"""

    def __init__(self, names, diameters, hub_heights, ct_funcs, power_funcs, power_unit=None):
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
        assert len(names) == len(diameters) == len(hub_heights) == len(ct_funcs) == len(power_funcs)

        def add_yaw_model(func_lst, yaw_model):
            return [(f, yaw_model(f))[len(signature(f).parameters) == 1] for f in func_lst]

        ct_funcs = add_yaw_model(ct_funcs, CTYawModel)
        power_funcs = add_yaw_model(power_funcs, YawModel)

        self._ct_funcs = ct_funcs
        if power_unit is not None:

            power_scale = {'w': 1, 'kw': 1e3, 'mw': 1e6, 'gw': 1e9}[power_unit.lower()]
            if power_scale != 1:
                power_funcs = list([PowerScaler(f, power_scale) for f in power_funcs])

        self._power_funcs = power_funcs
        self.powerCtFunction = WindTurbineFunction(['ws', 'type', 'yaw'], [], [])  # dummy for forward compatibility

    def _ct_power(self, ws_i, type=0, **kwargs):
        ws_i = np.asarray(ws_i)
        t = np.unique(type)  # .astype(int)
        if len(t) > 1:
            if type.shape != ws_i.shape:
                type = (np.zeros(ws_i.shape[0]) + type)
            type = type.astype(int)
            CT = np.array([self._ct_funcs[t](ws) for t, ws in zip(type, ws_i)])
            P = np.array([self._power_funcs[t](ws) for t, ws in zip(type, ws_i)])
            return CT, P
        else:
            return (self._ct_funcs[int(t[0])](ws_i, **kwargs),
                    self._power_funcs[int(t[0])](ws_i, **kwargs))

    def power(self, *args, **kwargs):
        return self._ct_power(*args, **kwargs)[1]

    def ct(self, *args, **kwargs):
        return self._ct_power(*args, **kwargs)[0]

    def set_gradient_funcs(self, power_grad_funcs, ct_grad_funcs):
        def add_grad(f_lst, df_lst):
            for i, f in enumerate(f_lst):
                @primitive
                def wrap(wsp, yaw, f=f):
                    return f(wsp, yaw)

                defvjp(wrap, lambda ans, wsp, yaw, df_lst=df_lst, i=i:
                       lambda g, df_lst=df_lst, i=i: g * df_lst[i](wsp))
                f_lst[i] = wrap

        add_grad(self._power_funcs, power_grad_funcs)
        add_grad(self._ct_funcs, ct_grad_funcs)

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
                                               '_ct_funcs', '_power_funcs']],
                            power_unit='w')


class DeprecatedOneTypeWindTurbines(DeprecatedWindTurbines):

    def __init__(self, name, diameter, hub_height, ct_func, power_func, power_unit=None):
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
        DeprecatedWindTurbines.__init__(self, [name], [diameter], [hub_height],
                                        [ct_func],
                                        [power_func],
                                        power_unit)

    @staticmethod
    def from_tabular(name, diameter, hub_height, ws, power, ct, power_unit):
        def power_func(u):
            return np.interp(u, ws, power)

        def ct_func(u):
            return np.interp(u, ws, ct)
        return DeprecatedOneTypeWindTurbines(name=name, diameter=diameter, hub_height=hub_height,
                                             ct_func=ct_func,
                                             power_func=power_func,
                                             power_unit=power_unit)

    def set_gradient_funcs(self, power_grad_funcs, ct_grad_funcs):
        DeprecatedWindTurbines.set_gradient_funcs(self, [power_grad_funcs], [ct_grad_funcs])


class PowerScaler():
    def __init__(self, f, power_scale):
        self.f = f
        self.power_scale = power_scale

    def __call__(self, ws, **kwargs):
        return self.f(ws, **kwargs) * self.power_scale


class YawModel():
    def __init__(self, func):
        self.func = func

    def __call__(self, ws, yaw=0):
        if yaw is None:
            return self.func(ws)
        return self.func(np.cos(yaw) * np.asarray(ws))


class CTYawModel(YawModel):
    def __call__(self, ws, yaw=0):
        # ct_n = ct_curve(cos(yaw)*ws)*cos^2(yaw)
        # mapping to downwind deficit, i.e. ct_x = ct_n*cos(yaw) = ct_curve(cos(yaw)*ws)*cos^3(yaw),
        # handled in deficit model
        if yaw is None:
            return self.func(ws)
        co = np.cos(yaw)
        return self.func(co * np.asarray(ws)) * co**2
