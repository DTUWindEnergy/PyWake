import numpy as np
from scipy.interpolate._cubic import PchipInterpolator
from scipy.interpolate.interpolate import RegularGridInterpolator
import types
from abc import abstractmethod, ABC
from autograd.core import defvjp, primitive
from py_wake.utils.gradients import fd
from scipy.interpolate.fitpack2 import UnivariateSpline
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.builtins import SequenceBox
from py_wake.utils.generic_power_ct_curves import standard_power_ct_curve
from py_wake.utils.check_input import check_input


class PowerCtModel():
    """Base class for all PowerCtModel classes"""

    def __init__(self, required_inputs, optional_inputs):
        if not hasattr(self, '_required_inputs'):
            self._required_inputs = set({})
            self._optional_inputs = set({})
        self.add_inputs(required_inputs, optional_inputs)

    @property
    def required_inputs(self):
        return sorted(self._required_inputs)

    @property
    def optional_inputs(self):
        return sorted(self._optional_inputs)

    def add_inputs(self, required_inputs, optional_inputs):
        lst = [i for sub_lst in required_inputs for i in ([sub_lst], sub_lst)[isinstance(sub_lst, (list, set))]]
        self._required_inputs |= set(lst)
        lst = [i for sub_lst in optional_inputs for i in ([sub_lst], sub_lst)[isinstance(sub_lst, (list, set))]]
        self._optional_inputs |= set(lst)

    def fix_shape(self, arr, arr_to_match, allow_number=False):
        if allow_number and isinstance(arr, (int, float)):
            return arr
        arr = np.asarray(arr)
        shape = np.asarray(arr_to_match).shape
        return np.broadcast_to(arr.reshape(arr.shape + (1,) * (len(shape) - len(arr.shape))), shape)


class PowerCtModelContainer(PowerCtModel, ABC):
    """Base class for PowerCtModels that may have additional models"""

    def __init__(self, required_inputs, optional_inputs, additional_models=[]):
        PowerCtModel.__init__(self, required_inputs, optional_inputs)
        for m in additional_models:
            self.add_inputs(m.required_inputs, m.optional_inputs)
        self.model_lst = additional_models

    def __call__(self, ws, **kwargs):
        """This function recursively calls all additional (intermediate) models.
        The last additional model calls the final PowerCtFunction.power_ct method
        The resulting power/ct array is propagated back through the additional models"""
        def recursive_wrap(model_idx, ws, **kwargs):
            if model_idx < len(self.model_lst):
                f = self.model_lst[model_idx]
                # more functions in f_lst to call
                if f.required_inputs or any([o in kwargs for o in f.optional_inputs]):
                    return f(lambda ws, model_idx=model_idx + 1, **kwargs: recursive_wrap(model_idx, ws, **kwargs),
                             ws, **kwargs)
                else:
                    # optional inputs not present => skip f and continue with next f in f_lst
                    return recursive_wrap(model_idx + 1, ws, **kwargs)
            else:
                return self.power_ct(ws, **kwargs)
        return recursive_wrap(0, ws, **kwargs)


class AdditionalModel(PowerCtModel, ABC):
    """AdditionalModel is intermediate model that wraps the final PowerCtFunction.
    It means that it can modify both inputs to and outputs from the final PowerCtFunction"""

    @abstractmethod
    def __call__(self, function, ws, **kwargs):
        """Method that wraps the next function in the call hierarchy (finally the PowerCtFunction)

        Parameters
        ----------
        function : function type
            Next function in the call hierarchy.
        ws : array_like
            wind speed
        Returns
        -------
        power_ct : array_like
            The (modified) result from calling the function
            called with (modified) ws and inputs must be returned
        """


class SimpleYawModel(AdditionalModel):
    """Simple model that replace ws with cos(yaw)*ws and scales the CT output with cos(yaw)**2"""

    def __init__(self):
        AdditionalModel.__init__(self, required_inputs=[], optional_inputs=['yaw'])

    def __call__(self, f, ws, yaw=None, **kwargs):
        if yaw is not None:
            co = np.cos(self.fix_shape(yaw, ws, True))
            power_ct_arr = f(ws * co, **kwargs)  # calculate for reduced ws (ws projection on rotor)

            # multiply ct by cos(yaw)**2 to compensate for reduced thrust
            power_ct_arr = [power_ct_arr[0], power_ct_arr[1] * co**2]
            return power_ct_arr
        else:
            return f(ws, **kwargs)


class DensityScale(AdditionalModel):
    """Scales the power and ct with density"""

    def __init__(self, air_density_ref):
        AdditionalModel.__init__(self, required_inputs=[], optional_inputs=['Air_density'])
        self.air_density_ref = air_density_ref

    def __call__(self, f, ws, Air_density=None, **kwargs):
        power_ct_arr = np.asarray(f(ws, **kwargs))
        if Air_density is not None:
            power_ct_arr *= self.fix_shape(Air_density, ws, True) / self.air_density_ref
        return power_ct_arr


default_additional_models = [SimpleYawModel(), DensityScale(1.225)]


class PowerCtFunction(PowerCtModelContainer, ABC):
    """Contains the final power_ct function and power scaling"""

    def __init__(self, input_keys, power_ct_func, power_unit, optional_inputs=[],
                 additional_models=default_additional_models):
        """
        Parameters
        ----------
        input_keys : array_like
            list of ordered input names. First element must be 'ws'
        power_ct_func : function type
            function(ws, input) -> power_ct_array
            shape of power_ct_array must be ws.shape+(2,)
            input is a dict and inputs consumed by the function must be popped, x = input.pop('x')
        power_unit : {'W','kW','MW','GW'}
            unit of power output from power_ct_func. Used to scale to W
        optional_inputs : list, optional
            list of keys in input_keys which are optional. Optional inputs which is not present is set to None
        additional_models : list, optional
            list of additional models.
        """
        assert input_keys[0] == 'ws'
        required_inputs = [k for k in input_keys[1:] if k not in optional_inputs]
        PowerCtModelContainer.__init__(self, required_inputs, optional_inputs, additional_models)
        self.input_keys = input_keys
        self.power_unit = power_unit
        self.power_scale = {'w': 1, 'kw': 1e3, 'mw': 1e6, 'gw': 1e9}[power_unit.lower()]
        self.power_ct = power_ct_func

    def enable_autograd(self):
        self.set_gradient_funcs(self.get_power_ct_grad_func())

    def __call__(self, ws, **kwargs):
        unused_inputs = set(kwargs) - self._required_inputs - self._optional_inputs
        if unused_inputs:
            raise TypeError("got unexpected keyword argument(s): '%s'" % ("', '".join(unused_inputs)))

        # Start call hierachy, i.e. recursively run all additional models and finally the self.power_ct
        power_ct_arr = PowerCtModelContainer.__call__(self, ws, **kwargs)
        if self.power_scale != 1:
            if isinstance(power_ct_arr, ArrayBox):
                return power_ct_arr * np.reshape([self.power_scale, 1], (2,) + (1,) * len(np.shape(ws)))
            return [power_ct_arr[0] * self.power_scale, power_ct_arr[1]]
        else:
            return power_ct_arr

    def set_gradient_funcs(self, power_ct_grad_func):

        def get_grad(ans, ws, **kwargs):
            def grad(g):
                dp, dct = power_ct_grad_func(ws, **kwargs)
                return np.asarray([g[0] * dp, g[1] * dct])
            return grad
        primitive_power_ct = primitive(self.power_ct)
        defvjp(primitive_power_ct, get_grad)
        self.power_ct = primitive_power_ct


class PowerCtTabular(PowerCtFunction):
    def __init__(self, ws, power, power_unit, ct, ws_cutin=None,
                 ws_cutout=None, power_idle=0, ct_idle=0, method='linear', additional_models=default_additional_models):
        """Tabular power and ct curve. Optionally insert extra points around cutin and cutout

        Parameters
        ----------
        ws : array_like
            wind speed values
        power : array_like
            power values
        power_unit : {'W','kW','MW','GW'}
            unit of power values
        ct : array_like
            ct values
        ws_cutin : number or None, optional
            if number, then the range [0,ws_cutin[ will be set to power_idle and ct_idle
        ws_cutout : number or None, optional
            if number, then the range ]ws_cutout,100] will be set to power_idle and ct_idle
        power_idle : number, optional
            see ws_cutin and ws_cutout
        ct_idle : number, optional
            see ws_cutin and ws_cutout
        method : {'linear', 'phip','spline}
            Interpolation method:\n
            - linear: fast, discontinous gradients\n
            - pchip: smooth\n
            - spline: smooth, closer to linear, small overshoots in transition to/from constant plateaus)
        additional_models : list, optional
            list of additional models.
        """

        eps = 1e-8
        if ws_cutin is not None:
            ws = np.r_[0, ws_cutin - eps, ws]
            power = np.r_[power_idle, power_idle, power]
            ct = np.r_[ct_idle, ct_idle, ct]
        if ws_cutout is not None:
            ws = np.r_[ws, ws_cutout + eps, 100]
            power = np.r_[power, power_idle, power_idle]
            ct = np.r_[ct, ct_idle, ct_idle]
        idx = np.argsort(ws)
        ws, power, ct = np.asarray(ws)[idx], np.asarray(power)[idx], np.asarray(ct)[idx]

        self.ws_tab = ws
        self.power_ct_tab = np.array([power, ct])

        assert method in ['linear', 'pchip', 'spline']
        if method == 'linear':
            interp = self.np_interp
        elif method == 'pchip':
            self._pchip_interpolator = PchipInterpolator(ws, self.power_ct_tab.T)
            interp = self.pchip_interp
        else:
            self.make_splines()
            interp = self.spline_interp
        self.interp = interp
        self.method = method
        PowerCtFunction.__init__(self, ['ws'], self.handle_cs, power_unit, [], additional_models)

    def handle_cs(self, ws):
        f = self.interp
        if np.asarray(ws).dtype == np.complex128:
            return np.asarray(f(ws.real)) + ws.imag * self.get_power_ct_grad_func()(ws.real) * 1j
        return np.asarray(f(ws))

    def get_power_ct_grad_func(self):
        if self.method == 'linear':
            power_ct_grad_func = fd(self.np_interp)  # fd is fine for linear interpolation
        elif self.method == 'pchip':
            def power_ct_grad_func(ws, f=self._pchip_interpolator.derivative()):
                return np.moveaxis(f(ws), -1, 0)
        else:
            def power_ct_grad_func(ws):
                return [f(ws) for f in self.power_ct_spline_derivative]
        return power_ct_grad_func

    def pchip_interp(self, ws):
        return np.moveaxis(self._pchip_interpolator(ws), -1, 0)

    def np_interp(self, ws):
        return np.interp(ws, self.ws_tab, self.power_ct_tab[0]), np.interp(ws, self.ws_tab, self.power_ct_tab[1])

    def spline_interp(self, ws):
        return self.power_spline(ws), self.ct_spline(ws)

    def make_splines(self, err_tol_factor=1e-2):
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
        power, ct = self.np_interp(ws)

        # smoothen curve to avoid spline oscillations around steps (especially around cut out)
        n, e = 99, 3
        lp_filter = ((np.cos(np.linspace(-np.pi, np.pi, n)) + 1) / 2)**e
        lp_filter /= lp_filter.sum()
        power = np.convolve(power, lp_filter, 'same')
        ct = np.convolve(ct, lp_filter, 'same')

        # make spline
        self.power_spline, self.ct_spline = [UnivariateSpline(ws, curve, s=(curve.max() * err_tol_factor)**2)
                                             for curve in [power, ct]]
        self.power_ct_spline_derivative = self.power_spline.derivative(), self.ct_spline.derivative()


class PowerCtFunctionList(PowerCtFunction):
    """Wraps a list of PowerCtFunction objects by adding a new discrete input argument,
    representing the index of the PowerCtFunction objects in the list"""

    def __init__(self, key, powerCtFunction_lst, default_value=None, additional_models=[]):
        """
        Parameters
        ----------
        key : string
            Name of new discrete input argument
        powerCtFunction_list : list
            List of PowerCtFunction objects
        default_value : int or None, optional
            If int, index of the default PowerCtFunction in the powerCtFunction_list
        additional_models : list, optional
            list of additional models.
        """
        if default_value is None:
            required_inputs, optional_inputs = [key], []
        else:
            required_inputs, optional_inputs = [], [key]
        # collect required and optional inputs from all powerCtFunctions
        required_inputs.extend([pcct.required_inputs for pcct in powerCtFunction_lst])
        optional_inputs.extend([pcct.optional_inputs for pcct in powerCtFunction_lst])
        PowerCtFunction.__init__(self, ['ws'] + required_inputs + optional_inputs, self._power_ct, power_unit='w',
                                 optional_inputs=optional_inputs, additional_models=additional_models)

        self.powerCtFunction_lst = powerCtFunction_lst
        self.default_value = default_value
        self.key = key

    def enable_autograd(self):
        for f in self.powerCtFunction_lst:
            f.enable_autograd()

    def _subset(self, arr, mask):
        if arr is None or isinstance(arr, types.FunctionType):
            return arr
        return np.broadcast_to(arr.reshape(arr.shape + (1,) * (len(mask.shape) - len(arr.shape))), mask.shape)[mask]

    def _power_ct(self, ws, **kwargs):
        try:
            idx = kwargs.pop(self.key)
        except KeyError:
            if self.default_value is None:
                raise KeyError(f"Argument, {self.key}, required to calculate power and ct not found")
            idx = self.default_value

        idx = np.asarray(idx, dtype=int)

        if idx.shape == (1,):
            idx = idx[0]
        if idx.shape == ():
            res = self.powerCtFunction_lst[idx](ws, **kwargs)
        else:
            res = np.empty((2,) + np.asarray(ws).shape)
            unique_idx = np.unique(idx)
            idx = np.zeros(ws.shape, dtype=int) + idx.reshape(idx.shape + (1,) * (len(ws.shape) - len(idx.shape)))
            for i in unique_idx:
                m = (idx == i)
                res[:, m] = self.powerCtFunction_lst[i](ws[m], **{k: self._subset(v, m) for k, v in kwargs.items()})
        for i in self.powerCtFunction_lst[0].required_inputs:
            kwargs.pop(i)
        for i in self.powerCtFunction_lst[0].optional_inputs:
            if i in kwargs:
                kwargs.pop(i)
        return res


class PowerCtNDTabular(PowerCtFunction):
    """Multi dimensional power/ct tabular"""

    def __init__(self, input_keys, value_lst, power_arr, power_unit,
                 ct_arr, default_value_dict={}, additional_models=default_additional_models):
        """
        Parameters
        ----------
        input_keys : array_like
            list of ordered input names. First element must be 'ws'
        value_lst : list of array_like
            list of values corresponding to each key in input_keys
        power_arr : array_like
            power array. Shape must be (len(value_lst[0]), .., len(value_lst[-1]))
        power_unit : {'W','kW','MW','GW'}
            unit of power values
        ct_arr : array_like
            ct array. Shape must be (len(value_lst[0]), .., len(value_lst[-1]))
        default_value_dict : dict
            dictionary with default values, e.g. {'Air_density':1.225}
        additional_models : list, optional
            list of additional models.
        """
        self.default_value_dict = default_value_dict
        self.interp = RegularGridInterpolator(value_lst, np.moveaxis([power_arr, ct_arr], 0, - 1))
        PowerCtFunction.__init__(self, input_keys, self._power_ct, power_unit,
                                 default_value_dict.keys(), additional_models)

    def _power_ct(self, ws, **kwargs):
        kwargs = {**self.default_value_dict, 'ws': ws, **kwargs}

        args = np.moveaxis([self.fix_shape(kwargs[k], ws)
                            for k in self.input_keys], 0, -1)
        try:
            return np.moveaxis(self.interp(args), -1, 0)
        except ValueError:
            check_input(self.interp.grid, args.T, self.input_keys)


class PowerCtXr(PowerCtNDTabular):
    """Multi dimensional power/ct tabular taking xarray dataset as input"""

    def __init__(self, ds, power_unit, method='linear', additional_models=default_additional_models):
        """
        Parameters
        ----------
        ds : xarray dataset
            Must contain data variables power and ct as well as the coordinate ws
        power_unit : {'W','kW','MW','GW'}
            unit of power values
        additional_models : list, optional
            list of additional models.
        """
        assert method == 'linear'
        assert 'power' in ds
        assert 'ct' in ds
        assert 'ws' in ds.dims
        power_arr, ct_arr = ds.to_array()

        if list(power_arr.dims).index('ws') > 0:
            power_arr, ct_arr = ds.transpose(*(['ws'] + [k for k in power_arr.dims if k != 'ws'])).to_array()

        PowerCtNDTabular.__init__(self, power_arr.dims, [power_arr[k] for k in power_arr.dims], power_arr, power_unit,
                                  ct_arr, additional_models=additional_models)


class CubePowerSimpleCt(PowerCtFunction):
    """Simple analytical power function and constant ct (until ws_rated
    whereafter it follows a second order polynomal to ws_cutout,ct_idle)"""

    def __init__(self, ws_cutin=3, ws_cutout=25, ws_rated=12,
                 power_rated=5000, power_unit='kw', ct=8 / 9, ct_idle=0.03,
                 additional_models=default_additional_models):
        """Parameters
        ----------
        ws_cutin : number
            cut-in wind speed
        ws_cutout : number
            cut-out wind speed
        ws_rated : number
            wind speed where rated power is reached
        power_rated : number
            rated power
        power_unit : {'W','kW','MW','GW'}
            unit of power_rated
        ct : number
            ct value applied in range [ws_cutin,ws_rated]
        ct_idle : number
            ct value applied for ws<ws_cutin and ws>ws_cutout
        additional_models : list, optional
            list of additional models.
        """
        PowerCtFunction.__init__(self, ['ws'], self._power_ct, power_unit, [], additional_models)
        self.ws_cutin = ws_cutin
        self.ws_rated = ws_rated
        self.ws_cutout = ws_cutout
        self.ct_idle = ct_idle
        self.ct = ct
        self.power_rated = power_rated

        if ct_idle is not None:
            # second order polynomial from (ws_rated,ct) to (ws_cutout,ct_idle) with slope(ws_cutout)=0
            a = (ct - ct_idle) / (ws_rated**2 - ws_cutout**2 - 2 * ws_cutout * ws_rated + 2 * ws_cutout**2)
            b = - 2 * a * ws_cutout
            c = ct - a * ws_rated**2 - b * ws_rated
            self.ct_rated2cutout = np.poly1d([a, b, c])
            self.dct_rated2cutout = np.poly1d([2 * a, b])
            self.abc = a, b, c

    def _power_ct(self, ws):
        ws = np.asarray(ws)

        power = np.where((ws > self.ws_cutin) & (ws <= self.ws_cutout),
                         np.minimum(self.power_rated * ((ws - self.ws_cutin) / (self.ws_rated - self.ws_cutin))**3,
                                    self.power_rated),
                         0)
        ws0 = ws * 0

        ct = ws0 + self.ct

        if self.ct_idle is not None:
            ct = np.where((ws < self.ws_cutin) | (ws > self.ws_cutout),
                          (ws0 + self.ct_idle),
                          ct)
            a, b, c = self.abc
            ct = np.where((ws > self.ws_rated) & (ws < self.ws_cutout),
                          a * ws**2 + b * ws + c,
                          ct)

        return np.asarray([power, ct])

    def get_power_ct_grad_func(self):
        return self._power_ct_grad

    def _power_ct_grad(self, ws):
        dp = np.where((ws > self.ws_cutin) & (ws <= self.ws_rated),
                      3 * self.power_rated * (ws - self.ws_cutin)**2 / (self.ws_rated - self.ws_cutin)**3,
                      0)
        dct = ws * 0
        if self.ct_idle is not None:
            dct = np.where((ws > self.ws_rated),
                           self.dct_rated2cutout(ws),
                           0)  # constant ct
        return np.asarray([dp, dct])
