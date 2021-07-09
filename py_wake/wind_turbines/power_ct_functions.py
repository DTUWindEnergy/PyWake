import numpy as np
from scipy.interpolate._cubic import PchipInterpolator
from scipy.interpolate.interpolate import RegularGridInterpolator
from abc import abstractmethod, ABC
from autograd.core import defvjp, primitive
from py_wake.utils.gradients import fd
from scipy.interpolate.fitpack2 import UnivariateSpline
from autograd.numpy.numpy_boxes import ArrayBox
from py_wake.wind_turbines.wind_turbine_functions import WindTurbineFunction, FunctionSurrogates,\
    WindTurbineFunctionList
from py_wake.utils.check_input import check_input
from py_wake.utils.model_utils import check_model, fix_shape


"""
sequenceDiagram
    participant wfm as WindFarmModel
    participant pctf as PowerCtFunction
    participant wtf as WindTurbineFunction
    participant pmc as PowerCtModelContainer
    participant ds as DensityScale
    participant cpct as CubePowerSimpleCt
    wfm->>+pctf: call(ws,**kwargs)
    pctf->>+wtf: call(ws,**kwargs)
    wtf->>wtf: check no unused input
    wtf->>+pctf: evaluate(ws,**kwargs)
    pctf->>+pmc: call(ws,**kwargs)
    pmc->>+pmc: recursive_wrap(ws,**kwargs)
    pmc->>+ds: call(ws, Air_density=None,**kwargs)
    ds->>+pmc: recursive_wrap(ws,**kwargs)
    pmc->>+cpct: power_ct(ws,**kwargs)
    cpct->>-pmc: power/ct
    pmc->>-ds: power/ct
    ds->ds: scale power/ct
    ds->>-pmc: power/ct
    pmc->>-pmc: power/ct
    pmc->>-pctf: power/ct
    pctf->>-wtf: power/ct
    wtf->>-pctf: power/ct
    pctf->pctf: scale power
    pctf->>-wfm: power[w]/ct
"""


class PowerCtModelContainer(WindTurbineFunction, ABC):
    """Base class for PowerCtModels that may have additional models"""

    def __init__(self, input_keys, optional_inputs, additional_models=[]):
        WindTurbineFunction.__init__(self, input_keys, optional_inputs, output_keys=['power', 'ct'])
        for i, m in enumerate(additional_models):
            check_model(m, AdditionalModel, f"Additional model, element {i}")
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
                if f.required_inputs or any([o in kwargs for o in f.optional_inputs if o is not None]):
                    return f(lambda ws, model_idx=model_idx + 1, **kwargs: recursive_wrap(model_idx, ws, **kwargs),
                             ws, **kwargs)
                else:
                    # optional inputs not present => skip f and continue with next f in f_lst
                    return recursive_wrap(model_idx + 1, ws, **kwargs)
            else:
                return self.power_ct(ws, **kwargs)
        return recursive_wrap(0, ws, **kwargs)


class AdditionalModel(WindTurbineFunction, ABC):
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
        AdditionalModel.__init__(
            self, input_keys=[
                'ws', 'yaw', 'tilt'], optional_inputs=['yaw', 'tilt'],
            output_keys=['power', 'ct'])

    def __call__(self, f, ws, yaw=None, tilt=None, **kwargs):
        if yaw is not None or tilt is not None:
            co = 1
            if yaw is not None:
                co *= np.cos(np.deg2rad(fix_shape(yaw, ws, True)))
            if tilt is not None:
                co *= np.cos(np.deg2rad(fix_shape(tilt, ws, True)))
            power_ct_arr = f(ws * co, **kwargs)  # calculate for reduced ws (ws projection on rotor)
            if kwargs['run_only'] == 1:  # ct
                # multiply ct by cos(yaw)**2 to compensate for reduced thrust
                return power_ct_arr * co**2
            return power_ct_arr
        else:
            return f(ws, **kwargs)


class DensityScale(AdditionalModel):
    """Scales the power and ct with density"""

    def __init__(self, air_density_ref):
        AdditionalModel.__init__(self, input_keys=['ws', 'Air_density'], optional_inputs=['Air_density'],
                                 output_keys=['power', 'ct'])
        self.air_density_ref = air_density_ref

    def __call__(self, f, ws, Air_density=None, **kwargs):
        power_ct_arr = np.asarray(f(ws, **kwargs))
        if Air_density is not None:
            power_ct_arr *= fix_shape(Air_density, ws, True) / self.air_density_ref
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
        PowerCtModelContainer.__init__(self, input_keys, optional_inputs, additional_models)
        self.input_keys = input_keys
        self.power_unit = power_unit
        self.power_scale = {'w': 1, 'kw': 1e3, 'mw': 1e6, 'gw': 1e9}[power_unit.lower()]
        self.power_ct = power_ct_func

    def enable_autograd(self):
        self.set_gradient_funcs(self.get_power_ct_grad_func())

    def __call__(self, ws, run_only=slice(None), **kwargs):
        if run_only not in [0, 1]:
            return np.array([self.__call__(ws, i, **kwargs) for i in np.arange(2)[run_only]])

        # Start call hierachy, i.e. recursively run all additional models and finally the self.power_ct
        power_ct_arr = PowerCtModelContainer.__call__(self, ws, run_only=run_only, **kwargs)
        if run_only == 0:
            return power_ct_arr * self.power_scale
        else:
            return power_ct_arr

    def set_gradient_funcs(self, power_ct_grad_func):

        def get_grad(ans, ws, **kwargs):
            def grad(g):
                return g * power_ct_grad_func(ws, **kwargs)
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
        self.ws_cutin, self.ws_cutout, self.ct_idle = ws_cutin, ws_cutout, ct_idle
        idx = np.argsort(ws)
        ws, power, ct = np.asarray(ws)[idx], np.asarray(power)[idx], np.asarray(ct)[idx]

        self.ws_tab = ws
        self.power_ct_tab = np.array([power, ct])

        assert method in ['linear', 'pchip', 'spline']
        if method == 'linear':
            interp = self.np_interp
        elif method == 'pchip':
            self._pchip_interpolator = [PchipInterpolator(ws, self.power_ct_tab[0]),
                                        PchipInterpolator(ws, self.power_ct_tab[1])]
            self._pchip_derivative = [pi.derivative() for pi in self._pchip_interpolator]
            interp = self.pchip_interp
        else:
            self.make_splines()
            interp = self.spline_interp
        self.interp = interp
        self.method = method
        PowerCtFunction.__init__(self, ['ws'], self.handle_cs, power_unit, [], additional_models)

    def handle_cs(self, ws, run_only, **_):
        if np.asarray(ws).dtype == np.complex128:
            return np.asarray(self.interp(ws.real, run_only)) + \
                ws.imag * self.get_power_ct_grad_func()(ws.real, run_only) * 1j
        else:
            return np.asarray(self.interp(ws, run_only))

    def get_power_ct_grad_func(self):
        if self.method == 'linear':
            def power_ct_grad_func(ws, run_only):
                # fd is fine for linear interpolation
                return fd(lambda ws, run_only=run_only, self=self: self.np_interp(ws, run_only))(ws)
        elif self.method == 'pchip':
            def power_ct_grad_func(ws, run_only):
                return self._pchip_derivative[run_only](ws)
        else:
            def power_ct_grad_func(ws, run_only):
                return self.power_ct_spline_derivative[run_only](ws)
        return power_ct_grad_func

    def pchip_interp(self, ws, run_only):
        return self._pchip_interpolator[run_only](ws)

    def np_interp(self, ws, run_only):
        return np.interp(ws, self.ws_tab, self.power_ct_tab[run_only])

    def spline_interp(self, ws, run_only):
        return [self.power_spline, self.ct_spline][run_only](ws)

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
        power, ct = [self.np_interp(ws, run_only=i) for i in [0, 1]]

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


class PowerCtFunctionList(WindTurbineFunctionList, PowerCtFunction):
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

        PowerCtFunction.__init__(
            self,
            input_keys=['ws'],
            power_ct_func=self.__call__,
            power_unit='w',
            additional_models=additional_models)
        WindTurbineFunctionList.__init__(self, key=key,
                                         windTurbineFunction_lst=powerCtFunction_lst, default_value=default_value)


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
        self.interp = [RegularGridInterpolator(value_lst, power_arr),
                       RegularGridInterpolator(value_lst, ct_arr)]
        PowerCtFunction.__init__(self, input_keys, self._power_ct, power_unit,
                                 default_value_dict.keys(), additional_models)

    def _power_ct(self, ws, run_only, **kwargs):
        kwargs = {**self.default_value_dict, 'ws': ws, **{k: v for k, v in kwargs.items() if v is not None}}

        args = np.moveaxis([fix_shape(kwargs[k], ws)
                            for k in self.input_keys], 0, -1)
        try:
            return self.interp[run_only](args)
        except ValueError:
            check_input(self.interp[run_only].grid, args.T, self.input_keys)


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
        ds = ds[['power', 'ct']]
        power_arr, ct_arr = ds.to_array()

        if list(power_arr.dims).index('ws') > 0:
            power_arr, ct_arr = ds.transpose(*(['ws'] + [k for k in power_arr.dims if k != 'ws'])).to_array()

        PowerCtNDTabular.__init__(self, power_arr.dims, [power_arr[k].values for k in power_arr.dims],
                                  power_arr.values, power_unit,
                                  ct_arr.values, additional_models=additional_models)


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

    def _power(self, ws):
        ws = np.asarray(ws)
        return np.where((ws > self.ws_cutin) & (ws <= self.ws_cutout),
                        np.minimum(self.power_rated * ((ws - self.ws_cutin) / (self.ws_rated - self.ws_cutin))**3,
                                   self.power_rated),
                        0)

    def _power_ct(self, ws, run_only):
        return (self._power, self._ct)[run_only](ws)

    def _ct(self, ws):
        ws = np.asarray(ws)

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

        return ct

    def get_power_ct_grad_func(self):
        return self._power_ct_grad

    def _power_ct_grad(self, ws, run_only):
        if run_only == 0:
            return np.where((ws > self.ws_cutin) & (ws <= self.ws_rated),
                            3 * self.power_rated * (ws - self.ws_cutin)**2 / (self.ws_rated - self.ws_cutin)**3,
                            0)
        else:
            dct = ws * 0
            if self.ct_idle is not None:
                dct = np.where((ws > self.ws_rated),
                               self.dct_rated2cutout(ws),
                               0)  # constant ct
            return dct


class PowerCtSurrogate(PowerCtFunction, FunctionSurrogates):
    def __init__(self, power_surrogate, power_unit, ct_surrogate, input_parser, additional_models=[]):
        assert power_surrogate.input_channel_names == ct_surrogate.input_channel_names

        PowerCtFunction.__init__(
            self,
            input_keys=['ws'],  # dummy, overriden below
            power_ct_func=self._power_ct,
            power_unit=power_unit,
            optional_inputs=[],  # dummy, overriden below
            additional_models=additional_models)
        FunctionSurrogates.__init__(self, [power_surrogate, ct_surrogate], input_parser, output_keys=['power', 'ct'])

    def _power_ct(self, ws, run_only=slice(None), **kwargs):
        return FunctionSurrogates.__call__(self, ws, run_only, **kwargs)
