import numpy as np
import inspect
from abc import ABC, abstractmethod
import types
"""
sequenceDiagram

    User->>+PowerCtFunction: call(ws)
    PowerCtFunction->>WindTurbineFunction: call(ws)
    WindTurbineFunction->>WindTurbineFunction: check for unused input
    WindTurbineFunction->>+PowerCtFunction: evaluate
    participant PMC as PowerCtModelContainer
    PowerCtFunction->>+PMC: call

    PMC->>+PMC: recursive_wrap
    participant CP as CubePowerSimpleCt
    PMC->>+DensityScale:call

    DensityScale->>PMC:recursive_wrap
    PMC->>+PMC: recursive_wrap
    PMC->>+CP: power_ct
    CP->>-PMC: power/ct
    PMC->>-PMC:
    PMC->>-DensityScale:recursive_wrap
    DensityScale->>-PMC:recursive_wrap
    PMC->>-PMC:
    PMC->>PowerCtFunction: test
    PowerCtFunction->>PowerCtFunction: scale power
    PowerCtFunction->>-User: power/ct
"""


class WindTurbineFunction():
    """Base class for all PowerCtModel classes"""

    def __init__(self, input_keys, optional_inputs):
        assert input_keys[0] == 'ws'
        required_inputs = [k for k in input_keys[1:] if k not in optional_inputs]
        self.input_keys = input_keys

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


class WindTurbineFunctionList(WindTurbineFunction):
    """Wraps a list of PowerCtFunction objects by adding a new discrete input argument,
    representing the index of the PowerCtFunction objects in the list"""

    def __init__(self, key, windTurbineFunction_lst, default_value=None):
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
        required_inputs.extend([pcct.required_inputs for pcct in windTurbineFunction_lst])
        optional_inputs.extend([pcct.optional_inputs for pcct in windTurbineFunction_lst])
        WindTurbineFunction.__init__(self, ['ws'] + required_inputs + optional_inputs,
                                     optional_inputs=optional_inputs)

        self.windTurbineFunction_lst = windTurbineFunction_lst
        self.default_value = default_value
        self.key = key

    def _subset(self, arr, mask):
        if arr is None or isinstance(arr, types.FunctionType):
            return arr
        return np.broadcast_to(arr.reshape(arr.shape + (1,) * (len(mask.shape) - len(arr.shape))), mask.shape)[mask]

    def enable_autograd(self):
        for f in self.windTurbineFunction_lst:
            f.enable_autograd()

    def __call__(self, ws, run_only=slice(None), **kwargs):
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
            res = self.windTurbineFunction_lst[idx](ws, **kwargs)
        else:
            res = np.empty((2,) + np.asarray(ws).shape)
            unique_idx = np.unique(idx)
            idx = np.zeros(ws.shape, dtype=int) + idx.reshape(idx.shape + (1,) * (len(ws.shape) - len(idx.shape)))
            for i in unique_idx:
                m = (idx == i)
                res[:, m] = self.windTurbineFunction_lst[i](ws[m], **{k: self._subset(v, m) for k, v in kwargs.items()})
        return res


class FunctionSurrogates(WindTurbineFunction, ABC):
    def __init__(self, function_surrogate_lst, input_parser):
        self.function_surrogate_lst = np.asarray(function_surrogate_lst)
        self.get_input = input_parser
        input_keys = inspect.getfullargspec(self.get_input).args
        if input_keys[0] == 'self':
            input_keys = input_keys[1:]
        defaults = inspect.getfullargspec(self.get_input).defaults
        optional_inputs = input_keys[1:] if defaults is None else input_keys[::-1][:len(defaults)]
        WindTurbineFunction.__init__(self, input_keys, optional_inputs)

    def __call__(self, ws, run_only=slice(None), **kwargs):
        x = self.get_input(ws=ws, **kwargs)
        x = np.array([self.fix_shape(v, ws).ravel() for v in x]).T
        return [fs.predict_output(x).reshape(ws.shape) for fs in np.asarray(self.function_surrogate_lst)[run_only]]

#     Commented out as no tests or examples currently uses this class directly
#     @property
#     def output_keys(self):
#         return [fs.output_channel_name for fs in self.function_surrogate_lst]
#
#     @property
#     def wohler_exponents(self):
#         return [fs.wohler_exponent for fs in self.function_surrogate_lst]
