import numpy as np
from numpy import asarray as np_asarray
from numpy import asanyarray as np_asanyarray
import numpy
import autograd.numpy as anp
from autograd.numpy.numpy_boxes import ArrayBox
import inspect
from autograd.core import defvjp, primitive
from autograd.differential_operators import grad, jacobian, elementwise_grad
import matplotlib.pyplot as plt
import sys
import os
import py_wake
from pathlib import Path
from inspect import signature
from functools import wraps
from xarray.core.dataarray import DataArray
from xarray.core import variable
from py_wake.utils import gradients
from scipy.interpolate._cubic import PchipInterpolator as scipy_PchipInterpolator

from itertools import count
from scipy.interpolate import UnivariateSpline as scipy_UnivariateSpline


def asarray(x, dtype=None, order=None):
    if isinstance(x, (ArrayBox)):
        return x
    # elif isinstance(x, DataArray) and isinstance(x.values, ArrayBox):
    #     return x.values
    return np_asarray(x, dtype, order)


# def asanyarray(x, dtype=None, order=None):
#     if isinstance(x, (ArrayBox)):
#         return x
#     elif isinstance(x, DataArray) and isinstance(x.values, ArrayBox):
#         return x.values
#     return np_asanyarray(x, dtype, order)


def minimum(x1, x2, out=None, where=True, **kwargs):
    if isinstance(x1, ArrayBox) or isinstance(x2, ArrayBox):
        return anp.where((x2 < x1) & where, x2, x1)
    else:
        return numpy.minimum(x1, x2, out=out, where=where, **kwargs)


def negative(x1, out=None, where=True, **kwargs):
    # if out is None:
    #     return numpy.negative(x1, out=out, where=where, **kwargs)
    # else:
    assert out is not None
    return anp.where(where, -x1, x1)


# replace functions to support autograd
anp.asarray = asarray
anp.minimum = minimum
anp.negative = negative


variable.np.asarray = gradients.asarray


# replace dsqrt to avoid divide by zero if x=0
eps = 2 * np.finfo(float).eps ** 2
defvjp(anp.sqrt, lambda ans, x: lambda g: g * 0.5 * np.where(x == 0, eps, x)**-0.5)  # @UndefinedVariable


class _use_autograd_in():
    def __init__(self, modules=["py_wake."]):
        self.dict_lst = []
        for m in modules:
            self.dict_lst.extend(self.get_dict(m))

    def get_dict(self, m):
        if isinstance(m, dict):
            return [m]
        if isinstance(m, str):
            def is_submodule(k, m):
                if k.startswith(m):
                    return True

                mod = sys.modules[k]
                if hasattr(mod, '__file__') and sys.modules[k].__file__:
                    mod_addr = os.path.relpath(mod.__file__,
                                               Path(py_wake.__file__).parent.parent)[:-3].replace("\\", '.')
                    if mod_addr.startswith(m):
                        return True
                return False

            return [v.__dict__ for k, v in sys.modules.items()
                    if (is_submodule(k, m) and k != __name__ and getattr(v, 'np', None) == np)]

        if inspect.ismodule(m):
            return [m.__dict__]

        if inspect.getmodule(m) is not None and 'np' in inspect.getmodule(m).__dict__:
            return [inspect.getmodule(m).__dict__]
        else:
            return []

    def __enter__(self):
        try:
            self.prev_np = {}
            for d in self.dict_lst:
                self.prev_np[d["__name__"]] = d['np']
                d['np'] = anp
        finally:
            return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for d in self.dict_lst:
            d['np'] = self.prev_np[d["__name__"]]

    def __call__(self, f):
        def wrap(*args, **kwargs):
            with self:
                return f(*args, **kwargs)
        return wrap


def set_vjp(df):
    def get_func(f, df_lst=df):

        if not isinstance(df, (list, tuple)):
            df_lst = [df_lst]

        pf = primitive(f)

        if inspect.getfullargspec(f).args[0] == 'self':
            defvjp(pf, *[lambda ans, *args: lambda g: g * df(*args) for df in df_lst], argnums=count(1))
        else:
            defvjp(pf, *[lambda ans, *args: lambda g: g * df(*args) for df in df_lst], argnums=count())
        return pf
    return get_func


def _step_grad(f, argnum, step_func, step, vector_interdependence):
    if isinstance(argnum, (list, tuple)):
        return lambda *args, **kwargs: [_step_grad(f, i, step_func, step,
                                                   vector_interdependence)(*args, **kwargs) for i in argnum]
    else:
        f_signature = signature(f)

        def wrap(*args, **kwargs):
            bound_arguments = f_signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args, kwargs = bound_arguments.args, bound_arguments.kwargs
            x = np.atleast_1d(args[argnum]).astype(float)
            if 'ref' in inspect.getfullargspec(step_func).args:
                ref = np.asarray(f(*args, **kwargs))
            else:
                ref = None
            if vector_interdependence:
                return np.array([step_func(f(*(args[:argnum] + (x_,) + args[argnum + 1:]), **kwargs), ref, step)
                                 for x_ in x + np.diag(np.ones_like(x) * step)]).T
            else:
                return step_func(f(*(args[:argnum] + (x + step,) + args[argnum + 1:]), **kwargs), ref, step)
        fname = getattr(f, '__name__', f'{f.__class__.__name__}.{f.__call__.__name__}')
        wrap.__name__ = "%s_of_%s_wrt_argnum_%d" % (step_func.__name__, fname, argnum)
        return wrap


def fd(f, vector_interdependence=False, argnum=0, step=1e-6):
    def fd_gradient(res, ref, step):
        return (res - ref) / step
    return _step_grad(f, argnum, fd_gradient, step, vector_interdependence)


def cs(f, vector_interdependence=False, argnum=0, step=1e-20):
    def cs_gradient(res, _, step):
        return np.imag(res) / np.imag(step)
    return _step_grad(f, argnum, cs_gradient, step * 1j, vector_interdependence)


def autograd(f, vector_interdependence=False, argnum=0):
    if isinstance(argnum, (list, tuple)):
        return lambda *args, **kwargs: [autograd(f, vector_interdependence, i)(*args, **kwargs) for i in argnum]
    else:
        if vector_interdependence:
            grad_func = jacobian(f, argnum)
        else:
            grad_func = elementwise_grad(f, argnum)

        @wraps(grad_func)
        def wrap(*args, **kwargs):
            bound_arguments = signature(f).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args = bound_arguments.args
            return grad_func(*(args[:argnum] + (anp.asarray(args[argnum], dtype=float),) + args[argnum + 1:]),
                             **bound_arguments.kwargs)

        return _use_autograd_in(modules=['py_wake.', f])(wrap)


color_dict = {}


def plot_gradients(f, dfdx, x, label, step=1, ax=None):
    global color_dict
    if ax is None:
        ax = plt
    c = color_dict.get(label, None)
    step = np.array([-step, 0, step])

    c = ax.plot(x + step, f + step * dfdx, ".-", color=c, label=('', label)[c is None])[0].get_color()

    if label not in color_dict:
        color_dict[label] = c
    ax.legend()


def hypot(a, b):
    """
    Given the “legs” of a right triangle, return its hypotenuse.

    Calls numpy.hypot(a, b) for real arguments and np.sqrt(a**2 + b**2) for complex arguments.

    Parameters
    ----------
    a, b : real or complex array_like
        Leg of the triangle(s).

    Returns
    -------
    c : real or complex array_like
        The hypotenuse of the triangle(s).
    """
    if isinstance(a, ArrayBox) or isinstance(b, ArrayBox):
        return anp.sqrt(a**2 + b**2)
    elif np.isrealobj(a) and np.isrealobj(b):
        return np.hypot(a, b)
    else:
        return np.sqrt(a**2 + b**2)


def cabs(a):
    """Absolute (non-negative) value for both real and complex number"""
    if isinstance(a, ArrayBox):
        return anp.abs(a)
    elif np.isrealobj(a):
        return np.abs(a)
    else:
        return np.where(a < 0, -a, a)


def dinterp(xp, x, y):
    if len(x) > 1:
        return np.interp(xp, np.repeat(x, 2)[1:-1], np.repeat(np.diff(y) / np.diff(x), 2))
    else:
        return np.ones_like(xp)


@primitive
def interp(xp, x, y, *args, **kwargs):
    if np.isrealobj(xp):
        return np.interp(xp, x, y, *args, **kwargs)
    else:
        yp = np.interp(xp.real, x, y, *args, **kwargs)
        dyp = dinterp(xp.real, x, y)
        return yp + xp.imag * 1j * dyp


defvjp(interp, lambda ans, xp, x, y: lambda g: g * dinterp(xp, x, y))


def logaddexp(x, y):
    if isinstance(x, ArrayBox) or isinstance(y, ArrayBox):
        return anp.logaddexp(x, y)
    elif np.isrealobj(x) and np.isrealobj(y):
        return np.logaddexp(x, y)
    else:
        x, y = map(np.asarray, [x, y])
        ans = np.logaddexp(x.real, y.real)
        return ans + 1j * (x.imag * np.exp(x - ans) + y.imag * np.exp(y - ans))


class PchipInterpolator(scipy_PchipInterpolator):
    def df(self, x, extrapolate=None):
        return scipy_PchipInterpolator.__call__(self, x, nu=1, extrapolate=extrapolate)

    @set_vjp(df)
    def __call__(self, x, extrapolate=None):
        y = scipy_PchipInterpolator.__call__(self, np.real(x), extrapolate=extrapolate)
        if np.iscomplexobj(x):
            dy = scipy_PchipInterpolator.__call__(self, np.real(x), nu=1, extrapolate=extrapolate)
            y = y + x.imag * dy * 1j
        return y


class UnivariateSpline(scipy_UnivariateSpline):
    def df(self, x, ext=None):
        return scipy_UnivariateSpline.__call__(self, x, nu=1, ext=ext)

    @set_vjp(df)
    def __call__(self, x, ext=None):
        y = scipy_UnivariateSpline.__call__(self, np.real(x), ext=ext)
        if np.iscomplexobj(x):
            dy = scipy_UnivariateSpline.__call__(self, np.real(x), nu=1, ext=ext)
            y = y + x.imag * dy * 1j
        return y
