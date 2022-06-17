from py_wake import np
from numpy import asarray as np_asarray
# from numpy import asanyarray as np_asanyarray
import numpy
import autograd.numpy as anp
from autograd.numpy.numpy_boxes import ArrayBox
import inspect
from autograd.core import defvjp, primitive
from autograd.differential_operators import jacobian, elementwise_grad
from inspect import signature
from functools import wraps
from xarray.core.dataarray import DataArray
from xarray.core import variable
from py_wake.utils import gradients
from scipy.interpolate._cubic import PchipInterpolator as scipy_PchipInterpolator

from itertools import count
from scipy.interpolate import UnivariateSpline as scipy_UnivariateSpline
from scipy.special import erf as scipy_erf
from autograd.scipy.special import erf as autograd_erf
from py_wake.utils.numpy_utils import AutogradNumpy
from autograd.numpy.numpy_vjps import unbroadcast_f


def asarray(x, dtype=None, order=None):
    if isinstance(x, (ArrayBox)):
        return x
    elif isinstance(x, DataArray) and isinstance(x.values, ArrayBox):
        return x.values
    return np_asarray(x, dtype, order)


# def asanyarray(x, dtype=None, order=None):
#     if isinstance(x, (ArrayBox)):
#         return x
#     elif isinstance(x, DataArray) and isinstance(x.values, ArrayBox):
#         return x.values
#     return np_asanyarray(x, dtype, order)


def minimum(x1, x2, out=None, where=True, **kwargs):
    if isinstance(x1, ArrayBox) or isinstance(x2, ArrayBox):
        return anp.where((x2 < x1) & where, x2, x1)  # @UndefinedVariable
    else:
        return numpy.minimum(x1, x2, out=out, where=where, **kwargs)


def negative(x1, out=None, where=True, **kwargs):
    # if out is None:
    #     return numpy.negative(x1, out=out, where=where, **kwargs)
    # else:
    assert out is not None
    return anp.where(where, -x1, x1)  # @UndefinedVariable


# replace functions to support autograd
anp.asarray = asarray
anp.minimum = minimum
anp.negative = negative


variable.np.asarray = gradients.asarray


# replace dsqrt to avoid divide by zero if x=0
eps = 2 * np.finfo(float).eps ** 2
defvjp(anp.sqrt, lambda ans, x: lambda g: g * 0.5 * np.where(x == 0, eps, x)**-0.5)  # @UndefinedVariable
defvjp(anp.arctan2,  # @UndefinedVariable
       lambda ans, x, y: unbroadcast_f(x, lambda g: g * y / (x**2 + np.where(y != 0, y, 1)**2)),
       lambda ans, x, y: unbroadcast_f(y, lambda g: g * -x / (np.where(x != 0, x, 1)**2 + y**2)))


def set_gradient_function(df):
    def get_func(f, df_lst=df):
        if not isinstance(df, (list, tuple)):
            df_lst = [df_lst]

        vjp = [lambda ans, *args, df=df, **kwargs: lambda g: g * df(*args, **kwargs) for df in df_lst]
        first_arg = int(len(inspect.getfullargspec(f).args) > 0 and inspect.getfullargspec(f).args[0] == 'self')

        return set_vjp(vjp, first_arg)(f)

    return get_func


def set_vjp(vjp_lst, first_arg=0):
    # set vjp (vector jacobian product) similar to this
    # lambda ans, *args, **kwargs: lambda g: g * gradient_function(*args, **kwargs)
    def get_func(f, vjp_lst=vjp_lst):

        pf = primitive(f)

        def fkwargs(*args, **kwargs):
            args, kwargs = kwargs2args(f, *args, **kwargs)
            return pf(*args, **kwargs)

        defvjp(pf, *vjp_lst, argnums=count(first_arg))
        return fkwargs
    return get_func


def _step_grad(f, argnum, step_func, step, vector_interdependence):
    if isinstance(argnum, (list, tuple)):
        return lambda *args, **kwargs: [_step_grad(f, i, step_func, step,
                                                   vector_interdependence)(*args, **kwargs) for i in argnum]
    else:
        def wrap(*args, **kwargs):
            args, kwargs = kwargs2args(f, *args, **kwargs)

            x = args[argnum]
            x_shape = np.shape(x)
            x = np.atleast_1d(x).flatten().astype(float)

            if 'ref' in inspect.getfullargspec(step_func).args:
                ref = np.asarray(f(*(args[:argnum] + (np.reshape(x, x_shape),) + args[argnum + 1:]), **kwargs))
            else:
                ref = None
            if vector_interdependence:
                res = np.moveaxis([step_func(f(*(args[:argnum] + (np.reshape(x_, x_shape),) + args[argnum + 1:]),
                                               **kwargs), ref, step)
                                   for x_ in x + np.diag(np.ones_like(x) * step)], 0, -1)
            else:
                res = step_func(f(*(args[:argnum] + (np.reshape(x + step, x_shape),) +
                                    args[argnum + 1:]), **kwargs), ref, step)
            return np.reshape(res, res.shape[:-1] + x_shape)
        fname = getattr(f, '__name__', f'{f.__class__.__name__}.{f.__call__.__name__}')
        wrap.__name__ = "%s_of_%s_wrt_argnum_%d" % (step_func.__name__, fname, argnum)
        return wrap


def fd(f, vector_interdependence=True, argnum=0, step=1e-6):
    def fd_gradient(res, ref, step):
        return (res - ref) / step
    return _step_grad(f, argnum, fd_gradient, step, vector_interdependence)


def cs(f, vector_interdependence=True, argnum=0, step=1e-20):
    def cs_gradient(res, _, step):
        return np.imag(res) / np.imag(step)
    return _step_grad(f, argnum, cs_gradient, step * 1j, vector_interdependence)


def kwargs2args(f, *args, **kwargs):
    # if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], dict) and not kwargs:
    #     kwargs = args[0]
    #     args = tuple()

    bound_arguments = signature(f).bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return bound_arguments.args, bound_arguments.kwargs


def autograd(f, vector_interdependence=True, argnum=0):
    if isinstance(argnum, (list, tuple)):
        # Calculating with respect to a merged xy list instead of first x then y takes around 40% less time
        # Here wrap collects the <argnum> arguments into one vector, computes the gradients wrt. this vector
        # and finally reshape the result
        def wrap(*args, **kwargs):
            args, kwargs = kwargs2args(f, *args, **kwargs)
            wrt_args = [args[i] for i in argnum]
            args = [a for i, a in enumerate(args) if i not in argnum]
            wrt_arg_shape = [np.shape(arg) for arg in wrt_args]
            wrt_1arg = np.concatenate([np.ravel(a) for a in wrt_args])
            wrt_arg_i = np.r_[0, np.cumsum([np.prod(s) for s in wrt_arg_shape]).astype(int)]

            def wrap_1inp(inp, *args):
                wrt_args = [inp[i0:i1].reshape(s) for i0, i1, s in zip(wrt_arg_i[:-1], wrt_arg_i[1:], wrt_arg_shape)]
                args = list(args)
                for i, wrt_arg in zip(argnum, wrt_args):
                    args.insert(i, wrt_arg)
                return f(*args, **kwargs)
            wrap_1inp.org_f = f
            dfdinp = autograd(wrap_1inp, vector_interdependence=vector_interdependence)(wrt_1arg, *args)
            return [dfdinp[i0:i1].reshape(s) for i0, i1, s in zip(wrt_arg_i[:-1], wrt_arg_i[1:], wrt_arg_shape)]
        return wrap
    else:
        if vector_interdependence:
            grad_func = jacobian(f, argnum)
        else:
            grad_func = elementwise_grad(f, argnum)

        @wraps(grad_func)
        def wrap2(*args, **kwargs):
            with AutogradNumpy():
                args, kwargs = kwargs2args(f, *args, **kwargs)
                return grad_func(*(args[:argnum] + (np.asarray(args[argnum], dtype=np.float),) + args[argnum + 1:]),  # @UndefinedVariable
                                 **kwargs)
        return wrap2


color_dict = {}


def plot_gradients(f, dfdx, x, label, step=1, ax=None):
    import matplotlib.pyplot as plt
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
    Given the "legs" of a right triangle, return its hypotenuse.

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
        return anp.sqrt(a**2 + b**2)  # @UndefinedVariable
    elif np.isrealobj(a) and np.isrealobj(b):
        return np.hypot(a, b)
    else:
        return np.sqrt(a**2 + b**2)


def cabs(a):
    """Absolute (non-negative) value for both real and complex number"""
    if isinstance(a, ArrayBox):
        return anp.abs(a)  # @UndefinedVariable
    elif np.isrealobj(a):
        return np.abs(a)
    else:
        return np.where(a < 0, -a, a)


def dinterp_dxp(xp, x, y):
    if len(x) > 1:
        return np.interp(xp, np.repeat(x, 2)[1:-1], np.repeat(np.diff(y) / np.diff(x), 2))
    else:
        return np.ones_like(xp)


@set_gradient_function([dinterp_dxp])
def interp(xp, x, y, *args, **kwargs):
    if all([np.isrealobj(v) for v in [xp, x, y]]):
        return np.interp(xp, x, y, *args, **kwargs)
    else:
        # yp = np.interp(xp.real, x.real, y.real, *args, **kwargs)
        # dyp_dxp = dinterp_dxp(xp.real, x.real, y.real)
        # dyp_dx = fd(np.interp, True, 1)(xp.real, x.real, y.real)
        # dyp_dy = fd(np.interp, True, 2)(xp.real, x.real, y.real)
        # return yp + 1j * (xp.imag * dyp_dxp + x.imag * dyp_dx + y.imag * dyp_dy)

        yp = np.interp(xp.real, x, y, *args, **kwargs)
        dyp = dinterp_dxp(xp.real, x, y)
        return yp + xp.imag * 1j * dyp


def logaddexp(x, y):
    if isinstance(x, ArrayBox) or isinstance(y, ArrayBox):
        return anp.logaddexp(x, y)  # @UndefinedVariable
    elif np.isrealobj(x) and np.isrealobj(y):
        return np.logaddexp(x, y)
    else:
        x, y = map(np.asarray, [x, y])
        ans = np.logaddexp(x.real, y.real)
        return ans + 1j * (x.imag * np.exp(x - ans) + y.imag * np.exp(y - ans))


class PchipInterpolator(scipy_PchipInterpolator):
    def df(self, x, extrapolate=None):
        return scipy_PchipInterpolator.__call__(self, x, nu=1, extrapolate=extrapolate)

    @set_gradient_function(df)
    def __call__(self, x, extrapolate=None):
        y = scipy_PchipInterpolator.__call__(self, np.real(x), extrapolate=extrapolate)
        if np.iscomplexobj(x):
            dy = scipy_PchipInterpolator.__call__(self, np.real(x), nu=1, extrapolate=extrapolate)
            y = y + x.imag * dy * 1j
        return y


class UnivariateSpline(scipy_UnivariateSpline):
    def df(self, x, ext=None):
        return scipy_UnivariateSpline.__call__(self, x, nu=1, ext=ext)

    @set_gradient_function(df)
    def __call__(self, x, ext=None):
        y = scipy_UnivariateSpline.__call__(self, np.real(x), ext=ext)
        if np.iscomplexobj(x):
            dy = scipy_UnivariateSpline.__call__(self, np.real(x), nu=1, ext=ext)
            y = y + x.imag * dy * 1j
        return y


def erf(z):
    if isinstance(z, ArrayBox):
        return autograd_erf(z)
    else:
        return scipy_erf(z)


# def get_dtype(arg_lst):
#     return (float, np.complex128)[any([np.iscomplexobj(v) for v in arg_lst])]
def trapz(y, x, axis=-1):
    if isinstance(y, ArrayBox) or isinstance(x, ArrayBox):
        x, y = asarray(x), asarray(y)
        axis = np.arange(len(np.shape(y)))[axis]
        # Silly implementation but np.take, np.diff and np.trapz did not seem to work with autograd
        # I tried to implement gradients of np.trapz manually but failed to make it work for arbitrary axis
        if axis == 0:
            return ((y[:-1] + y[1:]) / 2 * (x[1:] - x[:-1])).sum(0)
        elif axis == 1:
            return ((y[:, :-1] + y[:, 1:]) / 2 * (x[:, 1:] - x[:, :-1])).sum(1)
        elif axis == 2:
            return ((y[:, :, :-1] + y[:, :, 1:]) / 2 * (x[:, :, 1:] - x[:, :, :-1])).sum(2)
        elif axis == 4:
            return ((y[:, :, :, :, :-1] + y[:, :, :, :, 1:]) / 2 * (x[:, :, :, :, 1:] - x[:, :, :, :, :-1])).sum(4)
        else:   # pragma: no cover
            raise NotImplementedError()
    else:
        return np.trapz(y, x, axis=axis)


def mod(x1, x2):
    if np.iscomplexobj(x1):
        return np.mod(np.real(x1), x2) + x1.imag * 1j
    elif isinstance(x1, ArrayBox) or isinstance(x2, ArrayBox):
        return anp.mod(x1, x2)  # @UndefinedVariable
    else:
        return np.mod(x1, x2)


def modf(i):
    if isinstance(i, ArrayBox):
        i0 = i._value.astype(int)
    else:
        i0 = np.real(i).astype(int)
    i_f = i - i0
    return i_f, i0


def arctan2(y, x):
    if np.iscomplexobj(y) or np.iscomplexobj(x):
        r = np.atleast_1d(np.sign(y.real) * np.pi / 2).astype(np.complex128)
        m = x.real != 0
        r[m] = np.arctan(np.atleast_1d(y)[m] / np.atleast_1d(x)[m])
        r[(x.real < 0) & (y.real >= 0)] += np.pi
        r[(x.real < 0) & (y.real < 0)] -= np.pi
        return np.reshape(r, np.shape(y))
    elif isinstance(y, ArrayBox) or isinstance(x, ArrayBox):
        return anp.arctan2(y, x)  # @UndefinedVariable
    else:
        return np.arctan2(y, x)


def rad2deg(rad):
    return rad * 180 / np.pi


def deg2rad(deg):
    return deg * np.pi / 180
