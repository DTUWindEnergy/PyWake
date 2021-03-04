import numpy as np
import autograd.numpy as anp
from autograd.numpy.numpy_boxes import ArrayBox
from contextlib import contextmanager
import inspect
from autograd.core import defvjp
import numpy as np
from autograd.differential_operators import grad, jacobian, elementwise_grad
import matplotlib.pyplot as plt
import sys
from autograd.builtins import SequenceBox


def asarray(x, dtype=None, order=None):
    if isinstance(x, (ArrayBox)):
        return x
#     if any([isinstance(_x, (ArrayBox, SequenceBox)) for _x in np.atleast_1d(x).flatten()]):
#         return x
    return np.asarray(x, dtype, order)


# replace asarray to support autograd
anp.asarray = asarray


# replace dsqrt to avoid divide by zero if x=0
eps = 2 * np.finfo(float).eps ** 2
defvjp(anp.sqrt, lambda ans, x: lambda g: g * 0.5 * np.where(x == 0, eps, x)**-0.5)  # @UndefinedVariable


@contextmanager
def use_autograd_in(modules=["py_wake."]):

    def get_dict(m):
        if isinstance(m, dict):
            return [m]
        if isinstance(m, str):
            return [v.__dict__ for k, v in sys.modules.items()
                    if k.startswith(m) and k != __name__ and getattr(v, 'np', None) == np]

        if inspect.ismodule(m):
            return [m.__dict__]

        return [inspect.getmodule(m).__dict__]

    dict_lst = []
    for m in modules:
        dict_lst.extend(get_dict(m))

    try:
        prev_np = {}
        for d in dict_lst:
            prev_np[d["__name__"]] = d['np']
            d['np'] = anp
        yield
    finally:
        for d in dict_lst:
            d['np'] = prev_np[d["__name__"]]


def _step_grad(f, argnum, step_func, step, vector_interdependence):
    def wrap(*args, **kwargs):
        x = np.atleast_1d(args[argnum]).astype(float)
        ref = np.asarray(f(*args, **kwargs))
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
    if vector_interdependence:
        return jacobian(f, argnum)
    else:
        return elementwise_grad(f, argnum)


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
    plt.legend()
