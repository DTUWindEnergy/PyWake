import numpy
import py_wake
from numpy.lib.index_tricks import RClass
import inspect
import autograd.numpy as autograd_numpy


class NumpyBackend():

    def __enter__(self):
        self.old_backend = py_wake.np.backend
        py_wake.np.set_backend(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        py_wake.np.set_backend(self.old_backend)


class Cast32Wrapper():
    float = numpy.float32
    complex = numpy.complex64
    dtype_dict = {numpy.dtype('float64'): numpy.float32,
                  numpy.dtype('complex128'): numpy.complex64,
                  }

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **{'dtype': self.float, **kwargs})
        except TypeError:  # f does not take dtype argument
            res = self.f(*args, **kwargs)

            try:
                return res.astype(self.dtype_dict[res.dtype])
            except (KeyError, AttributeError):
                return res


class Numpy32(NumpyBackend):
    backend = numpy

    def __init__(self):
        wrapped_functions = {k: Cast32Wrapper(f) for k, f in self.backend.__dict__.items()
                             if not(isinstance(f, (type, int, float, RClass)) or f is None or inspect.ismodule(f))}
        self.__dict__ = {**self.backend.__dict__, **wrapped_functions}
        self.float = numpy.float32
        self.complex = numpy.complex64


class AutogradNumpy32(Numpy32):
    backend = autograd_numpy


class AutogradNumpy(NumpyBackend):

    def __enter__(self):
        self.old_backend = py_wake.np.backend

        if isinstance(self.old_backend, Numpy32):
            py_wake.np.set_backend(AutogradNumpy32())
        else:
            py_wake.np.set_backend(autograd_numpy)
            py_wake.np.backend.float = numpy.float64

    def __exit__(self, exc_type, exc_val, exc_tb):
        py_wake.np.set_backend(self.old_backend)


class NumpyWrapper():
    def __init__(self):
        self.set_backend(numpy)

    @property
    def float(self):
        if self.backend == numpy:
            return numpy.float64
        return getattr(self.backend, 'float', numpy.float64)

    @property
    def complex(self):
        if self.backend == numpy:
            return numpy.complex128
        return getattr(self.backend, 'complex', numpy.complex128)

    def set_backend(self, backend):
        self.backend = backend
        self.__dict__.update(backend.__dict__)
