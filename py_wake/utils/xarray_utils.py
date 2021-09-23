from numpy import newaxis as na

import numpy as np
import xarray as xr
from xarray.plot.plot import _PlotMethods
import warnings
from xarray.core.dataarray import DataArray


class ilk():
    def __init__(self, dataArray):
        self.dataArray = dataArray

    def __call__(self, shape=None):
        dims = self.dataArray.dims
        squeeze_dims = [d for d in self.dataArray.dims if d not in ['i', 'wt', 'wd', 'ws', 'time']]
        v = self.dataArray.squeeze(squeeze_dims, drop=True).data
        if 'wt' not in dims and 'i' not in dims:
            v = v[na]
        if 'time' in dims:
            v = v[:, :, na]
        else:
            if 'wd' not in dims:
                v = v[:, na]
            if 'ws' not in dims:
                v = v[:, :, na]

        if shape is None:
            return v
        else:
            return np.broadcast_to(v, shape)


class add_ilk():
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, name, value):
        dims = self.dataset.dims
        if 'time' in dims:
            allowed_dims = ['i', 'wt'], ['time'], ['ws']
        else:
            allowed_dims = ['i', 'wt'], ['wd'], ['ws']

        d = []
        i = 0

        for ad in allowed_dims:
            for k in ad:
                if i < len(np.shape(value)) and np.shape(value)[i] == dims.get(k, None):
                    d.append(k)
                    i += 1
                    break
        while len(np.shape(value)) > len(d) and np.shape(value)[-1] == 1:
            value = value[..., 0]
        self.dataset[name] = (d, da2py(value, include_dims=False))


class add_ijlk():
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, name, value):
        dims = self.dataset.dims
        if 'time' in dims:
            allowed_dims = ['i', 'wt'], ['i', 'wt'], ['time'], ['ws']
        else:
            allowed_dims = ['i', 'wt'], ['i', 'wt'], ['wd'], ['ws']

        d = []
        i = 0

        for ad in allowed_dims:
            for k in ad:
                if i < len(np.shape(value)) and np.shape(value)[i] == dims.get(k, None):
                    d.append(k)
                    i += 1
                    break
#         while len(value.shape) > len(d) and value.shape[-1] == 1:
#             value = value[..., 0]
        self.dataset[name] = (d, value)


class interp_all():
    def __init__(self, dataArray):
        self.dataArray = dataArray

    def __call__(self, dataArray2, **kwargs):
        interp_coords = {d: dataArray2[d] for d in self.dataArray.dims if d in dataArray2.coords}
        return self.dataArray.interp(**interp_coords, **kwargs)


class sel_interp_all():
    def __init__(self, dataArray):
        self.dataArray = dataArray

    def __call__(self, coords, method="linear", bounds_error=True, **kwargs):
        interp_coords = {}
        da = self.dataArray
        for d in self.dataArray.dims:
            if d in coords:
                try:
                    da = da.sel({d: coords[d]})
                except (KeyError, IndexError):
                    interp_coords[d] = coords[d]
        kwargs['bounds_error'] = bounds_error
        return da.interp(**interp_coords, method=method, kwargs=kwargs)


class plot_xy_map():
    def __init__(self, dataArray):
        self.dataArray = dataArray

    def __call__(self, **kwargs):
        if ('x' in self.dataArray.coords and 'y' in self.dataArray.coords and 'x' not in kwargs and

                self.dataArray.squeeze().shape == (len(np.atleast_1d(self.dataArray.x)), len(np.atleast_1d(self.dataArray.y)))):
            kwargs['x'] = 'x'
        _PlotMethods(self.dataArray)(**kwargs)


if not hasattr(xr.DataArray(None), 'ilk'):
    xr.register_dataarray_accessor("ilk")(ilk)
    xr.register_dataset_accessor("add_ilk")(add_ilk)
    xr.register_dataset_accessor("add_ijlk")(add_ijlk)
    xr.register_dataarray_accessor("interp_all")(interp_all)
    xr.register_dataarray_accessor("sel_interp_all")(sel_interp_all)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xr.register_dataarray_accessor('plot')(plot_xy_map)


def da2py(v, include_dims=False):
    if isinstance(v, tuple):
        return tuple([da2py(v, include_dims) for v in v])
    if isinstance(v, DataArray):
        if include_dims:
            return (v.dims, v.values)
        else:
            return v.values
    return v
