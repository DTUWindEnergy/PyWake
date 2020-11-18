from numpy import newaxis as na

import numpy as np
import xarray as xr


class ilk():
    def __init__(self, dataArray):
        self.dataArray = dataArray

    def __call__(self, shape=None):
        dims = self.dataArray.dims
        squeeze_dims = [d for d in self.dataArray.dims if d not in ['i', 'wt', 'wd', 'ws']]
        v = self.dataArray.squeeze(squeeze_dims, drop=True).data
        if 'wt' not in dims and 'i' not in dims:
            v = v[na]
        if 'wd' not in dims:
            v = v[:, na]
        if 'ws' not in dims:
            v = v[:, :, na]

        if shape is None:
            return v
        else:
            return np.broadcast_to(v, shape)


class interp_all():
    def __init__(self, dataArray):
        self.dataArray = dataArray

    def __call__(self, dataArray2, **kwargs):
        interp_coords = {d: dataArray2[d] for d in self.dataArray.dims if d in dataArray2}
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


if not hasattr(xr.DataArray(None), 'ilk'):
    xr.register_dataarray_accessor("ilk")(ilk)
    xr.register_dataarray_accessor("interp_all")(interp_all)
    xr.register_dataarray_accessor("sel_interp_all")(sel_interp_all)
