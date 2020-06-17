from numpy import newaxis as na

import numpy as np
import xarray as xr


class ilk():
    def __init__(self, dataArray):
        self.dataArray = dataArray

    def __call__(self, shape=None):
        dims = self.dataArray.dims
        v = self.dataArray.data
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


xr.register_dataarray_accessor("ilk")(ilk)


class interp_all():
    def __init__(self, dataArray):
        self.dataArray = dataArray

    def __call__(self, dataArray2, **kwargs):
        interp_coords = {d: dataArray2[d] for d in self.dataArray.dims if d in dataArray2}
        return self.dataArray.interp(**interp_coords, **kwargs)


xr.register_dataarray_accessor("interp_all")(interp_all)
