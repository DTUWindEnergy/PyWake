from numpy import newaxis as na

from py_wake import np
import xarray as xr
from xarray.plot.plot import _PlotMethods
import warnings
from py_wake.utils.grid_interpolator import GridInterpolator


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
        dtype = (np.float, np.complex)[np.iscomplexobj(v)]
        v = v.astype(dtype)
        if shape is None:
            return v
        else:
            return np.broadcast_to(v, shape)


def ilk2da(v_ilk, coords, desc=None):
    dim_i = ('i', 'wt')['wt' in coords]
    dims = [d for i, d in enumerate([dim_i, ('wd', 'time')['time' in coords], 'ws'])
            if v_ilk.shape[i] > 1]
    coords = {k: v for k, v in coords.items() if k in dims}
    attrs = {}
    if desc:
        attrs = {'description': desc}
    return xr.DataArray(v_ilk.squeeze(), dims=dims, coords=coords, attrs=attrs)


def ijlk2da(v_ijlk, coords):
    dim_i = ('i', 'wt')['wt' in coords]
    dims = [d for s, d in zip(v_ijlk.shape, [dim_i, dim_i, ('wd', 'time')['time' in coords], 'ws'])
            if s > 1]
    coords = {k: v for k, v in coords.items() if k in dims}
    return xr.DataArray(v_ijlk.squeeze(), dims=dims, coords=coords)


class interp_ilk():
    def __init__(self, dataArray):
        self.dataArray = dataArray

    def __call__(self, coords, deg=False, interp_method='linear', bounds='check'):
        # Interpolate via EqDistRegGridInterpolator (equidistance regular grid interpolator) which is much faster
        # than xarray.interp.
        # This function is comprehensive because var can contain any combinations of coordinates (i or (xy,h)) and wd,ws

        var = self.dataArray

        def sel(data, data_dims, indices, dim_name):
            i = data_dims.index(dim_name)
            ix = tuple([(slice(None), indices)[dim == i] for dim in range(data.ndim)])
            return data[ix]

        ip_dims = [n for n in ['i', 'x', 'y', 'h', 'time', 'wd', 'ws'] if n in var.dims]  # interpolation dimensions
        data = var.data
        data_dims = var.dims

        def pre_sel(data, name):
            # If only a single value is needed on the <name>-dimension, the data is squeezed to contain this value only
            # Otherwise the indices of the needed values are returned
            if name not in var.dims:
                return data, None
            c, v = coords[name], var[name].data
            indices = None
            if ip_dims and ip_dims[-1] == name and len(set(c) - set(np.atleast_1d(v))) == 0:
                # all coordinates in var, no need to interpolate
                ip_dims.remove(name)
                indices = np.searchsorted(v, c)
                if len(np.unique(indices)) == 1:
                    # only one index, select before interpolation
                    data = sel(data, data_dims, slice(indices[0], indices[0] + 1), name)
                    indices = [0]
                else:
                    indices = indices
            return data, indices

        # pre select, i.e. reduce input data size in case only one ws or wd is needed
        data, k_indices = pre_sel(data, 'ws')
        l_name = ['wd', 'time']['time' in coords]
        data, l_indices = pre_sel(data, l_name)

        if 'i' in ip_dims and 'i' in coords and len(var.i) != len(coords['i']):
            raise ValueError(
                "Number of points, i(=%d), in site data variable, %s, must match number of requested points(=%d)" %
                (len(var.i), var.name, len(coords['i'])))
        data, i_indices = pre_sel(data, 'i')

        if len(ip_dims) > 0:
            grid_interp = GridInterpolator([var.coords[k].data for k in ip_dims], data,
                                           method=interp_method, bounds=bounds)

            # get dimension of interpolation coordinates
            I = (1, len(coords.get('x', coords.get('y', coords.get('h', coords.get('i', [None]))))))[
                any([n in data_dims for n in 'xyhi'])]
            L, K = [(1, len(coords.get(n, [None])))[indices is None and n in data_dims]
                    for n, indices in [('wd', l_indices), ('ws', k_indices)]]

            # gather interpolation coordinates xp with len #xyh x #wd x #ws
            xp = [coords[n].repeat(L * K) for n in 'xyhi' if n in ip_dims]
            ip_data_dims = [n for n, l in [('i', ['x', 'y', 'h', 'i']), (l_name, ['wd']), ('ws', ['ws'])]
                            if any([l_ in ip_dims for l_ in l])]
            shape = [l for d, l in [('i', I), (l_name, L), ('ws', K)] if d in ip_data_dims]
            if 'wd' in ip_dims:
                xp.append(np.tile(coords['wd'].repeat(K), I))
            elif 'wd' in data_dims:
                shape.append(data.shape[data_dims.index('wd')])
            if 'ws' in ip_dims:
                xp.append(np.tile(coords['ws'], I * L))
            elif 'ws' in data_dims:
                shape.append(data.shape[data_dims.index('ws')])

            ip_data = grid_interp(np.array(xp).T, deg=deg)
            ip_data = ip_data.reshape(shape)
        else:
            ip_data = data
            ip_data_dims = []

        if i_indices is not None:
            ip_data_dims.append('i')
            ip_data = sel(ip_data, ip_data_dims, i_indices, 'i')
        if l_indices is not None:
            ip_data_dims.append(l_name)
            ip_data = sel(ip_data, ip_data_dims, l_indices, l_name)
        if k_indices is not None:
            ip_data_dims.append('ws')
            ip_data = sel(ip_data, ip_data_dims, k_indices, 'ws')

        for i, d in enumerate(['i', l_name, 'ws']):
            if d not in ip_data_dims:
                ip_data = np.expand_dims(ip_data, i)
        return ip_data


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


class plot_xy_map(_PlotMethods):
    def __init__(self, darray):
        _PlotMethods.__init__(self, darray)

    def __call__(self, **kwargs):
        if ('x' in self._da.coords and 'y' in self._da.coords and 'x' not in kwargs and

                self._da.squeeze().shape == (len(np.atleast_1d(self._da.x)), len(np.atleast_1d(self._da.y)))):
            kwargs['x'] = 'x'
        _PlotMethods(self._da)(**kwargs)


if not hasattr(xr.DataArray(None), 'ilk'):
    xr.register_dataarray_accessor("ilk")(ilk)
    xr.register_dataarray_accessor("interp_ilk")(interp_ilk)
    xr.register_dataarray_accessor("sel_interp_all")(sel_interp_all)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xr.register_dataarray_accessor('plot')(plot_xy_map)
