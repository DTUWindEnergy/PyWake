import xarray as xr
from py_wake.site.distance import StraightDistance
import numpy as np
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.site._site import Site
from py_wake.utils import weibull


class XRSite(Site):
    use_WS_bins = False

    def __init__(self, ds, initial_position=None, interp_method='linear', shear=None, distance=StraightDistance(),
                 default_ws=np.arange(3, 26), bounds='check'):
        assert interp_method in [
            'linear', 'nearest'], 'interp_method "%s" not implemented. Must be "linear" or "nearest"' % interp_method
        assert bounds in ['check', 'limit', 'ignore'], 'bounds must be "check", "limit" or "ignore"'

        self.interp_method = interp_method
        self.shear = shear
        self.bounds = bounds

        Site.__init__(self, distance)
        self.default_ws = default_ws

        if 'ws' not in ds.dims:
            ds.update({'ws': self.default_ws})
        else:
            self.default_ws = ds.ws

        if 'wd' in ds and len(np.atleast_1d(ds.wd)) > 1:
            wd = ds.coords['wd']
            sector_widths = np.diff(wd)
            assert np.allclose(sector_widths, sector_widths[0]), \
                "all sectors must have same width"
            sector_width = sector_widths[0]
        else:
            sector_width = 360
        if 'P' not in ds:
            assert 'Weibull_A' in ds and 'Weibull_k' in ds and 'Sector_frequency' in ds
        ds.attrs['sector_width'] = sector_width

        if initial_position is not None:
            ds.attrs['initial_position'] = initial_position

        # add 360 deg to all wd dependent datavalues
        if 'wd' in ds and ds.wd[-1] != 360 and 360 - ds.wd[-1] == sector_width:
            ds = xr.concat([ds, ds.sel(wd=0)], 'wd', data_vars='minimal')
            ds.update({'wd': np.r_[ds.wd[:-1], 360]})

        self.ds = ds

    @property
    def initial_position(self):
        return self.ds.initial_position

    @initial_position.setter
    def initial_position(self, initial_position):
        self.ds.attrs['initial_position'] = initial_position

    def save(self, filename):
        self.ds.to_netcdf(filename)

    @staticmethod
    def load(filename, interp_method='nearest', shear=None, distance=StraightDistance()):
        ds = xr.load_dataset(filename)
        return XRSite(ds, interp_method=interp_method, shear=shear, distance=distance)

    @staticmethod
    def from_flow_box(flowBox, interp_method='linear', distance=StraightDistance()):
        ds = flowBox.drop_vars(['WS', 'TI']).rename_vars(WS_eff='WS', TI_eff='TI').squeeze()
        ds = ds.transpose(*[n for n in ['x', 'y', 'h', 'wd', 'ws'] if n in ds.dims])
        site = XRSite(ds, interp_method=interp_method, distance=distance)

        # Correct P from propability pr. deg to sector probability as expected by XRSite
        site.ds['P'] = site.ds.P * site.ds.sector_width
        return site

    def elevation(self, x_i, y_i):
        if 'Elevation' in self.ds:
            return self.ds.Elevation.interp(x=xr.DataArray(x_i, dims='z'), y=xr.DataArray(y_i, dims='z'),
                                            method=self.interp_method, kwargs={'bounds_error': True})
        else:
            return x_i * 0

    def interp(self, var, coords):

        # Interpolate via EqDistRegGridInterpolator (equidistance regular grid interpolator) which is much faster
        # than xarray.interp.
        # This function is comprehensive because var can contain any combinations of coordinates (i or (xy,h)) and wd,ws

        def sel(data, data_dims, indices, dim_name):
            i = data_dims.index(dim_name)
            ix = tuple([(slice(None), indices)[dim == i] for dim in range(data.ndim)])
            return data[ix]

        ip_dims = [n for n in ['i', 'x', 'y', 'h', 'wd', 'ws'] if n in var.dims]  # interpolation dimensions
        data = var.data
        data_dims = var.dims

        def pre_sel(data, name):
            # If only a single value is needed on the <name>-dimension, the data is squeezed to contain this value only
            # Otherwise the indices of the needed values are returned
            if name not in var.dims:
                return data, None
            c, v = coords[name].data, var[name].data
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
        data, l_indices = pre_sel(data, 'wd')

        if 'i' in ip_dims:
            if 'i' in coords and len(var.i) != len(coords['i']):
                raise ValueError(
                    "Number of points, i(=%d), in site data variable, %s, must match number of requested points(=%d)" %
                    (len(var.i), var.name, len(coords['i'])))
            # requesting all points(wt positions) in site
            # ip_dims.remove('i')
            # ip_data_dims = ['i']

        if len(ip_dims) > 0:
            grid_interp = GridInterpolator([var.coords[k].data for k in ip_dims], data,
                                           method=self.interp_method, bounds=self.bounds)

            # get dimension of interpolation coordinates
            I = (1, len(coords.get('x', coords.get('y', coords.get('h', coords.get('i', [None]))))))[
                any([n in data_dims for n in 'xyhi'])]
            L, K = [(1, len(coords.get(n, [None])))[indices is None and n in data_dims]
                    for n, indices in [('wd', l_indices), ('ws', k_indices)]]

            # gather interpolation coordinates xp with len #xyh x #wd x #ws
            xp = [coords[n].data.repeat(L * K) for n in 'xyhi' if n in ip_dims]
            ip_data_dims = [n for n, l in [('i', ['x', 'y', 'h', 'i']), ('wd', ['wd']), ('ws', ['ws'])]
                            if any([l_ in ip_dims for l_ in l])]
            shape = [l for d, l in [('i', I), ('wd', L), ('ws', K)] if d in ip_data_dims]
            if 'wd' in ip_dims:
                xp.append(np.tile(coords['wd'].data.repeat(K), I))
            elif 'wd' in data_dims:
                shape.append(data.shape[data_dims.index('wd')])
            if 'ws' in ip_dims:
                xp.append(np.tile(coords['ws'].data, I * L))
            elif 'ws' in data_dims:
                shape.append(data.shape[data_dims.index('ws')])

            ip_data = grid_interp(np.array(xp).T)
            ip_data = ip_data.reshape(shape)
        else:
            ip_data = data
            ip_data_dims = []

#         if 'i' in var.dims:
#             ip_data_dims.insert(0, 'i')
        if l_indices is not None:
            ip_data_dims.append('wd')
            ip_data = sel(ip_data, ip_data_dims, l_indices, 'wd')
        if k_indices is not None:
            ip_data_dims.append('ws')
            ip_data = sel(ip_data, ip_data_dims, k_indices, 'ws')

        ds = coords.to_dataset()
        if ip_data_dims:
            ds[var.name] = (ip_data_dims, ip_data)
        else:
            ds[var.name] = ip_data
        return ds[var.name]

    def weibull_weight(self, localWind, A, k):

        P = weibull.cdf(localWind.ws_upper, A=A, k=k) - weibull.cdf(localWind.ws_lower, A=A, k=k)
        P.attrs['Description'] = "Probability of wind flow case (i.e. wind direction and wind speed)"
        return P

    def _local_wind(self, localWind, ws_bins=None):
        """
        Returns
        -------
        LocalWind object containing:
            WD : array_like
                local free flow wind directions
            WS : array_like
                local free flow wind speeds
            TI : array_like
                local free flow turbulence intensity
            P : array_like
                Probability/weight
        """
        lw = localWind

        def get(n, default=None):
            if n in self.ds:
                return self.interp(self.ds[n], lw.coords)
            else:
                return default

        WS, WD, TI, TI_std = [get(n, d) for n, d in [('WS', lw.ws), ('WD', lw.wd), ('TI', None), ('TI_std', None)]]

        if 'Speedup' in self.ds:
            WS = self.interp(self.ds.Speedup, lw.coords) * WS

        if self.shear:
            assert 'h' in lw and np.all(lw.h.data != None), "Height must be specified and not None"  # nopep8
            h = np.unique(lw.h)
            if len(h) > 1:
                h = lw.h
            else:
                h = h[0]
            WS = self.shear(WS, lw.wd, h)

        if 'Turning' in self.ds:
            WD = (self.interp(self.ds.Turning, lw.coords) + WD) % 360

        lw.set_W(WS, WD, TI, ws_bins, self.use_WS_bins)
        lw.set_data_array(TI_std, 'TI_std', 'Standard deviation of turbulence intensity')

        if 'P' in self.ds:
            if ('ws' in self.ds.P.dims and 'ws' in lw.coords):
                d_ws = self.ds.P.ws.values
                c_ws = lw.coords['ws'].values
                i = np.searchsorted(d_ws, c_ws[0])
                if (np.any([ws not in d_ws for ws in c_ws]) or  # check all coordinate ws in data ws
                    len(d_ws[i:i + len(c_ws)]) != len(c_ws) or  # check subset has same length
                        np.any(d_ws[i:i + len(c_ws)] != c_ws)):  # check subset are equal
                    raise ValueError("Cannot interpolate ws-dependent P to other range of ws")
            lw['P'] = self.interp(self.ds.P, lw.coords) / \
                self.ds.sector_width * lw.wd_bin_size
        else:
            sf = self.interp(self.ds.Sector_frequency, lw.coords)
            p_wd = sf / self.ds.sector_width * lw.wd_bin_size
            A, k = self.interp(self.ds.Weibull_A, lw.coords), self.interp(self.ds.Weibull_k, lw.coords)
            lw['Weibull_A'] = A
            lw['Weibull_k'] = k
            lw['Sector_frequency'] = p_wd
            lw['P'] = p_wd * self.weibull_weight(lw, A, k)
        return lw


class UniformSite(XRSite):
    """Site with uniform (same wind over all, i.e. flat uniform terrain) and
    constant wind speed probability of 1. Only for one fixed wind speed
    """

    def __init__(self, p_wd, ti, ws=12, interp_method='nearest', shear=None, initial_position=None):
        ds = xr.Dataset(
            data_vars={'P': ('wd', p_wd), 'TI': ti},
            coords={'wd': np.linspace(0, 360, len(p_wd), endpoint=False)})

        XRSite.__init__(self, ds, interp_method=interp_method, shear=shear, initial_position=initial_position,
                        default_ws=np.atleast_1d(ws))


class UniformWeibullSite(XRSite):
    """Site with uniform (same wind over all, i.e. flat uniform terrain) and
    weibull distributed wind speed
    """

    def __init__(self, p_wd, a, k, ti, interp_method='nearest', shear=None):
        """Initialize UniformWeibullSite

        Parameters
        ----------
        p_wd : array_like
            Probability of wind direction sectors
        a : array_like
            Weilbull scaling parameter of wind direction sectors
        k : array_like
            Weibull shape parameter
        ti : float or array_like
            Turbulence intensity
        interp_method : 'nearest', 'linear'
            p_wd, a, k, ti and alpha are interpolated to 1 deg sectors using this
            method
        shear : Shear object
            Shear object, e.g. NoShear(), PowerShear(h_ref, alpha)

        Notes
        ------
        The wind direction sectors will be: [0 +/- w/2, w +/- w/2, ...]
        where w is 360 / len(p_wd)

        """
        ds = xr.Dataset(
            data_vars={'Sector_frequency': ('wd', p_wd), 'Weibull_A': ('wd', a), 'Weibull_k': ('wd', k), 'TI': ti},
            coords={'wd': np.linspace(0, 360, len(p_wd), endpoint=False)})
        XRSite.__init__(self, ds, interp_method=interp_method, shear=shear)
