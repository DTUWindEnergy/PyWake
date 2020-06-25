from py_wake.site._site import Site, UniformWeibullSite
import xarray as xr
from py_wake.site.distance import StraightDistance
import numpy as np
from py_wake.utils.eq_distance_interpolator import EqDistRegGridInterpolator
import warnings


class XRSite(UniformWeibullSite):
    use_WS_bins = False

    def __init__(self, ds, initial_position=None, interp_method='linear', shear=None, distance=StraightDistance()):
        self.interp_method = interp_method
        self.shear = shear
        self.distance = distance
        Site.__init__(self, distance)

        assert 'TI' in ds
        if 'wd' in ds and len(np.atleast_1d(ds.wd)) > 1:
            wd = ds.coords['wd']
            sector_widths = np.diff(wd)
            assert np.all(sector_widths == sector_widths[0]), \
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
        if 'wd' in ds and len(np.atleast_1d(ds.wd)) > 1 and ds.wd[-1] != 360 and 360 - ds.wd[-1] == sector_width:
            ds = xr.concat([ds, ds.sel(wd=0)], 'wd', data_vars='minimal')
            ds.update({'wd': np.r_[ds.wd[:-1], 360]})

        self.ds = ds

    @property
    def initial_position(self):
        return self.ds.initial_position

    def save(self, filename):
        self.ds.to_netcdf(filename)

    @staticmethod
    def load(filename, interp_method='nearest', shear=None, distance=StraightDistance()):
        ds = xr.load_dataset(filename)
        return XRSite(ds, interp_method=interp_method, shear=shear, distance=distance)

    @staticmethod
    def from_flow_box(flowBox, interp_method='linear', distance=StraightDistance()):
        ds = flowBox.drop_vars(['WS', 'TI']).rename_vars(WS_eff='WS', TI_eff='TI').squeeze()
        site = XRSite(ds, interp_method=interp_method, distance=distance)

        # Correct P from propability pr. deg to sector probability as expected by XRSite
        site.ds['P'] = site.ds.P * site.ds.sector_width
        return site

    def elevation(self, x_i, y_i):
        if 'Elevation' in self.ds:
            return self.ds.Elevation.interp(x=xr.DataArray(x_i, dims='z'), y=xr.DataArray(y_i, dims='z'),
                                            method=self.interp_method, kwargs={'bounds_error': True})
        else:
            return 0

    def interp(self, var, coords):

        sel_dims = []
        ip_dims = list(var.dims)
        if 'ws' in ip_dims and len(set(coords['ws'].data) - set(np.atleast_1d(var.ws.data))) == 0:
            # All ws is in var - no need to interpolate
            ip_dims.remove('ws')
            sel_dims.append('ws')
        if ip_dims and ip_dims[-1] == 'wd' in ip_dims and len(set(coords['wd'].data) - set(var.wd.data)) == 0:
            # All wd is in var - no need to interpolate
            ip_dims.remove('wd')
            sel_dims.append('wd')
        if 'i' in ip_dims and 'i' in coords and len(var.i) != len(coords['i']):
            raise ValueError(
                "Number of points, i(=%d), in site data variable, %s, must match number of requested points(=%d)" %
                (len(var.i), var.name, len(coords['i'])))
        if ip_dims and ip_dims[-1] == 'i':
            ip_dims.remove('i')
            sel_dims.append('i')

        if len(ip_dims) > 0:
            try:
                eq_interp = EqDistRegGridInterpolator([var.coords[k].data for k in ip_dims], var.data,
                                                      method=self.interp_method)
            except ValueError as e:
                warnings.warn("""The fast EqDistRegGridInterpolator fails (%s).
                Falling back on the slower xarray interp""" % e,
                              RuntimeWarning)

                return var.sel_interp_all(coords, method=self.interp_method)
            xp = np.array([coords[k].data for k in ip_dims]).T
            res = eq_interp(xp).T
        else:
            res = var.data

        if 'wd' in sel_dims:
            l_indices = np.searchsorted(var.wd.data, coords['wd'].data)
            if var.dims[-1] == 'wd':
                res = res[..., l_indices]
            else:
                res = res[..., l_indices, :]
        if 'ws' in sel_dims:
            k_indices = np.searchsorted(var.ws.data, coords['ws'].data)
            res = res[..., k_indices]

        ds = coords.to_dataset()
        ds[var.name] = ([d for d in ['i', 'wd', 'ws'] if d in var.dims or d.replace('i', 'x') in var.dims], res)
        return ds[var.name]

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

        WS, WD, TI = [get(n, d) for n, d in [('WS', lw.ws), ('WD', lw.wd), ('TI', None)]]

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
        if 'P' in self.ds:
            lw['P'] = self.interp(self.ds.P, lw.coords) / \
                self.ds.sector_width * lw.wd_bin_size
        else:
            sf = self.interp(self.ds.Sector_frequency, lw.coords)
            p_wd = sf / self.ds.sector_width * lw.wd_bin_size
            lw['P'] = p_wd * self.weibull_weight(lw,
                                                 self.interp(self.ds.Weibull_A, lw.coords),
                                                 self.interp(self.ds.Weibull_k, lw.coords))
        return lw
