from py_wake.site._site import Site, UniformWeibullSite
import xarray as xr
from py_wake.site.distance import StraightDistance
import numpy as np


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
            ds = ds.reindex(wd=np.r_[ds.wd, 360])
            for n, v in ds.data_vars.items():
                if 'wd' in v.dims:
                    ds[n].loc[{'wd': 360}] = ds[n].sel(wd=0)
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
                return self.ds[n].sel_interp_all(lw.coords, method=self.interp_method)
            else:
                return default

        WS, WD, TI = [get(n, d) for n, d in [('WS', lw.ws), ('WD', lw.wd), ('TI', None)]]

        if 'Speedup' in self.ds:
            WS = self.ds.Speedup.sel_interp_all(lw.coords, method=self.interp_method) * WS

        if self.shear:
            assert 'h' in lw and np.all(lw.h.data != None), "Height must be specified and not None"  # nopep8
            h = np.unique(lw.h)
            if len(h) > 1:
                h = lw.h
            else:
                h = h[0]
            WS = self.shear(WS, lw.wd, h)

        if 'Turning' in self.ds:
            WD = (self.ds.Turning.sel_interp_all(lw.coords, method=self.interp_method) + WD) % 360

        lw.set_W(WS, WD, TI, ws_bins, self.use_WS_bins)
        if 'P' in self.ds:
            lw['P'] = self.ds.P.sel_interp_all(lw.coords, method=self.interp_method) / \
                self.ds.sector_width * lw.wd_bin_size
        else:
            sf = self.ds.Sector_frequency.sel_interp_all(lw.coords, method=self.interp_method)
            p_wd = sf / self.ds.sector_width * lw.wd_bin_size
            lw['P'] = p_wd * self.weibull_weight(lw,
                                                 self.ds.Weibull_A.sel_interp_all(lw.coords, method=self.interp_method),
                                                 self.ds.Weibull_k.sel_interp_all(lw.coords, method=self.interp_method))
        return lw
