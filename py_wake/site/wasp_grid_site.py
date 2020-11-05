from py_wake.site._site import UniformWeibullSite
import numpy as np
import xarray as xr
import pickle
import os
import glob
from _collections import defaultdict
import re
from scipy.interpolate.interpolate import RegularGridInterpolator
import copy
from numpy import newaxis as na
from py_wake.site.distance import TerrainFollowingDistance
from builtins import AttributeError


class WaspGridSite(UniformWeibullSite):
    """Site with non-uniform (different wind at different locations, e.g. complex non-flat terrain)
    weibull distributed wind speed. Data obtained from WAsP grid files"""

    def __init__(self, ds, distance=TerrainFollowingDistance(), mode='valid'):
        """
        Parameters
        ----------
        ds : xarray
            dataset as returned by load_wasp_grd
        distance : Distance object, optional
            Alternatives are StraightDistance and TerrainFollowingDistance2
        mode : {'valid', 'extrapolate'}, optional
            if valid, terrain elevation outside grid area is NAN
            if extrapolate, the terrain elevation at the grid border is returned outside the grid area
        """
        self._ds = ds
        self.mode = mode
        self.interp_funcs = {}
        self.interp_funcs_initialization()
        self.elevation_interpolator = EqDistRegGrid2DInterpolator(self._ds.coords['x'].data,
                                                                  self._ds.coords['y'].data,
                                                                  self._ds['elev'].data)
        self.TI_data_exist = 'tke' in self.interp_funcs.keys()
        super().__init__(p_wd=np.nanmean(self._ds['f'].data, (0, 1, 2)),
                         a=np.nanmean(self._ds['A'].data, (0, 1, 2)),
                         k=np.nanmean(self._ds['k'].data, (0, 1, 2)),
                         ti=0)
        self.distance = distance

    def _local_wind(self, localWind, ws_bins=None):

        lw = localWind

        # TODO: delete?
        self.wd = lw.wd

        WS = xr.DataArray(lw.ws, coords=[('ws', lw.ws)])
        WD = xr.DataArray(lw.wd, coords=[('wd', lw.wd)])

        if self.TI_data_exist:
            speed_up, turning, TI = [xr.DataArray(v, coords=[('i', lw.i), ('wd', lw.wd)])
                                     for v in self.interpolate(['spd', 'orog_trn', 'tke'], lw.x, lw.y, lw.h, lw.wd, lw.ws)]
            TI = TI * (.75 + 3.8 / WS)
        else:  # pragma: no cover
            raise Exception("Not tested")
            speed_up, turning = self.interpolate(['spd', 'orog_trn'], lw.x, lw.y, lw.h, lw.wd, lw.ws)

        WS = speed_up * WS
        turning = turning.fillna(0)
        lw.set_W(WS, WD, TI, ws_bins, use_WS=True)
        lw['P'] = self.probability(lw)

        lw['WD'] = turning + WD

        return lw

    def probability(self, localWind):
        """See Site.probability
        """
        lw = localWind
        Weibull_A, Weibull_k, freq = \
            [xr.DataArray(v, coords=[('i', lw.i), ('wd', lw.wd)])
             for v in self.interpolate(['A', 'k', 'f'], lw.x, lw.y, lw.h, lw.wd)]

        P_wd = freq * lw.wd_bin_size

        P = self.weibull_weight(lw, Weibull_A, Weibull_k) * P_wd
        return P

    def elevation(self, x_i, y_i):
        return self.elevation_interpolator(x_i, y_i, self.mode)

    def interp_funcs_initialization(self,
                                    interp_keys=['A', 'k', 'f', 'tke', 'spd', 'orog_trn', 'flow_inc', 'elev']):
        """ Initialize interpolating functions using RegularGridInterpolator
        for specified variables defined in interp_keys.
        """

        interp_funcs = {}

        for key in interp_keys:
            try:
                dr = self._ds.data_vars[key]
            except KeyError:
                print('Warning! {0} is not included in the current'.format(key) +
                      ' WindResourceGrid object.\n')
                continue

            coords = []

            data = dr.data
            for dim in dr.dims:
                # change sector index into wind direction (deg)
                if dim == 'sec':
                    num_sec = len(dr.coords[dim].data)   # number of sectors
                    coords.append(
                        np.linspace(0, 360, num_sec + 1))
                    data = np.concatenate((data, data[:, :, :, :1]), axis=3)
                elif dim == 'z' and len(dr.coords[dim].data) == 1:
                    # special treatment for only one layer of data
                    height = dr.coords[dim].data[0]
                    coords.append(np.array([height - 1.0, height + 1.0]))
                    data = np.concatenate((data, data), axis=2)
                else:
                    coords.append(dr.coords[dim].data)

            interp_funcs[key] = RegularGridInterpolator(
                coords,
                data, bounds_error=False)

        self.interp_funcs.update(interp_funcs)

    def interpolate(self, keys, x_i, y_i, h_i, wd=None, ws=None):
        if wd is None:
            wd = self.default_wd
        if ws is None:
            ws = self.default_ws
        wd, ws = np.asarray(wd), np.asarray(ws)
        x_il, y_il, h_il = [np.repeat([np.asarray(v)], len(wd), 0).T for v in [x_i, y_i, h_i]]
        wd_il = np.repeat([wd], len(np.atleast_1d(x_i)), 0)
        return [self.interp_funcs[n]((x_il, y_il, h_il, wd_il)) for n in np.asarray(keys)]

    def plot_map(self, data_name, height=None, sector=None, xlim=[None, None], ylim=[None, None], ax=None):
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        if data_name not in self._ds:
            raise AttributeError("%s not found in dataset. Available data variables are:\n%s" %
                                 (data_name, ",".join(list(self._ds.data_vars.keys()))))
        data = self._ds[data_name].sel(x=slice(*xlim), y=slice(*ylim))
        if 'sec' in data.coords:
            if sector not in data.coords['sec'].values:
                raise AttributeError("Sector %s not found. Available sectors are: %s" %
                                     (sector, data.coords['sec'].values))
            data = data.sel(sec=sector)
        if 'z' in data.coords:
            if height is None:
                raise AttributeError("Height missing for '%s'" % data_name)
            data = data.interp(z=height)
        data.plot(x='x', y='y', ax=ax)
        ax.axis('equal')

    @classmethod
    def from_wasp_grd(cls, path, globstr='*.grd', speedup_using_pickle=True,
                      distance=TerrainFollowingDistance(), mode='valid'):
        ds = load_wasp_grd(path, globstr, speedup_using_pickle)
        return WaspGridSite(ds, distance, mode)


def load_wasp_grd(path, globstr='*.grd', speedup_using_pickle=True):
    '''
    Reader for WAsP .grd resource grid files.

    Parameters
    ----------
    path: str
        path to file or directory containing goldwind excel files

    globstr: str
        string that is used to glob files if path is a directory.

    Returns
    -------
    WindResourceGrid: :any:`WindResourceGrid`:

    Examples
    --------
    >>> from mowflot.wind_resource import WindResourceGrid
    >>> path = '../mowflot/tests/data/WAsP_grd/'
    >>> wrg = WindResourceGrid.from_wasp_grd(path)
    >>> print(wrg)
        <xarray.Dataset>
        Dimensions:            (sec: 12, x: 20, y: 20, z: 3)
        Coordinates:
          * sec                (sec) int64 1 2 3 4 5 6 7 8 9 10 11 12
          * x                  (x) float64 5.347e+05 5.348e+05 5.349e+05 5.35e+05 ...
          * y                  (y) float64 6.149e+06 6.149e+06 6.149e+06 6.149e+06 ...
          * z                  (z) float64 10.0 40.0 80.0
        Data variables:
            flow_inc           (x, y, z, sec) float64 1.701e+38 1.701e+38 1.701e+38 ...
            ws_mean            (x, y, z, sec) float64 3.824 3.489 5.137 5.287 5.271 ...
            meso_rgh           (x, y, z, sec) float64 0.06429 0.03008 0.003926 ...
            obst_spd           (x, y, z, sec) float64 1.701e+38 1.701e+38 1.701e+38 ...
            orog_spd           (x, y, z, sec) float64 1.035 1.039 1.049 1.069 1.078 ...
            orog_trn           (x, y, z, sec) float64 -0.1285 0.6421 0.7579 0.5855 ...
            power_density      (x, y, z, sec) float64 77.98 76.96 193.5 201.5 183.9 ...
            rix                (x, y, z, sec) float64 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
            rgh_change         (x, y, z, sec) float64 6.0 10.0 10.0 10.0 6.0 4.0 0.0 ...
            rgh_spd            (x, y, z, sec) float64 1.008 0.9452 0.8578 0.9037 ...
            f                  (x, y, z, sec) float64 0.04021 0.04215 0.06284 ...
            tke                (x, y, z, sec) float64 1.701e+38 1.701e+38 1.701e+38 ...
            A                  (x, y, z, sec) float64 4.287 3.837 5.752 5.934 5.937 ...
            k                  (x, y, z, sec) float64 1.709 1.42 1.678 1.74 1.869 ...
            flow_inc_tot       (x, y, z) float64 1.701e+38 1.701e+38 1.701e+38 ...
            ws_mean_tot        (x, y, z) float64 5.16 6.876 7.788 5.069 6.85 7.785 ...
            power_density_tot  (x, y, z) float64 189.5 408.1 547.8 178.7 402.2 546.6 ...
            rix_tot            (x, y, z) float64 0.0 0.0 0.0 9.904e-05 9.904e-05 ...
            tke_tot            (x, y, z) float64 1.701e+38 1.701e+38 1.701e+38 ...
            A_tot              (x, y, z) float64 5.788 7.745 8.789 5.688 7.716 8.786 ...
            k_tot              (x, y, z) float64 1.725 1.869 2.018 1.732 1.877 2.018 ...
            elev               (x, y) float64 37.81 37.42 37.99 37.75 37.46 37.06 ...

    '''

    var_name_dict = {
        'Flow inclination': 'flow_inc',
        'Mean speed': 'ws_mean',
        'Meso roughness': 'meso_rgh',
        'Obstacles speed': 'obst_spd',
        'Orographic speed': 'orog_spd',
        'Orographic turn': 'orog_trn',
        'Power density': 'power_density',
        'RIX': 'rix',
        'Roughness changes': 'rgh_change',
        'Roughness speed': 'rgh_spd',
        'Sector frequency': 'f',
        'Turbulence intensity': 'tke',
        'Weibull-A': 'A',
        'Weibull-k': 'k',
        'Elevation': 'elev',
        'AEP': 'aep'}

    def _read_grd(filename):

        def _parse_line_floats(f):
            return [float(i) for i in f.readline().strip().split()]

        def _parse_line_ints(f):
            return [int(i) for i in f.readline().strip().split()]

        with open(filename, 'rb') as f:
            file_id = f.readline().strip().decode()
            nx, ny = _parse_line_ints(f)
            xl, xu = _parse_line_floats(f)
            yl, yu = _parse_line_floats(f)
            zl, zu = _parse_line_floats(f)
            values = np.array([l.split() for l in f.readlines() if l.strip() != b""],
                              dtype=np.float)  # around 8 times faster

        xarr = np.linspace(xl, xu, nx)
        yarr = np.linspace(yl, yu, ny)

        # note that the indexing of WAsP grd file is 'xy' type, i.e.,
        # values.shape == (xarr.shape[0], yarr.shape[0])
        # we need to transpose values to match the 'ij' indexing
        values = values.T
        #############
        # note WAsP denotes unavailable values using very large numbers, here
        # we change them into np.nan, to avoid strange results.
        values[values > 1e20] = np.nan

        return xarr, yarr, values

    if speedup_using_pickle:
        if os.path.isdir(path):
            pkl_fn = os.path.join(path, os.path.split(os.path.dirname(path))[1] + '.pkl')
            if os.path.isfile(pkl_fn):
                with open(pkl_fn, 'rb') as f:
                    return pickle.load(f)
            else:
                ds = load_wasp_grd(path, globstr, speedup_using_pickle=False)
                with open(pkl_fn, 'wb') as f:
                    pickle.dump(ds, f, protocol=-1)
                return ds
        else:
            raise NotImplementedError

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, globstr)))
    else:
        raise Exception('Path was not a directory...')

    file_height_dict = defaultdict(list)

    pattern = re.compile(r'Sector (\w+|\d+) \s+ Height (\d+\.?\d*)m \s+ ([a-zA-Z0-9- ]+)')
    for f in files:
        sector, height, var_name = re.findall(pattern, f)[0]
        # print(sector, height, var_name)
        name = var_name_dict.get(var_name, var_name)
        file_height_dict[height].append((f, sector, name))

    elev_avail = False
    first = True
    for height, files_subset in file_height_dict.items():

        first_at_height = True
        for file, sector, var_name in files_subset:
            xarr, yarr, values = _read_grd(file)

            if sector == 'All':

                # Only 'All' sector has the elevation files.
                # So here we make sure that, when the elevation file
                # is read, it gets the right (x,y) coords/dims.
                if var_name == 'elev':
                    elev_avail = True
                    elev_vals = values
                    elev_coords = {'x': xarr,
                                   'y': yarr}
                    elev_dims = ('x', 'y')
                    continue

                else:
                    var_name += '_tot'

                    coords = {'x': xarr,
                              'y': yarr,
                              'z': np.array([float(height)])}

                    dims = ('x', 'y', 'z')

                    da = xr.DataArray(values[..., np.newaxis],
                                      coords=coords,
                                      dims=dims)

            else:

                coords = {'x': xarr,
                          'y': yarr,
                          'z': np.array([float(height)]),
                          'sec': np.array([int(sector)])}

                dims = ('x', 'y', 'z', 'sec')

                da = xr.DataArray(values[..., np.newaxis, np.newaxis],
                                  coords=coords,
                                  dims=dims)

            if first_at_height:
                ds_tmp = xr.Dataset({var_name: da})
                first_at_height = False
            else:
                ds_tmp = xr.merge([ds_tmp, xr.Dataset({var_name: da})])

        if first:
            ds = ds_tmp
            first = False
        else:
            ds = xr.concat([ds, ds_tmp], dim='z')

    if elev_avail:
        ds['elev'] = xr.DataArray(elev_vals,
                                  coords=elev_coords,
                                  dims=elev_dims)
    ############
    # Calculate the compund speed-up factor based on orog_spd, rgh_spd
    # and obst_spd
    spd = 1
    for dr in ds.data_vars:
        if dr in ['orog_spd', 'obst_spd', 'rgh_spd']:
            # spd *= np.where(ds.data_vars[dr].data > 1e20, 1, ds.data_vars[dr].data)
            spd *= np.where(np.isnan(ds.data_vars[dr].data), 1, ds.data_vars[dr].data)

    ds['spd'] = copy.deepcopy(ds['orog_spd'])
    ds['spd'].data = spd

    #############
    # change the frequency from per sector to per deg
    ds['f'].data = ds['f'].data * len(ds['f']['sec'].data) / 360.0

#         #############
#         # note WAsP denotes unavailable values using very large numbers, here
#         # we change them into np.nan, to avoid strange results.
#         for var in ds.data_vars:
#             ds[var].data = np.where(ds[var].data > 1e20, np.nan, ds[var].data)

    # make sure coords along z is asending order
    ds = ds.loc[{'z': sorted(ds.coords['z'].values)}]

    ######################################################################

    if 'tke' in ds and np.mean(ds['tke']) > 1:
        ds['tke'] *= 0.01

    return ds


class EqDistRegGrid2DInterpolator():
    def __init__(self, x, y, Z):
        self.x = x
        self.y = y
        self.Z = Z
        self.dx, self.dy = [xy[1] - xy[0] for xy in [x, y]]
        self.x0 = x[0]
        self.y0 = y[0]
        xi_valid = np.where(np.any(~np.isnan(self.Z), 1))[0]
        yi_valid = np.where(np.any(~np.isnan(self.Z), 0))[0]
        self.xi_valid_min, self.xi_valid_max = xi_valid[0], xi_valid[-1]
        self.yi_valid_min, self.yi_valid_max = yi_valid[0], yi_valid[-1]

    def __call__(self, x, y, mode='valid'):
        xp, yp = x, y
        xi = (xp - self.x0) / self.dx
        xif, xi0 = np.modf(xi)
        xi0 = xi0.astype(np.int)

        yi = (yp - self.y0) / self.dy
        yif, yi0 = np.modf(yi)
        yi0 = yi0.astype(np.int)
        if mode == 'extrapolate':
            xif[xi0 < self.xi_valid_min] = 0
            xif[xi0 > self.xi_valid_max - 2] = 1
            yif[yi0 < self.yi_valid_min] = 0
            yif[yi0 > self.yi_valid_max - 2] = 1
            xi0 = np.minimum(np.maximum(xi0, self.xi_valid_min), self.xi_valid_max - 2)
            yi0 = np.minimum(np.maximum(yi0, self.yi_valid_min), self.yi_valid_max - 2)
        xi1 = xi0 + 1
        yi1 = yi0 + 1
        valid = (xif >= 0) & (yif >= 0) & (xi1 < len(self.x)) & (yi1 < len(self.y))
        z = np.empty_like(xp) + np.nan
        xi0, xi1, xif, yi0, yi1, yif = [v[valid] for v in [xi0, xi1, xif, yi0, yi1, yif]]
        z00 = self.Z[xi0, yi0]
        z10 = self.Z[xi1, yi0]
        z01 = self.Z[xi0, yi1]
        z11 = self.Z[xi1, yi1]
        z0 = z00 + (z10 - z00) * xif
        z1 = z01 + (z11 - z01) * xif
        z[valid] = z0 + (z1 - z0) * yif
        return z


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path
        import matplotlib.pyplot as plt
        site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=False)
        x, y = site._ds.coords['x'].data, site._ds.coords['y'].data,
        Y, X = np.meshgrid(y, x)

        # plot elevation
        Z = site.elevation(X.flatten(), Y.flatten()).reshape(X.shape)
        ax1, ax2 = plt.subplots(1, 2)[1]
        c = ax1.contourf(X, Y, Z, 100)
        plt.colorbar(c)
        i, j = 15, 12
        ax1.plot(X[:, j], Y[:, j], 'r')
        ax1.axis('equal')
        Z = site.elevation_interpolator(X[:, j], Y[:, j], mode='extrapolate')
        ax2.plot(X[:, j], Z, 'r')

        # plot wind speed
        Z = site.local_wind(X.flatten(), Y.flatten(), h_i=70, ws=[10], wd=[0]).WS_ilk.reshape(X.shape)
        ax1, ax2 = plt.subplots(1, 2)[1]
        c = ax1.contourf(X, Y, Z, 100)
        plt.colorbar(c)
        i, j = 15, 12
        ax1.plot(X[:, j], Y[:, j], 'r')
        ax1.axis('equal')

        ax2.plot(X[:, j], Z[:, j], 'r')

        z = np.arange(35, 200, 1)
        u_z = site.local_wind([x[i]] * len(z), y_i=[y[i]] * len(z),
                              h_i=z, wd=[0], ws=[10]).WS_ilk[:, 0, 0]
        Z = site.elevation_interpolator(X[i], Y[i], mode='extrapolate')

        site.local_wind([263655.0], [6506601.0], h_i=70, wd=[0]).WS_ilk[0, 0]
        site.local_wind([263655.0], [6506601.0], h_i=70, ws=[10]).WD_ilk[0, :, 0]

        plt.figure()
        plt.plot(u_z, z)
        plt.show()


main()
