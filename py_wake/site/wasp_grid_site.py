import numpy as np
import xarray as xr
import pickle
import os
import glob
from _collections import defaultdict
import re
import copy
from py_wake.site.distance import TerrainFollowingDistance
from py_wake.site.xrsite import XRSite


class WaspGridSite(XRSite):
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
        self.use_WS_bins = True
        ds = ds.rename(A="Weibull_A", k="Weibull_k", f="Sector_frequency", spd='Speedup',
                       orog_trn='Turning',
                       elev='Elevation', sec='wd', z='h')
        ds = ds.assign_coords(wd=(ds.wd - 1) * (360 / len(ds.wd)))
        ds = ds.isel(x=np.where(~np.all(np.isnan(ds.Elevation), 1))[0],
                     y=np.where(~np.all(np.isnan(ds.Elevation), 0))[0])
        super().__init__(ds, distance=distance)

    def _local_wind(self, localWind, ws_bins=None):
        lw = super()._local_wind(localWind.copy(), ws_bins)

        # ti is assumed to be the turbulence intensity given by CFD
        # (expected value of TI at 15m/s). The Normal Turbulence model
        # is used to calculate TI at different wind speed,
        # see footnote 4 at page 24 of IEC 61400-1 (2005)
        lw['TI'] = self.interp(self.ds.ti15ms, lw.coords) * (.75 + 3.8 / lw.ws)
        return lw

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
        'Turbulence intensity': 'ti15ms',
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
                              dtype=float)  # around 8 times faster

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
                try:
                    with open(pkl_fn, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    print('loading %s failed. Loading from grid files instead' % pkl_fn)

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
    # ds['f'].data = ds['f'].data * len(ds['f']['sec'].data) / 360.0

#         #############
#         # note WAsP denotes unavailable values using very large numbers, here
#         # we change them into np.nan, to avoid strange results.
#         for var in ds.data_vars:
#             ds[var].data = np.where(ds[var].data > 1e20, np.nan, ds[var].data)

    # make sure coords along z is asending order
    ds = ds.loc[{'z': sorted(ds.coords['z'].values)}]

    ######################################################################

    if 'ti15ms' in ds and np.mean(ds['ti15ms']) > 1:
        ds['ti15ms'] *= 0.01

    return ds


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path
        import matplotlib.pyplot as plt
        site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=False)
        x, y = site.ds.x.values, site.ds.y.values
        Y, X = np.meshgrid(y, x)

        # plot elevation
        ax1, ax2 = plt.subplots(1, 2)[1]
        site.ds.Elevation.plot(ax=ax1, levels=100)
        i, j = 15, 12
        ax1.plot(X[:, j], Y[:, j], 'r')
        ax1.axis('equal')
        Z = site.elevation_interpolator(X[:, j], Y[:, j], mode='extrapolate')
        ax2.plot(X[:, j], Z, 'r')

        # plot wind speed
        ax1, ax2 = plt.subplots(1, 2)[1]
        WS = (site.ds.ws * site.ds.Speedup).sel(ws=10, wd=0).interp(h=70)
        WS.plot(ax=ax1, levels=100)
        i, j = 15, 12
        ax1.plot(X[:, j], Y[:, j], 'r')
        ax1.axis('equal')
        ax2.plot(X[:, j], WS[:, j], 'r')

        # plot shear
        z = np.arange(35, 200, 1)
        u_z = site.local_wind([x[i]] * len(z), y_i=[y[i]] * len(z), h_i=z, wd=[0], ws=[10]).WS_ilk[:, 0, 0]
        plt.figure()
        plt.plot(u_z, z)
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Height [m]')
        plt.show()


main()
