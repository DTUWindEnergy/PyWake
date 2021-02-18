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
        lw['TI'] = self.interp(self.ds.tke, lw.coords) * (.75 + 3.8 / lw.ws)
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
    # ds['f'].data = ds['f'].data * len(ds['f']['sec'].data) / 360.0

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
