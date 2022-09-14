import os
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel
from py_wake import np
from numpy import newaxis as na
import xarray as xr
from tqdm import tqdm
from py_wake.utils.grid_interpolator import GridInterpolator


class GaussianOverlapAvgModel(RotorAvgModel):
    def __init__(self, filename=os.path.dirname(__file__) + f'/gaussian_overlap_.02_.02_128_512.nc'):
        table = xr.load_dataarray(filename)
        R_sigma = np.arange(0, 20.001, 0.01)
        CW_sigma = np.arange(0, 10.01, 0.01)
        dat = table.interp(R_sigma=R_sigma, CW_sigma=CW_sigma, method='cubic')
        self.overlap_interpolator = GridInterpolator([R_sigma, CW_sigma], dat, bounds='limit')

    def _calc_layout_terms(self, func, cw_ijlk, **kwargs):
        func(cw_ijlk=cw_ijlk * 0, **kwargs)

    def __call__(self, func, cw_ijlk, D_dst_ijl, **kwargs):
        res_ijlk = func(cw_ijlk=cw_ijlk * 0, D_dst_ijl=D_dst_ijl, **kwargs)
        if not hasattr(func.__self__, 'sigma_ijlk'):
            raise AttributeError(
                f"'{func.__self__.__class__.__name__}' has no attribute 'sigma_ijlk', which is needed by the GaussianOverlapAvgModel")
        sigma_ijlk = func.__self__.sigma_ijlk(**kwargs)
        overlap_factor_ijlk = self.overlap_interpolator(
            np.array([((D_dst_ijl / 2)[..., na] / sigma_ijlk).flatten(), (cw_ijlk / sigma_ijlk).flatten()]).T).reshape(sigma_ijlk.shape)
        return res_ijlk * overlap_factor_ijlk


def make_lookup_table(n_theta, n_r, dr=.5, dcw=.5):  # pragma: no cover
    # Downstream rotor radius normalized with characteristic wake width, sigma
    R_sigma = np.arange(0, 20 + dr, dr)
    # Crosswind distance normalized with characteristic wake width, sigma
    cw_sigma = np.arange(0, 10 + dcw, dcw)[:, na, na]
    theta = np.linspace(0, 2 * np.pi, n_theta)[na, :, na]  # Azimuthal discretization of downstream rotor
    r_lst = np.linspace(0, 1, n_r)[na, na, :]

    def cw_table(R_sigma):
        if R_sigma > 0:
            r = r_lst * R_sigma  # Radial discretization of downstream rotor
            dat = np.exp(- 1 / 2 * (r ** 2 + cw_sigma**2 - 2 * r * cw_sigma * np.cos(theta))) * r
            dtheta = np.diff(theta.flatten()[:2])
            dr = np.diff(r.flatten()[:2])
            return np.trapz(np.trapz(dat, dx=dtheta, axis=1), dx=dr, axis=1)
        else:
            return np.exp(-(cw_sigma[:, 0, 0]**2) / 2)  # use point value
    dat = np.array([cw_table(R) for R in tqdm(R_sigma)])
    m = R_sigma > 0
    dat[m] = dat[m] / (np.pi * R_sigma[m, na]**2)
    return xr.DataArray(dat, dims=('R_sigma', 'CW_sigma'), coords={'R_sigma': R_sigma, 'CW_sigma': cw_sigma[:, 0, 0]})


def find_n_r(n_theta, max_err):  # pragma: no cover
    res = make_lookup_table(n_theta, 2)
    for n in 2**np.arange(2, 12):
        ref = make_lookup_table(n_theta, n)
        print(n, np.abs(ref - res).max())
        if np.abs(ref - res).max() < max_err:
            return n
        res = ref
    raise ValueError("Error limit not reached")


def find_n_theta(n_r, max_err):  # pragma: no cover
    res = make_lookup_table(2, n_r)
    for n in 2**np.arange(2, 12):
        ref = make_lookup_table(n, n_r)
        print(n, np.abs(ref - res).max())
        if np.abs(ref - res).max() < max_err:
            return n
        res = ref
    raise ValueError("Error limit not reached")


def compare2orsted(table):  # pragma: no cover
    import scipy.io
    mat = scipy.io.loadmat(r'C:\tmp\TurbOPark/gauss_lookup_table')
    dist, radius_down, overlap = mat['overlap_lookup_table'][0][0]
    ref = xr.DataArray(overlap, dims=('CW_sigma', 'R_sigma'), coords={'R_sigma': radius_down[0], 'CW_sigma': dist[0]})
    # ref.to_netcdf(os.path.dirname(__file__) + f'/gaussian_overlap_orsted.nc')

    interp = GridInterpolator([table.R_sigma.values, table.CW_sigma.values], table.values)

    X, Y = np.meshgrid(ref.R_sigma, ref.CW_sigma)
    err = ref - interp(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)
    err = ref - table.interp(R_sigma=ref.R_sigma, CW_sigma=ref.CW_sigma, method='cubic')
    print(np.abs(err).max())
    (err).plot()
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':  # pragma: no cover
    print(find_n_r(256, 1e-6))
    n_r = 128
    n_r = 512
    print(find_n_theta(512, 1e-6))
    n_theta = 64
    n_theta = 128

    #
    table = make_lookup_table(n_theta, n_r, .02, .02)
    table.to_netcdf(os.path.dirname(__file__) + f'/gaussian_overlap_.02_.02_{n_theta}_{n_r}.nc')
    #
    # interp = GridInterpolator([table.R_sigma.values, table.CW_sigma.values], table.values)
    # da = da.sel(radius_down=radius_down[0, :], dist=dist[0, :])
    # X, Y = np.meshgrid(da.radius_down, da.dist)
    # err = da - interp(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)
    # print(np.abs(err).max())
    # (err).plot()
    #
    # table.plot()
    table = xr.load_dataarray(os.path.dirname(__file__) + f'/gaussian_overlap_.02_.02_{n_theta}_{n_r}.nc')

    compare2orsted(table)
    print(table)
