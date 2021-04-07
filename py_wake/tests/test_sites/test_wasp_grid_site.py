import numpy as np
from numpy import newaxis as na
from py_wake.tests import npt
import pytest
from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path, ParqueFicticioSite
from py_wake.site.wasp_grid_site import WaspGridSite
import os
import time
from py_wake.tests.test_files.wasp_grid_site import one_layer
from py_wake.site.distance import TerrainFollowingDistance, StraightDistance, TerrainFollowingDistance2
import math
from py_wake import NOJ
from py_wake.wind_turbines import OneTypeWindTurbines
import matplotlib.pyplot as plt
from py_wake.site.xrsite import XRSite
import shutil


@pytest.fixture(autouse=True)
def close_plots():
    try:
        yield
    finally:
        try:
            plt.close('all')
        except Exception:
            pass


@pytest.fixture
def site():
    return ParqueFicticioSite()


@pytest.fixture
def site2():
    site = ParqueFicticioSite(distance=TerrainFollowingDistance2())
    x, y = site.initial_position.T
    return site, x, y


def test_WaspGridSiteDistanceClass(site):
    wgs = XRSite(site.ds, distance=TerrainFollowingDistance(distance_resolution=2000))
    assert wgs.distance.distance_resolution == 2000
    assert wgs.distance.__call__.__func__ == TerrainFollowingDistance().__call__.__func__
    wgs = XRSite(site.ds, distance=StraightDistance())
    assert wgs.distance.__call__.__func__ == StraightDistance().__call__.__func__


def test_local_wind(site):
    x_i, y_i = site.initial_position.T
    h_i = x_i * 0 + 70
    wdir_lst = np.arange(0, 360, 90)
    wsp_lst = np.arange(3, 6)
    WS_ilk = site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i, wd=wdir_lst, ws=wsp_lst).WS_ilk
    npt.assert_array_equal(WS_ilk.shape, (8, 4, 3))

    WS_ilk = site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i).WS_ilk
    npt.assert_array_equal(WS_ilk.shape, (8, 360, 23))

    # check probability local_wind()[-1]
    npt.assert_almost_equal(site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i, wd=[0], ws=[10], wd_bin_size=1).P_ilk,
                            site.local_wind(x_i=x_i, y_i=y_i, h_i=h_i, wd=[0], ws=[10], wd_bin_size=2).P_ilk / 2, 6)


def test_shear(site):
    npt.assert_array_almost_equal(site.ds['Speedup'].sel(x=262878, y=6504714, wd=0), [.6240589, .8932919])
    x = [262878.0001] * 3
    y = [6504714.0001] * 3
    z = [30, 115, 200]
    ws = site.local_wind(x_i=x, y_i=y, h_i=z, wd=[0], ws=[10]).WS_ilk[:, 0, 0]

    if 0:
        plt.plot(ws, z, '.-')
        plt.show()
    plt.close('all')

    # linear interpolation
    npt.assert_array_almost_equal(ws, [6.240589, np.mean([6.240589, 8.932919]), 8.932919])


def test_wasp_resources_grid_point(site):
    #     x = np.array([l.split() for l in """0.6010665    -10.02692    32.71442    -6.746912
    # 0.5007213    -4.591617    37.10247    -11.0699
    # 0.3104101    -1.821247    59.18301    -12.56743
    # 0.4674515    16.14293    44.84665    -9.693183
    # 0.8710347    5.291974    26.01634    -6.154611
    # 0.9998786    -2.777032    15.72486    1.029988
    # 0.9079611    -7.882853    16.33216    6.42329
    # 0.759553    -5.082487    17.23354    10.18187
    # 0.7221162    4.606324    17.96417    11.45838
    # 0.8088576    8.196074    16.16308    9.277925
    # 0.8800673    3.932325    14.82337    5.380589
    # 0.8726974    -3.199536    19.99724    -1.433086""".split("\n")], dtype=float)
    #     for x_ in x.T:
    #         print(list(x_))
    x = [262978]
    y = [6504814]
    npt.assert_almost_equal(site.elevation(x, y), 227.8, 1)

    # Data from WAsP:
    # - add turbine (262878,6504814,30)
    # - Turbine (right click) - reports - Turbine Site Report full precision
    wasp_A = [2.197305, 1.664085, 1.353185, 2.651781, 5.28438, 5.038289,
              4.174325, 4.604496, 5.043066, 6.108261, 6.082033, 3.659798]
    wasp_k = [1.771484, 2.103516, 2.642578, 2.400391, 2.357422, 2.306641,
              2.232422, 2.357422, 2.400391, 2.177734, 1.845703, 1.513672]
    wasp_f = [5.188083, 2.509297, 2.869334, 4.966141, 13.16969, 9.514355,
              4.80275, 6.038354, 9.828702, 14.44174, 16.60567, 10.0659]
    wasp_spd = [0.6010665, 0.5007213, 0.3104101, 0.4674515, 0.8710347, 0.9998786,
                0.9079611, 0.759553, 0.7221162, 0.8088576, 0.8800673, 0.8726974]
    wasp_trn = [-10.02692, -4.591617, -1.821247, 16.14293, 5.291974, -
                2.777032, -7.882853, -5.082487, 4.606324, 8.196074, 3.932325, -3.199536]
    wasp_inc = [-6.746912, -11.0699, -12.56743, -9.693183, -6.154611,
                1.029988, 6.42329, 10.18187, 11.45838, 9.277925, 5.380589, -1.433086]
    wasp_ti = [32.71442, 37.10247, 59.18301, 44.84665, 26.01634, 15.72486,
               16.33216, 17.23354, 17.96417, 16.16308, 14.82337, 19.99724]
    rho = 1.179558

    wasp_u_mean = [1.955629, 1.473854, 1.202513, 2.350761, 4.683075,
                   4.463644, 3.697135, 4.080554, 4.470596, 5.409509, 5.402648, 3.300305]
    wasp_p_air = [9.615095, 3.434769, 1.556282, 12.45899, 99.90289,
                  88.03519, 51.41135, 66.09097, 85.69466, 164.5592, 193.3779, 56.86945]
#     wasp_aep = np.array([3725293.0, 33722.71, 0.3093564, 3577990.0, 302099600.0, 188784100.0,
#                          48915640.0, 84636210.0, 189009800.0, 549195100.0, 691258600.0, 120013000.0]) / 1000
    wasp_aep_no_density_correction = np.array([3937022.0, 36046.93, 0.33592, 3796496.0, 314595600.0,
                                               196765700.0, 51195440.0, 88451200.0, 197132700.0, 568584400.0, 712938400.0, 124804600.0]) / 1000
#     wasp_aep_total = 2.181249024
    wasp_aep_no_density_correction_total = 2.26224
    wt_u = np.array([3.99, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    wt_p = np.array([0, 55., 185., 369., 619., 941., 1326., 1741., 2133., 2436., 2617., 2702., 2734.,
                     2744., 2747., 2748., 2748., 2750., 2750., 2750., 2750., 2750., 2750.])
    wt_ct = np.array([0, 0.871, 0.853, 0.841, 0.841, 0.833, 0.797, 0.743, 0.635, 0.543, 0.424,
                      0.324, 0.258, 0.21, 0.175, 0.147, 0.126, 0.109, 0.095, 0.083, 0.074, 0.065, 0.059])
    wt = OneTypeWindTurbines.from_tabular(name="NEG-Micon 2750/92 (2750 kW)", diameter=92, hub_height=70,
                                          ws=wt_u, ct=wt_ct, power=wt_p, power_unit='kw')

    lw = site.local_wind(x, y, 30, wd=range(0, 360, 30))
    A_lst, k_lst, f_lst, spd_lst, orog_trn_lst, flow_inc_lst, ti15ms_lst = [
        site.interp(site.ds[n], lw.coords).sel(i=0).values
        for n in ['Weibull_A', 'Weibull_k', 'Sector_frequency', 'Speedup', 'Turning', 'flow_inc', 'ti15ms']]
    pdf_lst = [lambda x, A=A, k=k: k / A * (x / A)**(k - 1) * np.exp(-(x / A)**k) * (x[1] - x[0])
               for A, k in zip(A_lst, k_lst)]
#     cdf_lst = [lambda x, A=A, k=k: 1 - np.exp(-(x / A) ** k) for A, k in zip(A_lst, k_lst)]
    dx = .1
    ws = np.arange(dx / 2, 35, dx)

    # compare to wasp data
    npt.assert_array_equal(A_lst, wasp_A)
    npt.assert_array_equal(k_lst, wasp_k)
    npt.assert_array_almost_equal(f_lst, np.array(wasp_f) / 100)
    npt.assert_array_almost_equal(spd_lst, wasp_spd)
    npt.assert_array_almost_equal(orog_trn_lst, wasp_trn)
    npt.assert_array_almost_equal(flow_inc_lst, wasp_inc)
    npt.assert_array_almost_equal(ti15ms_lst, np.array(wasp_ti) / 100)

    # compare pdf, u_mean and aep to wasp
    lw = site.local_wind(x, np.array(y) + 1e-6, 30, wd=np.arange(0, 360, 30), ws=ws)
    P = lw.P_ilk / lw.P_ilk.sum(2)[:, :, na]  # only wind speed probablity (not wdir)

    # pdf
    for l in range(12):
        npt.assert_array_almost_equal(np.interp(ws, lw.WS_ilk[0, l], np.cumsum(P[0, l])),
                                      np.cumsum(pdf_lst[l](ws)), 1)

    # u_mean
    npt.assert_almost_equal([A * math.gamma(1 + 1 / k) for A, k in zip(A_lst, k_lst)], wasp_u_mean, 5)
    npt.assert_almost_equal([(pdf(ws) * ws).sum() for pdf in pdf_lst], wasp_u_mean, 5)
    npt.assert_almost_equal((P * lw.WS_ilk).sum((0, 2)), wasp_u_mean, 5)

    # air power
    p_air = [(pdf(ws) * 1 / 2 * rho * ws**3).sum() for pdf in pdf_lst]
    npt.assert_array_almost_equal(p_air, wasp_p_air, 3)
    npt.assert_array_almost_equal((P * 1 / 2 * rho * lw.WS_ilk**3).sum((0, 2)), wasp_p_air, 2)

    # AEP
    AEP_ilk = NOJ(site, wt)(x, y, h=30, wd=np.arange(0, 360, 30), ws=ws).aep_ilk(
        with_wake_loss=False, normalize_probabilities=True)
    if 0:
        plt.plot(wasp_aep_no_density_correction / 1000, '.-', label='WAsP')
        plt.plot(AEP_ilk.sum((0, 2)) * 1e3, label='PyWake')
        plt.xlabel('Sector')
        plt.ylabel('AEP [MWh]')
        plt.legend()
        plt.show()
    npt.assert_array_less(np.abs(wasp_aep_no_density_correction - AEP_ilk.sum((0, 2)) * 1e6), 300)
    npt.assert_almost_equal(AEP_ilk.sum(), wasp_aep_no_density_correction_total, 3)


@pytest.mark.parametrize('site,dw_ref', [
    (ParqueFicticioSite(distance=TerrainFollowingDistance2()),
     [0., 207.3842238, 484.3998264, 726.7130743, 1039.148129, 1263.1335982, 1490.3841602, 1840.6508086]),
    (ParqueFicticioSite(distance=TerrainFollowingDistance()),
     [0, 209.803579, 480.8335365, 715.6003233, 1026.9476322, 1249.5510034, 1475.1467251, 1824.1317343]),
    (ParqueFicticioSite(distance=StraightDistance()),
     [-0, 207, 477, 710, 1016, 1236, 1456, 1799])])
def test_distances(site, dw_ref):
    x, y = site.initial_position.T
    site.distance.setup(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
                        dst_xyh_j=(x, y, np.array([70])))
    dw_ijl, cw_ijl, dh_ijl = site.distance(wd_il=np.array([[0]]))
    npt.assert_almost_equal(dw_ijl[0, :, 0], dw_ref)

    cw_ref = [236.1, 0., -131.1, -167.8, -204.5, -131.1, -131.1, -45.4]
    npt.assert_almost_equal(cw_ijl[:, 1, 0], cw_ref)
    npt.assert_almost_equal(dh_ijl, np.zeros_like(dh_ijl))


def test_distances_different_points(site2):
    site, x, y = site2
    with pytest.raises(NotImplementedError):
        site.distance.setup(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
                            dst_xyh_j=(x[1:], y[1:], np.array([70])))
        site.distance(wd_il=np.array([[0]]))


# def test_distances_wd_shape():
#     site = ParqueFicticioSite(distance=TerrainFollowingDistance2())
#     x, y = site.initial_position.T
#     dw_ijl, cw_ijl, dh_ijl, dwo = site.distances(src_x_i=x, src_y_i=y, src_h_i=np.array([70]),
#                                                  dst_x_j=x, dst_y_j=y, dst_h_j=np.array([70]),
#                                                  wd_il=np.ones((len(x), 1)) * 180)
#     npt.assert_almost_equal(dw_ijl[0, :, 0], np.array([0., -207., -477., -710., -1016., -1236., -1456., -1799.]))
#     npt.assert_almost_equal(cw_ijl[:, 1, 0], np.array([-236.1, 0., 131.1, 167.8, 204.5, 131.1, 131.1, 45.4]))
#     npt.assert_almost_equal(dh_ijl, np.zeros_like(dh_ijl))


def test_speed_up_using_pickle():
    pkl_fn = ParqueFicticio_path + "ParqueFicticio.pkl"
    if os.path.exists(pkl_fn):
        os.remove(pkl_fn)
    start = time.time()
    site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=False)
    time_wo_pkl = time.time() - start
    site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=True)
    assert os.path.exists(pkl_fn)
    start = time.time()
    site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=True)
    time_w_pkl = time.time() - start
    npt.assert_array_less(time_w_pkl * 10, time_wo_pkl)


def test_speed_up_using_pickle_wrong_pkl():
    pkl_fn = ParqueFicticio_path + "ParqueFicticio.pkl"
    if os.path.exists(pkl_fn):
        os.remove(pkl_fn)
    shutil.copy(__file__, pkl_fn)
    site = WaspGridSite.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=True)


def test_one_layer():
    site = WaspGridSite.from_wasp_grd(os.path.dirname(one_layer.__file__) + "/", speedup_using_pickle=False)


def test_missing_path():
    with pytest.raises(NotImplementedError):
        WaspGridSite.from_wasp_grd("missing_path/", speedup_using_pickle=True)

    with pytest.raises(Exception, match='Path was not a directory'):
        WaspGridSite.from_wasp_grd("missing_path/", speedup_using_pickle=False)


def test_elevation(site):
    x_i, y_i = site.initial_position.T
    npt.assert_array_less(np.abs(site.elevation(x_i=x_i, y_i=y_i) -
                                 [519.4, 567.7, 583.6, 600, 574.8, 559.9, 517.7, 474.5]  # ref from wasp
                                 ), 5)


def test_plot_map(site):
    site.ds.Elevation.plot()
    plt.figure()
    site.ds.ws_mean.sel(h=200, wd=0).plot()
    if 0:
        plt.show()
    plt.close('all')


def test_elevation_outside_map(site):

    site.ds.Elevation.plot()
    x = np.linspace(262500, 265500, 500)
    y = x * 0 + 6505450
    plt.plot(x, y, '--', label='Terrain line')
    plt.plot(x, y + site.elevation(x, y), label='Elevation')
    npt.assert_array_equal(np.round(site.elevation(x, y)[::50]),
                           [np.nan, np.nan, 303, 390, 491, 566, 486, 524, np.nan, np.nan])
    if 0:
        plt.legend()
        plt.show()
    plt.close('all')


def test_plot_ws_distribution(site):
    x, y = site.initial_position[0]
    p1 = site.plot_ws_distribution(x=x, y=y, h=70, wd=[0, 90, 180, 270])
    p2 = site.plot_ws_distribution(x=x, y=y, h=70, wd=[0, 90, 180, 270], include_wd_distribution=True)
    if 0:
        print(np.round(p1[-1, ::30].data, 4).tolist())
        print(np.round(p2[-1, ::30].data, 4).tolist())
        plt.show()

    npt.assert_array_almost_equal(p1[-1, ::30], [0.0001, 0.0079, 0.011, 0.0082, 0.0039,
                                                 0.0013, 0.0003, 0.0, 0.0, 0.0], 4)
    npt.assert_array_almost_equal(p2[-1, ::30], [0.0001, 0.0036, 0.0047, 0.0033, 0.0014,
                                                 0.0004, 0.0001, 0.0, 0.0, 0.0], 4)
    plt.close('all')


def test_plot_wd_distribution(site):
    x, y = site.initial_position[0]
    p = site.plot_wd_distribution(x=x, y=y, h=70, n_wd=12, ax=plt)
    # print(np.round(p, 3).tolist())

    if 0:
        plt.show()
    plt.close('all')
    npt.assert_array_almost_equal(p, [0.052, 0.043, 0.058, 0.085, 0.089, 0.061,
                                      0.047, 0.083, 0.153, 0.152, 0.108, 0.068], 3)


def test_plot_wd_distribution_with_ws_levels(site):
    x, y = site.initial_position[0]
    p = site.plot_wd_distribution(x=x, y=y, n_wd=12, ws_bins=[0, 5, 10, 15, 20, 25])
    if 0:
        print(np.round(p, 3).data.tolist())
        plt.show()
    npt.assert_array_almost_equal(p, [[0.039, 0.013, 0.001, 0.0, 0.0],
                                      [0.034, 0.009, 0.0, 0.0, 0.0],
                                      [0.035, 0.022, 0.0, 0.0, 0.0],
                                      [0.034, 0.048, 0.003, 0.0, 0.0],
                                      [0.031, 0.052, 0.007, 0.0, 0.0],
                                      [0.03, 0.03, 0.002, 0.0, 0.0],
                                      [0.027, 0.019, 0.001, 0.0, 0.0],
                                      [0.034, 0.043, 0.006, 0.0, 0.0],
                                      [0.047, 0.082, 0.023, 0.001, 0.0],
                                      [0.048, 0.074, 0.026, 0.003, 0.0],
                                      [0.044, 0.046, 0.015, 0.002, 0.0],
                                      [0.041, 0.023, 0.003, 0.0, 0.0]], 3)
    plt.close('all')


def test_additional_input():
    site = ParqueFicticioSite()

    x, y = site.initial_position.T
    h = 70 * np.ones_like(x)

    lw = site.local_wind(x, y, h)
    ws_mean = site.interp(site.ds.ws_mean, lw.coords)
    npt.assert_array_almost_equal(ws_mean[0, :50],
                                  np.array([4.77080802, 4.77216214, 4.77351626, 4.77487037, 4.77622449,
                                            4.77757861, 4.77893273, 4.78028685, 4.78164097, 4.78299508,
                                            4.7843492, 4.78570332, 4.78705744, 4.78841156, 4.78976567,
                                            4.79111979, 4.79247391, 4.79382803, 4.79518215, 4.79653626,
                                            4.79789038, 4.7992445, 4.80059862, 4.80195274, 4.80330685,
                                            4.80466097, 4.80601509, 4.80736921, 4.80872333, 4.81007744,
                                            4.81143156, 4.87114138, 4.9308512, 4.99056102, 5.05027084,
                                            5.10998066, 5.16969047, 5.22940029, 5.28911011, 5.34881993,
                                            5.40852975, 5.46823957, 5.52794939, 5.5876592, 5.64736902,
                                            5.70707884, 5.76678866, 5.82649848, 5.8862083, 5.94591812]))


if __name__ == '__main__':
    test_wasp_resources_grid_point(ParqueFicticioSite())
