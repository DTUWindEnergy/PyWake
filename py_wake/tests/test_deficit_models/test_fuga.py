import numpy as np
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake import Fuga
from py_wake.examples.data import hornsrev1
import matplotlib.pyplot as plt
from py_wake.deficit_models.fuga import FugaBlockage, FugaDeficit, LUTInterpolator, FugaUtils, FugaYawDeficit
from py_wake.flow_map import HorizontalGrid, XYGrid
from py_wake.tests.check_speed import timeit
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
import pytest
from pathlib import Path
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine


def test_fuga():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'
    site = UniformSite([1, 0, 0, 0], ti=0.075)
    wake_model = Fuga(path, site, wts)
    res, _ = timeit(wake_model.__call__, verbose=0, line_profile=0,
                    profile_funcs=[FugaDeficit.interpolate, LUTInterpolator.__call__, GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])

    npt.assert_array_almost_equal(res.WS_eff_ilk.flatten(),
                                  [10.00669629, 10., 8.47606501, 10.03143097, 9.37288077,
                                   8.49301941, 8.07462708], 8)
    npt.assert_array_almost_equal(res.ct_ilk.flatten(), [0.7926384, 0.793, 0.80647607, 0.79130273, 0.80177967,
                                                         0.80649302, 0.80607463], 8)

    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    wake_model = Fuga(path, site, wts)
    sim_res = wake_model(wt_x, wt_y, wd=[30], ws=[10])
    flow_map70 = sim_res.flow_map(HorizontalGrid(x_j, y_j, h=70))
    flow_map73 = sim_res.flow_map(HorizontalGrid(x_j, y_j, h=73))

    X, Y = flow_map70.XY
    Z70 = flow_map70.WS_eff_xylk[:, :, 0, 0]
    Z73 = flow_map73.WS_eff_xylk[:, :, 0, 0]

    if 0:
        flow_map70.plot_wake_map(levels=np.arange(6, 10.5, .1))
        plt.plot(X[0], Y[140])
        plt.figure()
        plt.plot(X[0], Z70[140, :], label="Z=70m")
        plt.plot(X[0], Z73[140, :], label="Z=73m")
        plt.plot(X[0, 100:400:10], Z70[140, 100:400:10], '.')
        print(list(np.round(Z70.data[140, 100:400:10], 4)))
        print(list(np.round(Z73.data[140, 100:400:10], 4)))
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(
        Z70[140, 100:400:10],
        [10.0407, 10.0438, 10.0438, 10.013, 9.6847, 7.8787, 6.9561, 9.2251, 9.9686, 10.0382, 10.0498,
         10.0569, 10.0325, 9.1787, 7.4004, 9.1384, 10.0329, 10.0297, 10.0232, 9.9265, 9.3163, 8.0768,
         6.8858, 8.3754, 9.9592, 10.0197, 10.0118, 10.0141, 10.0118, 10.0095], 4)

    npt.assert_array_almost_equal(
        Z73[140, 100:400:10],
        [10.0404, 10.0435, 10.0433, 10.0113, 9.6836, 7.9206, 7.0218, 9.2326, 9.9665, 10.0376, 10.0494,
         10.0563, 10.0304, 9.1896, 7.4515, 9.15, 10.0317, 10.0294, 10.0226, 9.9245, 9.3252, 8.1192, 6.9462,
         8.3988, 9.9574, 10.0194, 10.0117, 10.014, 10.0117, 10.0094], 4)


def test_fuga_blockage_wt_row():
    wts = HornsrevV80()
    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'
    site = hornsrev1.Hornsrev1Site()
    fuga_pdw = Fuga(path, site, wts)
    fuga_a2a = FugaBlockage(path, site, wts)

    x, y = [np.asarray(xy)[np.arange(0, 73, 8)] for xy in (hornsrev1.wt_x, hornsrev1.wt_y)]

    sim_res_pdw = fuga_pdw(x, y, wd=[270])
    aep = sim_res_pdw.aep_ilk()[:, 0, :]
    sim_res_a2a = fuga_a2a(x, y, wd=[270])
    aep_blockage = sim_res_a2a.aep_ilk()[:, 0, :]

    # blockage reduce aep(wd=270) by .24%
    npt.assert_almost_equal((aep.sum() - aep_blockage.sum()) / aep.sum() * 100, 0.2433161515321294)

    if 0:
        plt.plot((sim_res_pdw.WS_eff_ilk[:, 0, 7] - sim_res_a2a.WS_eff_ilk[:, 0, 7]) /
                 sim_res_pdw.WS_eff_ilk[:, 0, 7] * 100)
        plt.grid()
        plt.show()


def test_fuga_new_format():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    path = tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/'
    site = UniformSite([1, 0, 0, 0], ti=0.075)
    wake_model = Fuga(path, site, wts)
    res = wake_model(x=wt_x, y=wt_y, wd=[30], ws=[10])

    npt.assert_array_almost_equal(res.WS_eff_ilk.flatten(), [10.00647891, 10., 8.21713928, 10.03038884, 9.36889964,
                                                             8.23084088, 7.80662141], 8)
    npt.assert_array_almost_equal(res.ct_ilk.flatten(), [0.79265014, 0.793, 0.80621714, 0.791359, 0.80183541,
                                                         0.80623084, 0.80580662], 8)

    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    wake_model = Fuga(path, site, wts)
    sim_res = wake_model(wt_x, wt_y, wd=[30], ws=[10])
    flow_map70 = sim_res.flow_map(HorizontalGrid(x_j, y_j, h=70))
    flow_map73 = sim_res.flow_map(HorizontalGrid(x_j, y_j, h=73))

    X, Y = flow_map70.XY
    Z70 = flow_map70.WS_eff_xylk[:, :, 0, 0]
    Z73 = flow_map73.WS_eff_xylk[:, :, 0, 0]

    if 0:
        flow_map70.plot_wake_map(levels=np.arange(6, 10.5, .1))
        plt.plot(X[0], Y[140])
        plt.figure()
        plt.plot(X[0], Z70[140, :], label="Z=70m")
        plt.plot(X[0], Z73[140, :], label="Z=73m")
        plt.plot(X[0, 100:400:10], Z70[140, 100:400:10], '.')
        print(list(np.round(Z70.values[140, 100:400:10], 4)))
        print(list(np.round(Z73.values[140, 100:400:10], 4)))
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(
        Z70[140, 100:400:10],
        [10.0384, 10.042, 10.044, 10.0253, 9.7194, 7.7561, 6.7421, 9.2308, 9.9894, 10.0413, 10.0499,
         10.0579, 10.0437, 9.1626, 7.2334, 9.1208, 10.0396, 10.0322, 10.0276, 9.9504, 9.2861, 7.8375,
         6.6608, 8.3343, 9.9756, 10.0229, 10.0136, 10.0142, 10.0118, 10.0094], 4)

    npt.assert_array_almost_equal(
        Z73[140, 100:400:10],
        [10.0384, 10.042, 10.044, 10.0253, 9.7194, 7.7561, 6.7421, 9.2308, 9.9894, 10.0413, 10.0499,
         10.0579, 10.0437, 9.1626, 7.2334, 9.1208, 10.0396, 10.0322, 10.0276, 9.9504, 9.2861, 7.8375,
         6.6608, 8.3343, 9.9756, 10.0229, 10.0136, 10.0142, 10.0118, 10.0094], 4)


def test_fuga_downwind():
    wts = HornsrevV80()

    path = tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00'
    site = UniformSite([1, 0, 0, 0], ti=0.075)
    wfm_UL = Fuga(path, site, wts)

    wfm_ULT = PropagateDownwind(site, wts, FugaYawDeficit(path))

    (ax1, ax2), (ax3, ax4) = plt.subplots(2, 2)[1]

    def plot(wfm, yaw, ax, min_ws):
        levels = np.arange(6.5, 10.5, .5)
        sim_res = wfm([0], [0], wd=270, ws=10, yaw=[[[yaw]]])
        fm = sim_res.flow_map(XYGrid(x=np.arange(-100, 500, 5)))
        npt.assert_almost_equal(fm.WS_eff.min(), min_ws)
        fm.plot_wake_map(ax=ax, levels=levels)
        fm.min_WS_eff(fm.x, 70).plot(ax=ax, color='r')
        plt.axhline(0, color='k')
    plot(wfm_UL, 0, ax1, 7.15853738)
    plot(wfm_UL, 30, ax2, 7.83219266)
    plot(wfm_ULT, 0, ax3, 7.15853738)
    plot(wfm_ULT, 30, ax4, 8.12261872)

    if 0:
        plt.show()
    plt.close('all')


def test_fuga_downwind_vs_notebook():

    powerCtFunction = PowerCtTabular([0, 100], [0, 0], 'w', [0.850877, 0.850877])
    wt = WindTurbine(name='', diameter=80, hub_height=70, powerCtFunction=powerCtFunction)

    path = tfp + 'fuga/2MW/Z0=0.00001000Zi=00400Zeta0=0.00E+00'
    site = UniformSite([1, 0, 0, 0], ti=0.075)
    wfm_ULT = PropagateDownwind(site, wt, FugaYawDeficit(path))
    WS = 10
    p = Path(tfp) / "fuga/v80_wake_4d_y_no_deflection.csv"
    y, notebook_deficit_4d = np.array([v.split(",") for v in p.read_text().strip().split("\n")], dtype=float).T
    sim_res = wfm_ULT([0], [0], wd=270, ws=WS, yaw=[[[17.4493]]])
    fm = sim_res.flow_map(XYGrid(4 * wt.diameter(), y=y))
    npt.assert_allclose(fm.WS_eff.squeeze() - WS, notebook_deficit_4d, atol=1e-6)

    if 0:
        plt.plot(y, notebook_deficit_4d, label='Notebook deficit 4d')
        plt.plot(y, fm.WS_eff.squeeze() - WS)
        plt.show()
    plt.close('all')


def test_fuga_table_edges():

    wts = HornsrevV80()
    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'
    site = hornsrev1.Hornsrev1Site()
    fuga = FugaBlockage(path, site, wts)

    D = 80
    flow_map_dw = fuga([0], [0], wd=270, ws=10).flow_map(HorizontalGrid(np.arange(-200 * D, 450 * D), y=[0]))
    flow_map_cw = fuga([0], [0], wd=270, ws=10).flow_map(HorizontalGrid([0], np.arange(-20 * D, 20 * D)))
    flow_map = fuga([0], [0], wd=270, ws=10).flow_map(HorizontalGrid(np.arange(-150, 400) * D, np.arange(-20, 21) * D))

    if 0:
        plt.plot(flow_map_dw.x / D, flow_map_dw.WS_eff.squeeze())
        plt.grid()
        plt.ylim([9.9, 10.1])
        plt.figure()
        plt.plot(flow_map_cw.y / D, flow_map_cw.WS_eff.squeeze())
        plt.grid()
        plt.ylim([9.9, 10.1])
        plt.figure()
        flow_map.WS_eff.plot()

        plt.show()

    npt.assert_array_equal(flow_map.WS_eff.squeeze()[[0, -1], :], 10)
    npt.assert_array_equal(flow_map.WS_eff.squeeze()[:, [0, -1]], 10)


def test_fuga_wriggles():
    wts = HornsrevV80()
    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'
    site = hornsrev1.Hornsrev1Site()
    fuga = PropagateDownwind(site, wts, FugaDeficit(path, remove_wriggles=True))

    D = 80
    flow_map_cw = fuga([0], [0], wd=270, ws=10).flow_map(HorizontalGrid([0], np.arange(-20 * D, 20 * D)))

    y = np.linspace(-5 * D, 5 * D, 100)

    dw_lst = range(10)
    flow_map_cw_lst = np.array([fuga([0], [0], wd=270, ws=10).flow_map(HorizontalGrid([dw * D], y)).WS_eff.squeeze()
                                for dw in dw_lst])

    if 0:
        for flow_map_cw, dw in zip(flow_map_cw_lst, dw_lst):
            plt.plot(y, flow_map_cw, label="%dD" % dw)
        plt.xlabel('y [m]')
        plt.ylabel('ws [m/s')
        plt.ylim([9.9, 10.1])
        plt.grid()
        plt.legend(loc=1)
        plt.show()
    assert np.all(flow_map_cw_lst > 0)


# def test_fuga_utils_mismatch():
#     path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'
#     with pytest.raises(ValueError, match="Mismatch between CaseData.bin and 2MW_FIT_input.par: low_level 102!=155"):
#         FugaUtils(path)


def test_mirror():
    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'
    fuga_utils = FugaUtils(path, on_mismatch='input_par')
    npt.assert_array_almost_equal(fuga_utils.mirror([0, 1, 3]), [3, 1, 0, 1, 3])
    npt.assert_array_almost_equal(fuga_utils.mirror([0, 1, 3], anti_symmetric=True), [-3, -1, 0, 1, 3])


def test_lut_exists():
    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'
    fuga_utils = FugaUtils(path, on_mismatch='input_par')
    assert sorted(fuga_utils.lut_exists()) == ['UL', 'UT', 'VL', 'VT']
    assert fuga_utils.lut_exists([154]) == set([])
    assert sorted(fuga_utils.lut_exists([155])) == ['UL', 'UT', 'VL', 'VT']

    path = tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/'
    fuga_utils = FugaUtils(path, on_mismatch='input_par')
    assert fuga_utils.lut_exists() == {'UL', 'UT', 'VL', 'VT'}
    assert fuga_utils.lut_exists([154]) == set([])
    assert fuga_utils.lut_exists([9999]) == {'UL', 'UT', 'VL', 'VT'}


def test_interpolation():
    wts = HornsrevV80()

    path = tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/'
    site = UniformSite([1, 0, 0, 0], ti=0.075)

    plot = 0
    if plot:
        ax1 = plt.gca()
        ax2 = plt.twinx()

    for wdm, n_d_values in ((FugaDeficit(path, method='linear'), 4),
                            (FugaDeficit(path, method='spline'), 20),
                            (FugaYawDeficit(path, method='linear'), 4),
                            (FugaYawDeficit(path, method='spline'), 20),
                            ):
        wfm = PropagateDownwind(site, wts, wdm)

        sim_res = wfm(x=[0], y=[0], wd=[270], ws=[10], yaw=[[[10]]])
        fm = sim_res.flow_map(XYGrid(x=[200], y=np.arange(-10, 11)))
        fm = sim_res.flow_map(XYGrid(x=np.arange(-100, 800, 10), y=np.arange(-10, 11)))

        # linear has 4 line segments with same gradient, while spline has 20 different gradient values
        npt.assert_equal(len(np.unique(np.round(np.diff(fm.WS_eff.sel(x=500).squeeze()), 6))), n_d_values)
        if plot:
            ax1.plot(fm.y, fm.WS_eff.sel(x=500).squeeze())
            ax2.plot(fm.y[:-1], np.diff(fm.WS_eff.sel(x=500).squeeze()), '--')

    if plot:
        plt.show()
        plt.close('all')

# def cmp_fuga_with_colonel():
#     from py_wake.aep_calculator import AEPCalculator
#
#     # move turbine 1 600 300
#     wt_x = [-250, 600, -500, 0, 500, -250, 250]
#     wt_y = [433, 300, 0, 0, 0, -433, -433]
#     wts = HornsrevV80()
#
#     xy, Z = [v for _, v in np.load(tfp + "fuga/U_XY_70m_.txt_30deg.npz").items()]
#
#     x_min, x_max, x_step, y_min, y_max, y_step = xy
#     x_j = np.arange(x_min, np.round((x_max - x_min) / x_step) * x_step + x_min + x_step, x_step)
#     y_j = np.arange(y_min, np.round((y_max - y_min) / y_step) * y_step + y_min + y_step, y_step)
#
#     path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'
#
#     site = UniformSite([1, 0, 0, 0], ti=0.075)
#
#     wfm = Fuga(path, site, wts)
#     aep = AEPCalculator(wfm)
#     wfm(wt_x, wt_y, h_i=70, wd=[30], ws=[10]).flow_map(x_j, y_j, 70)
#     X, Y, Z2 = aep.wake_map(x_j, y_j, 70, wt_x, wt_y, h_i=70, wd=[30], ws=[10])
#
#     print(x_j)
#     print(y_j)
#     m = (X == 500) & (Y == -880)
#     print(Z[m])
#     print(Z2[m])
#     if 0:
#         plt.clf()
#         c = plt.contourf(X, Y, Z, np.arange(6, 10.5, .1))
#         plt.colorbar(c, label="m/s")
#         plt.axis('equal')
#         plt.tight_layout()
#         wts.plot(wt_x, wt_y)
#
#         plt.figure()
#         c = plt.contourf(X, Y, Z2, np.arange(6, 10.5, .1))
#         plt.colorbar(c)
#
#         wts.plot(wt_x, wt_y)
#
#         plt.figure()
#         c = plt.contourf(X, Y, Z2 - Z, np.arange(-.01, .01, .001))
#         plt.colorbar(c, label="m/s")
#
#         wts.plot(wt_x, wt_y)
#         plt.show()
#
#     npt.assert_array_almost_equal(Z, Z2, 2)
#
#
# if __name__ == '__main__':
#     cmp_fuga_with_colonel()
