import numpy as np
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake import Fuga
from py_wake.examples.data import hornsrev1
import matplotlib.pyplot as plt
from py_wake.deficit_models.fuga import FugaBlockage, FugaDeficit, LUTInterpolator
from py_wake.flow_map import HorizontalGrid
from py_wake.tests.check_speed import timeit
from py_wake.utils.grid_interpolator import GridInterpolator


def test_fuga():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'
    site = UniformSite([1, 0, 0, 0], ti=0.075)
    wake_model = Fuga(path, site, wts)
    res, _ = timeit(wake_model.__call__, verbose=0, line_profile=0,
                    profile_funcs=[FugaDeficit.interpolate, LUTInterpolator.__call__, GridInterpolator.__call__])(x=wt_x, y=wt_y, wd=[30], ws=[10])

    npt.assert_array_almost_equal(res.WS_eff_ilk.flatten(),
                                  [10.002683492812844, 10.0, 8.413483643142389, 10.036952526815286,
                                   9.371203842245153, 8.437429367715435, 8.012759083790058], 8)
    npt.assert_array_almost_equal(res.ct_ilk.flatten(), [0.79285509, 0.793, 0.80641348, 0.79100456,
                                                         0.80180315, 0.80643743, 0.80601276], 8)

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
        [10.0547, 10.0519, 10.0718, 10.0093, 9.6786, 7.8589, 6.8539, 9.2199,
         9.9837, 10.036, 10.0796, 10.0469, 10.0439, 9.1866, 7.2552, 9.1518,
         10.0449, 10.0261, 10.0353, 9.9256, 9.319, 8.0062, 6.789, 8.3578,
         9.9393, 10.0332, 10.0191, 10.0186, 10.0191, 10.0139], 4)

    npt.assert_array_almost_equal(
        Z73[140, 100:400:10],
        [10.0542, 10.0514, 10.0706, 10.0075, 9.6778, 7.9006, 6.9218, 9.228,
         9.9808, 10.0354, 10.0786, 10.0464, 10.0414, 9.1973, 7.3099, 9.1629,
         10.0432, 10.0257, 10.0344, 9.9236, 9.3274, 8.0502, 6.8512, 8.3813,
         9.9379, 10.0325, 10.0188, 10.0183, 10.019, 10.0138], 4)


def test_fuga_blockage_wt_row():
    wts = HornsrevV80()
    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'
    site = hornsrev1.Hornsrev1Site()
    fuga_pdw = Fuga(path, site, wts)
    fuga_a2a = FugaBlockage(path, site, wts)

    x, y = [np.asarray(xy)[np.arange(0, 73, 8)] for xy in (hornsrev1.wt_x, hornsrev1.wt_y)]

    sim_res_pdw = fuga_pdw(x, y, wd=[270])
    aep = sim_res_pdw.aep_ilk()[:, 0, :]
    sim_res_a2a = fuga_a2a(x, y, wd=[270])
    aep_blockage = sim_res_a2a.aep_ilk()[:, 0, :]

    # blockage reduce aep(wd=270) by .4%
    npt.assert_almost_equal((aep.sum() - aep_blockage.sum()) / aep.sum() * 100, 0.4162679)

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

    path = tfp + 'fuga/2MW/Z0=0.00014617Zi=00399Zeta0=0.00E+0/'
    site = UniformSite([1, 0, 0, 0], ti=0.075)
    wake_model = Fuga(path, site, wts)
    res = wake_model(x=wt_x, y=wt_y, wd=[30], ws=[10])

    npt.assert_array_almost_equal(res.WS_eff_ilk.flatten(), [10.00725165, 10., 7.92176401, 10.02054952, 9.40501317,
                                                             7.92609363, 7.52384558], 8)
    npt.assert_array_almost_equal(res.ct_ilk.flatten(), [0.79260841, 0.793, 0.80592176, 0.79189033, 0.80132982,
                                                         0.80592609, 0.80552385], 8)

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
        print(list(np.round(Z70[140, 100:400:10], 4)))
        print(list(np.round(Z73[140, 100:400:10], 4)))
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(
        Z70[140, 100:400:10],
        [10.051, 10.0337, 10.0649, 10.0374, 9.7865, 7.7119, 6.4956, 9.2753, 10.0047, 10.0689, 10.0444,
         10.0752, 10.0699, 9.1852, 6.9783, 9.152, 10.0707, 10.0477, 10.0365, 9.9884, 9.2867, 7.5714,
         6.4451, 8.3276, 9.9976, 10.0251, 10.0261, 10.023, 10.0154, 9.9996], 4)

    npt.assert_array_almost_equal(
        Z73[140, 100:400:10],
        [10.051, 10.0337, 10.0649, 10.0374, 9.7865, 7.7119, 6.4956, 9.2753, 10.0047, 10.0689, 10.0444,
         10.0752, 10.0699, 9.1852, 6.9783, 9.152, 10.0707, 10.0477, 10.0365, 9.9884, 9.2867, 7.5714,
         6.4451, 8.3276, 9.9976, 10.0251, 10.0261, 10.023, 10.0154, 9.9996], 4)


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
#     path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'
#
#     site = UniformSite([1, 0, 0, 0], ti=0.075)
#
#     wake_model = Fuga(path, site, wts)
#     aep = AEPCalculator(wake_model)
#     wake_model(wt_x, wt_y, h_i=70, wd=[30], ws=[10]).flow_map(x_j, y_j, 70)
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
