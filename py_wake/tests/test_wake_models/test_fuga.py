import numpy as np
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.wake_models.fuga import Fuga
from py_wake.aep_calculator import AEPCalculator


def test_fuga():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'
    site = UniformSite([1, 0, 0, 0], ti=0.075)

    wake_model = Fuga(path, site, wts)
    aep = AEPCalculator(wake_model)
    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    _, _, Z70 = aep.wake_map(x_j, y_j, 70, wt_x, wt_y, wt_height=70, wd=[30], ws=[10])
    X, Y, Z73 = aep.wake_map(x_j, y_j, 73, wt_x, wt_y, wt_height=70, wd=[30], ws=[10])

    if 0:
        import matplotlib.pyplot as plt

        c = plt.contourf(X, Y, Z70, np.arange(6, 10.5, .1))
        plt.colorbar(c)
        plt.plot(X[0], Y[140])
        wts.plot(wt_x, wt_y)
        plt.figure()
        plt.plot(X[0], Z70[140, :], label="Z=70m")
        plt.plot(X[0], Z73[140, :], label="Z=73m")
        plt.plot(X[0, 100:400:10], Z70[140, 100:400:10], '.')
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


def cmp_fuga_with_colonel():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    xy, Z = [v for _, v in np.load(tfp + "fuga/U_XY_70m_.txt_30deg.npz").items()]

    x_min, x_max, x_step, y_min, y_max, y_step = xy
    x_j = np.arange(x_min, np.round((x_max - x_min) / x_step) * x_step + x_min + x_step, x_step)
    y_j = np.arange(y_min, np.round((y_max - y_min) / y_step) * y_step + y_min + y_step, y_step)

    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'

    site = UniformSite([1, 0, 0, 0], ti=0.075)

    wake_model = Fuga(path, site, wts)
    aep = AEPCalculator(wake_model)
    X, Y, Z2 = aep.wake_map(x_j, y_j, 70, wt_x, wt_y, h_i=70, wd=[30], ws=[10])

    print(x_j)
    print(y_j)
    m = (X == 500) & (Y == -880)
    print(Z[m])
    print(Z2[m])
    if 1:
        import matplotlib.pyplot as plt
        plt.clf()
        c = plt.contourf(X, Y, Z, np.arange(6, 10.5, .1))
        plt.colorbar(c, label="m/s")
        plt.axis('equal')
        plt.tight_layout()
        wts.plot(wt_x, wt_y)

        plt.figure()
        c = plt.contourf(X, Y, Z2, np.arange(6, 10.5, .1))
        plt.colorbar(c)

        wts.plot(wt_x, wt_y)

        plt.figure()
        c = plt.contourf(X, Y, Z2 - Z, np.arange(-.01, .01, .001))
        plt.colorbar(c, label="m/s")

        wts.plot(wt_x, wt_y)
        plt.show()

    npt.assert_array_almost_equal(Z, Z2, 2)


if __name__ == '__main__':
    cmp_fuga_with_colonel()
