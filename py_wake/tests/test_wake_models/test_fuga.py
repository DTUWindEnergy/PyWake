
import numpy as np
from py_wake.aep._aep import AEP
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.wake_models.fuga import FugaWakeModel


def test_fuga():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/'
    site = UniformSite([1, 0, 0, 0], ti=0.75)

    wake_model = FugaWakeModel(path, wts)
    aep = AEP(site, wts, wake_model)
    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    X, Y, Z2 = aep.wake_map(x_j, y_j, 70, wt_x, wt_y, h_i=70, wd=[30], ws=[10])

    npt.assert_array_almost_equal(
        Z2[140, 100:400:10],
        [10.0547, 10.0519, 10.0718, 10.0093, 9.6786, 7.8589, 6.8539, 9.2199,
         9.9837, 10.036, 10.0796, 10.0469, 10.0439, 9.1866, 7.2552, 9.1518,
         10.0449, 10.0261, 10.0353, 9.9256, 9.319, 8.0062, 6.789, 8.3578,
         9.9393, 10.0332, 10.0191, 10.0186, 10.0191, 10.0139], 4)

    if 0:
        import matplotlib.pyplot as plt

        c = plt.contourf(X, Y, Z2, np.arange(6, 10.5, .1))
        plt.colorbar(c)
        plt.plot(X[0], Y[140])
        wts.plot(wt_x, wt_y)
        plt.figure()
        plt.plot(X[0], Z2[140, :])
        plt.plot(X[0, 100:400:10], Z2[140, 100:400:10], '.')
        plt.show()


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

    site = UniformSite([1, 0, 0, 0], ti=0.75)

    wake_model = FugaWakeModel(path, wts)
    aep = AEP(site, wts, wake_model)
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
