from py_wake.examples.data.hornsrev_setup import HornsrevV80
import numpy as np
from py_wake.tests.test_files import tfp
from py_wake.wake_models.fuga import FugaWakeModel
from py_wake.aep._aep import AEP
from py_wake.site._site import UniformSite
from py_wake.tests import npt


def test_fuga():
    # move turbine 1 600 300
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    path = r'C:\mmpe\programming\python\Topfarm\TopFarm2\topfarm\cost_models\fuga\Colonel\LUTs-T\Vestas_V80_(2_MW_offshore)[h=70.00]\Z0=0.03000000Zi=00401Zeta0=0.00E+0/'
    site = UniformSite([1, 0, 0, 0], ti=0.75)

    wake_model = FugaWakeModel(path, wts)
    aep = AEP(site, wts, wake_model)
    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 300)

    X, Y, Z2 = aep.wake_map(x_j, y_j, 70, wt_x, wt_y, h_i=70, wd=[30], ws=[10])

    if 1:
        import matplotlib.pyplot as plt
        c = plt.contourf(X, Y, Z2, np.arange(6, 10.5, .1))
        plt.colorbar(c)

        wts.plot(wt_x, wt_y)
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

    path = r'C:\mmpe\programming\python\Topfarm\TopFarm2\topfarm\cost_models\fuga\Colonel\LUTs-T\Vestas_V80_(2_MW_offshore)[h=70.00]\Z0=0.03000000Zi=00401Zeta0=0.00E+0/'

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
