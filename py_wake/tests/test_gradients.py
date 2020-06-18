import matplotlib.pyplot as plt
import numpy as np
from autograd import numpy as anp
from py_wake.examples.data.iea37 import iea37_reader
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.utils.gradients import use_autograd_in, autograd, plot_gradients, fd, cs
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines
from py_wake import wind_turbines
from py_wake.examples.data.hornsrev1 import V80
import pytest


@pytest.mark.parametrize('obj', [wind_turbines, WindTurbines, V80().power, wind_turbines.__dict__])
def test_use_autograd_in(obj):
    assert wind_turbines.np == np
    with use_autograd_in([obj]):
        assert wind_turbines.np == anp
    assert wind_turbines.np == np


def test_gradients():
    wt = IEA37_WindTurbines()
    with use_autograd_in([WindTurbines, iea37_reader]):
        ws_lst = np.arange(3, 25, .1)

        ws_pts = np.array([3., 6., 9., 12.])
        dpdu_lst = np.diag(autograd(wt.power)(ws_pts))
        if 0:
            plt.plot(ws_lst, wt.power(ws_lst))
            for dpdu, ws in zip(dpdu_lst, ws_pts):
                plot_gradients(wt.power(ws), dpdu, ws, "", 1)

            plt.show()
        dpdu_ref = np.where((ws_pts > 4) & (ws_pts <= 9.8),
                            3 * 3350000 * (ws_pts - 4)**2 / (9.8 - 4)**3,
                            0)

        npt.assert_array_almost_equal(dpdu_lst, dpdu_ref)

    fd_dpdu_lst = np.diag(fd(wt.power)(ws_pts))
    npt.assert_array_almost_equal(fd_dpdu_lst, dpdu_ref, 0)

    cs_dpdu_lst = np.diag(cs(wt.power)(ws_pts))
    npt.assert_array_almost_equal(cs_dpdu_lst, dpdu_ref)


def test_plot_gradients():
    x = np.arange(-3, 4, .1)
    plt.plot(x, x**2)
    plot_gradients(1.5**2, 3, 1.5, "test", 1)
    if 0:
        plt.show()
    plt.close()
