import numpy as np
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windrose,\
    read_iea37_windturbine, read_iea37_windfarm, read_iea37_windturbine_deprecated
from py_wake.tests import npt
from py_wake.utils.gradients import cs


def test_read_iea37_windrose():
    wdir, wsp, freq = read_iea37_windrose(iea37_path + "iea37-windrose.yaml")
    assert wdir[1] == 22.5
    assert wsp[0] == 9.8
    assert freq[1] == .024


def test_read_iea_windturbine():
    wt_id, hubheight, diameter, power, ct, dpower, dct = read_iea37_windturbine_deprecated(
        iea37_path + 'iea37-335mw.yaml')
    assert wt_id == "3.35MW"
    assert hubheight == 110
    assert diameter == 130

    u = np.arange(30)
    p_r = 3350000
    npt.assert_array_almost_equal([0, 1 / 5.8**3 * p_r, p_r, p_r, 0], power([4, 5, 9.8, 25, 25.1]))
    ct_ = 4 * 1 / 3 * (1 - 1 / 3)
    npt.assert_array_almost_equal([0, ct_, ct_, 0], ct([3.9, 4, 25, 25.1]))
    npt.assert_almost_equal(dpower(7), cs(power)(7))
    npt.assert_equal(dct(7), 0)
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(u, power(u) / 1e6)
        plt.plot(u, ct(u))
        plt.show()


def test_read_iea_windfarm():
    x, y, aep = read_iea37_windfarm(iea37_path + 'iea37-ex16.yaml')
    assert len(x) == 16
    assert x[1] == 650
    assert y[2] == 618.1867
    assert aep[0] == 366941.57116
    assert aep[1][0] == 9444.60012
