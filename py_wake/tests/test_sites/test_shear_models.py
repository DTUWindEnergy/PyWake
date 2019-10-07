from py_wake.site.shear import PowerShear
import numpy as np
from py_wake.tests import npt


def test_power_shear():
    shear = PowerShear(70, alpha=[.1, .2])
    h_lst = np.arange(10, 100, 10)
    u1, u2 = np.array([shear(WS_ilk=[[[10]], [[10]]], WD_ilk=[[[0]], [[180]]], h_i=[h, h])
                       for h in h_lst])[:, :, 0, 0].T
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(u1, h_lst, label='alpha=0.1')
        plt.plot((h_lst / 70)**0.1 * 10, h_lst, ':')
        plt.plot(u2, h_lst, label='alpha=0.2')
        plt.plot((h_lst / 70)**0.2 * 10, h_lst, ':')
        plt.show()
    npt.assert_array_almost_equal(u1, [8.23, 8.82, 9.19, 9.46, 9.67, 9.85, 10., 10.13, 10.25], 2)
    npt.assert_array_almost_equal(u2, [6.78, 7.78, 8.44, 8.94, 9.35, 9.7, 10., 10.27, 10.52], 2)


if __name__ == '__main__':
    test_power_shear()
