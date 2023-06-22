import os

import pytest

from py_wake import np
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.utils import fuga_utils
from py_wake.utils.fuga_utils import dat2netcdf, phi, psi
import xarray as xr
import matplotlib.pyplot as plt


@pytest.mark.parametrize('name', ['Z0=0.03000000Zi=00401Zeta0=0.00E+00',
                                  'Z0=0.00408599Zi=00400Zeta0=0.00E+00'])
def test_dat2netcdf(name):
    ds = dat2netcdf(tfp + f'fuga/2MW/{name}')
    ref = xr.load_dataset(tfp + f"fuga/2MW/{name}.nc")
    assert ds == ref
    os.remove(ds.filename)


@pytest.mark.parametrize('zeta0', [-6e-7, 0, 6e-7])
def test_ti_z0(zeta0):

    ti_ref = np.array([0.09, .1, .18, 0.09, .1, .18])
    zhub = np.array([70, 70, 70, 100, 100, 100])
    ti = fuga_utils.ti(fuga_utils.z0(ti_ref, zhub, zeta0, z0_limit=0), zhub, zeta0)
    if 0:
        plt.plot(ti_ref - ti)
        plt.show()
    npt.assert_array_almost_equal(ti, ti_ref, 15)


def test_phi():
    zeta = np.linspace(-.1, .1, 11)
    print(list(np.round(phi(zeta), 3)))
    if 0:
        plt.plot(zeta, phi(zeta))
        plt.show()
    npt.assert_array_almost_equal(phi(zeta), [0.764, 0.792, 0.825, 0.867, 0.922, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 3)


def test_psi():
    zeta = np.linspace(-.1, .1, 51)
    print(list(np.round(psi(zeta[::5]), 3)))
    print(list(np.round(psi(zeta[::5], 'Wilson'), 3)))
    if 0:
        plt.plot(zeta, psi(zeta))
        plt.plot(zeta, psi(zeta, 'Wilson'), label='Wilson')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal(psi(zeta[::5]),
                                  [0.326, 0.276, 0.221, 0.159, 0.087, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5], 3)
    npt.assert_array_almost_equal(psi(zeta[::5], 'Wilson'),
                                  [2.541, 2.488, 2.427, 2.355, 2.261, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5], 3)


def test_z0_from_TI():
    if 0:
        TI = np.linspace(3, 10, 100) / 100
        zref = 70
        plt.figure()

        z0_stable = fuga_utils.z0(TI, zref, 6e-7)
        z0_neutral = fuga_utils.z0(TI, zref, 0.0)
        z0_unstable = fuga_utils.z0(TI, zref, -6e-7)
        plt.plot(TI * 100, z0_stable, label='stable')
        plt.plot(TI * 100, z0_neutral, label='neutral')
        plt.plot(TI * 100, z0_unstable, label='unstable')

        plt.xlabel('ti [%]')
        plt.ylabel('z0')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal(fuga_utils.z0([.06, .12, .06, .12, .06, .12], 70, [-6e-7, -6e-7, 0, 0, 6e-7, 6e-7]),
                                  [1.00000e-05, 1.66251e-02, 1.00000e-05, 1.68259e-02, 7.26921e-05, 1.70345e-02], 5)
