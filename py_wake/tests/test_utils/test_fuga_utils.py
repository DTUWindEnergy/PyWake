import os

import pytest

from py_wake import np
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.utils import fuga_utils
from py_wake.utils.fuga_utils import dat2netcdf
import xarray as xr


@pytest.mark.parametrize('name', ['Z0=0.03000000Zi=00401Zeta0=0.00E+00',
                                  'Z0=0.00408599Zi=00400Zeta0=0.00E+00'])
def test_dat2netcdf(name):
    ds = dat2netcdf(tfp + f'fuga/2MW/{name}')
    ref = xr.load_dataset(tfp + f"fuga/2MW/{name}.nc")
    assert ds == ref
    os.remove(ds.filename)


def test_ti_z0():
    ti = np.array([0.06, .1, .18, 0.06, .1, .18])
    zhub = np.array([70, 70, 70, 100, 100, 100])
    npt.assert_array_almost_equal(fuga_utils.ti(fuga_utils.z0(ti, zhub), zhub), ti)
