import pytest

import numpy as np
from py_wake.tests import npt
from py_wake.site._site import UniformSite
from py_wake.site.shear import LogShear
from py_wake.literature.cumulative_sum import nrel5mw
from py_wake.literature import CumulativeWake


@pytest.fixture(scope='module')
def setup():
    u_h, ti_h = 8., 0.1
    wt = nrel5mw()
    site = UniformSite(shear=LogShear(h_ref=wt.hub_height(), z0=.15))

    return site, wt, u_h, ti_h


def test_aligned(setup):

    site, wt, u_h, ti_h = setup

    wfm = CumulativeWake(site, wt)
    y, x = [v.flatten() for v in np.meshgrid(np.arange(3) * wt.diameter() * 3, np.arange(5) * wt.diameter() * 5)]
    sim_res = wfm(x, y, ws=u_h, wd=270., TI=ti_h)

    i_wt = [1, 4, 7, 10, 13]
    eff_ref = np.array([0.5065040650406504, 0.22886178861788614, 0.2207317073170732, 0.22479674796747962, 0.2219512195121951])
    diff = np.squeeze(sim_res.Power.isel(wt=i_wt)) / (0.5 * 1.225 * u_h**3 * (wt.diameter()**2 * np.pi / 4.)) - eff_ref

    npt.assert_allclose(diff, [0.0163493, -0.00182291, 0.00418705, 0.00489763, 0.00455533], rtol=1e-5)


def test_slanted(setup):

    site, wt, u_h, ti_h = setup

    wfm = CumulativeWake(site, wt)
    y, x = [v.flatten() for v in np.meshgrid(np.arange(3) * wt.diameter() * 3, np.arange(5) * wt.diameter() * 5)]
    y_off = y.copy()
    for i in range(5):
        y_off[3 * i: 3 * (i + 1)] += 0.75 * i * wt.diameter()
    sim_res = wfm(x, y_off, ws=u_h, wd=270., TI=ti_h)

    i_wt = [1, 4, 7, 10, 13]
    eff_ref = np.array([0.5068437180796731, 0.40510725229826355, 0.38835546475995913, 0.32420837589376916, 0.2686414708886619])
    diff = np.squeeze(sim_res.Power.isel(wt=i_wt)) / (0.5 * 1.225 * u_h**3 * (wt.diameter()**2 * np.pi / 4.)) - eff_ref

    npt.assert_allclose(diff, [0.01600965, 0.04234562, 0.03718484, 0.0274449, 0.01951109], rtol=1e-5)
