import matplotlib.pyplot as plt
from py_wake import np
from py_wake.deficit_models.fuga import FugaDeficit, FugaBlockage
from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.flow_map import HorizontalGrid
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
import xarray as xr
from py_wake.deficit_models.deficit_model import XRLUTDeficitModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.deficit_models.noj import NOJDeficit, NOJ
from numpy import newaxis as na
from py_wake.superposition_models import SquaredSum
from py_wake.utils.gradients import cabs


def test_noj():

    # Make NOJ dataarray look-up table
    noj = NOJDeficit()
    x = np.linspace(-3000, 3000, 500)
    y = np.linspace(-3000, 3000, 1000)
    ct = np.linspace(0.1, 8 / 9, 50)
    DW, CW = np.meshgrid(x, y)
    D_src_il = np.array([[80]])
    dw_ijlk = DW.flatten()[na, :, na, na]
    cw_ijlk = CW.flatten()[na, :, na, na]
    ct_ilk = ct[na, na]
    wake_radius_ijlk = noj.wake_radius(D_src_il=D_src_il, dw_ijlk=dw_ijlk)
    output = noj.calc_deficit(ct_ilk=ct_ilk, D_src_il=D_src_il, wake_radius_ijl=wake_radius_ijlk[:, :, :, 0],
                              dw_ijlk=dw_ijlk, cw_ijlk=np.abs(cw_ijlk), WS_ilk=np.array([[[1]]]))
    da = xr.DataArray(output.reshape(DW.shape + ct.shape),
                      coords={'cw_ijlk': y, 'dw_ijlk': x, 'ct_ilk': ct})
    xrdeficit = XRLUTDeficitModel(da, use_effective_ws=False)

    # compare LUT based deficit model with original NOJDeficit
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    site = UniformSite([1, 0, 0, 0], ti=0.075)

    wfm = PropagateDownwind(site, wts, wake_deficitModel=xrdeficit, superpositionModel=SquaredSum())
    wfm_noj = NOJ(site, wts, rotorAvgModel=None)
    res = wfm(x=wt_x, y=wt_y, wd=[30], ws=[10])
    res_noj = wfm_noj(x=wt_x, y=wt_y, wd=[30], ws=[10])

    if 0:
        plt.plot(res_noj.WS_eff.squeeze())
        plt.plot(res.WS_eff.squeeze())
        plt.show()

    npt.assert_array_almost_equal(res.WS_eff, res_noj.WS_eff, 3)
    npt.assert_array_almost_equal(res.CT, res_noj.CT, 6)

    x_j = np.linspace(-1500, 1500, 400)
    y_j = np.linspace(-1500, 1500, 30)

    sim_res, sim_res_ref = [wfm(wt_x, wt_y, wd=[30], ws=[10]) for wfm in [wfm, wfm_noj]]
    flow_map, flow_map_ref = [s.flow_map(HorizontalGrid(x_j, y_j)) for s in [sim_res, sim_res_ref]]
    X, Y = flow_map.XY
    Z, Z_ref = [fm.WS_eff_xylk[:, :, 0, 0] for fm in [flow_map, flow_map_ref]]

    if 0:
        flow_map.plot_wake_map(levels=np.arange(6, 10.5, .1))
        plt.plot(X[14, 100:400:10], Y[14, 100:400:10], '.-')
        plt.figure()
        plt.plot(X[0], Z[14, :], label="Z=70m")
        plt.plot(X[0], Z_ref[14, :], label="ref")

        plt.plot(X[0, 100:400:10], Z[14, 100:400:10], '.')
        print(list(np.round(Z.data[14, 100:400:10], 4)))
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(Z[14, 100:400:10], Z_ref[14, 100:400:10], 1)


def test_fuga():
    wt_x = [-250, 600, -500, 0, 500, -250, 250]
    wt_y = [433, 300, 0, 0, 0, -433, -433]
    wts = HornsrevV80()

    site = UniformSite([1, 0, 0, 0], ti=0.075)
    path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc'

    x, y, z, du = FugaDeficit(path).load()
    da = xr.DataArray(du, coords={'z': z, 'y': y, 'x': x})

    def get_input(dw_ijlk, hcw_ijlk, h_ilk, dh_ijlk, **_):
        # map input to interpolation coordinates, (z,y,x)
        return [(h_ilk[:, na, :] + dh_ijlk), cabs(hcw_ijlk), dw_ijlk]

    def get_output(output_ijlk, ct_ilk, WS_eff_ilk, WS_ilk, **_):
        # scale interpolated output with ct and WS
        return output_ijlk * (ct_ilk * WS_eff_ilk**2 / WS_ilk)[:, na]

    xrdeficit = XRLUTDeficitModel(da, get_input=get_input, get_output=get_output)
    wfm = All2AllIterative(site, wts, wake_deficitModel=xrdeficit, blockage_deficitModel=xrdeficit)

    wfm_ref = FugaBlockage(path, site, wts)
    res, res_ref = [w(x=wt_x, y=wt_y, wd=[30], ws=[10]) for w in [wfm, wfm_ref]]

    if 0:
        plt.plot(res_ref.WS_eff.squeeze())
        plt.plot(res.WS_eff.squeeze())
        plt.show()

    npt.assert_array_almost_equal(res.WS_eff, res_ref.WS_eff, 8)
    npt.assert_array_almost_equal(res.CT, res_ref.CT, 8)

    x_j = np.linspace(-1500, 1500, 500)
    y_j = np.linspace(-1500, 1500, 30)

    flow_map70 = res.flow_map(HorizontalGrid(x_j, y_j, h=70))
    flow_map70_ref = res_ref.flow_map(HorizontalGrid(x_j, y_j, h=70))
    flow_map73 = res.flow_map(HorizontalGrid(x_j, y_j, h=73))
    flow_map73_ref = res_ref.flow_map(HorizontalGrid(x_j, y_j, h=73))
    X, Y = flow_map70.XY
    Z70 = flow_map70.WS_eff_xylk[:, :, 0, 0]
    Z70_ref = flow_map70_ref.WS_eff_xylk[:, :, 0, 0]
    Z73 = flow_map73.WS_eff_xylk[:, :, 0, 0]
    Z73_ref = flow_map73_ref.WS_eff_xylk[:, :, 0, 0]

    if 0:
        flow_map70.plot_wake_map(levels=np.arange(6, 10.5, .1))
        plt.plot(X[14, 100:400:10], Y[14, 100:400:10], '.-')
        plt.figure()
        plt.plot(X[0], Z70[14, :], label="Z=70m")
        plt.plot(X[0], Z73[14, :], label="Z=73m")
        plt.plot(X[0, 100:400:10], Z70[14, 100:400:10], '.')
        plt.plot(X[0, 100:400:10], Z70_ref[14, 100:400:10], '.')

        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(Z70[14, 100:400:10], Z70_ref[14, 100:400:10], 4)
    npt.assert_array_almost_equal(Z73[14, 100:400:10], Z73_ref[14, 100:400:10], 4)
