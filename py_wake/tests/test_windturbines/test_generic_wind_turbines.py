from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine, GenericTIRhoWindTurbine
from py_wake.examples.data import wtg_path
from py_wake.examples.data.dtu10mw import DTU10MW
import numpy as np
import matplotlib.pyplot as plt
from py_wake.tests import npt
import pytest
from py_wake.deficit_models.noj import NOJ
from py_wake.site.xrsite import XRSite


def test_GenericWindTurbine():
    for ref, ti, cut_in, cut_out, p_tol, ct_tol in [(V80(), .1, 4, None, 0.03, .14),
                                                    (WindTurbine.from_WAsP_wtg(wtg_path +
                                                                               "Vestas V112-3.0 MW.wtg"), .05, 3, 25, 0.035, .07),
                                                    (DTU10MW(), .05, 4, 25, 0.06, .12)]:

        power_norm = ref.power(np.arange(10, 20)).max()
        wt = GenericWindTurbine('Generic', ref.diameter(), ref.hub_height(), power_norm / 1e3,
                                turbulence_intensity=ti, ws_cutin=cut_in, ws_cutout=cut_out)

        if 0:
            u = np.arange(0, 30, .1)
            p, ct = wt.power_ct(u)
            plt.plot(u, p / 1e6, label='Generic')

            plt.plot(u, ref.power(u) / 1e6, label=ref.name())

            plt.ylabel('Power [MW]')
            plt.legend(loc='center left')
            ax = plt.twinx()
            ax.plot(u, ct, '--')
            ax.plot(u, ref.ct(u), '--')

            ax.set_ylim([0, 1])
            plt.ylabel('Ct')
            plt.show()

        u = np.arange(5, 25)
        p, ct = wt.power_ct(u)
        p_ref, ct_ref = ref.power_ct(u)
        # print(np.abs(p_ref - p).max() / power_norm)
        npt.assert_allclose(p, p_ref, atol=power_norm * p_tol)
        # print(np.abs(ct_ref - ct).max())
        npt.assert_allclose(ct, ct_ref, atol=ct_tol)


@pytest.mark.parametrize(['power_idle', 'ct_idle'], [(0, 0), (100, .1)])
def test_GenericWindTurbine_cut_in_out(power_idle, ct_idle):
    ref = V80()
    power_norm = ref.power(15)

    wt = GenericWindTurbine('Generic', ref.diameter(), ref.hub_height(), power_norm / 1e3,
                            turbulence_intensity=0, ws_cutin=3, ws_cutout=25, power_idle=power_idle, ct_idle=ct_idle)
    if 0:
        u = np.arange(0, 30, .1)
        p, ct = wt.power_ct(u)
        plt.plot(u, p / 1e6, label='Generic')

        plt.plot(u, ref.power(u) / 1e6, label=ref.name())

        plt.ylabel('Power [MW]')
        plt.legend()
        ax = plt.twinx()
        ax.plot(u, ct, '--')
        ax.plot(u, ref.ct(u), '--')
        plt.ylabel('Ct')
        plt.show()
    assert wt.ct(2.9) == ct_idle
    assert wt.power(2.9) == power_idle
    assert wt.ct(25.1) == ct_idle
    assert wt.power(25.1) == power_idle


def test_GenericTIRhoWindTurbine():
    wt = GenericTIRhoWindTurbine('2MW', 80, 70, 2000, )
    ws_lst = [11, 11, 11]
    ti_lst = [0, .1, .2]
    p11, ct11 = wt.power_ct(ws=ws_lst, TI_eff=ti_lst, Air_density=1.225)
    p11 /= 1e6
    if 0:
        u = np.arange(3, 28, .1)
        ax1 = plt.gca()
        ax2 = plt.twinx()
        for ti in ti_lst:
            p, ct = wt.power_ct(u, TI_eff=ti, Air_density=1.225)
            ax1.plot(u, p / 1e6, label='TI=%f' % ti)
            ax2.plot(u, ct, '--')
        ax1.plot(ws_lst, p11, '.')
        ax2.plot(ws_lst, ct11, 'x')
        print(p11.tolist())
        print(ct11.tolist())
        ax1.legend()
        ax1.set_ylabel('Power [MW]')
        ax2.set_ylabel('Ct')
        plt.show()
    npt.assert_array_almost_equal([1.833753, 1.709754, 1.568131], p11)
    npt.assert_array_almost_equal([0.793741, 0.694236, 0.544916], ct11)

    ws_lst = [10] * 3
    rho_lst = [0.9, 1.225, 1.5]
    p10, ct10 = wt.power_ct(ws=ws_lst, TI_eff=0.1, Air_density=rho_lst)
    p10 /= 1e6
    if 0:
        u = np.arange(3, 28, .1)
        ax1 = plt.gca()
        ax2 = plt.twinx()
        for rho in rho_lst:
            p, ct = wt.power_ct(u, TI_eff=0.1, Air_density=rho)
            ax1.plot(u, p / 1e6, label='Air density=%f' % rho)
            ax2.plot(u, ct, '--')
        ax1.plot(ws_lst, p10, '.')
        ax2.plot(ws_lst, ct10, 'x')
        print(p10.tolist())
        print(ct10.tolist())
        ax1.legend()
        ax1.set_ylabel('Power [MW]')
        ax2.set_ylabel('Ct')
        plt.show()
    npt.assert_array_almost_equal([1.040377569594173, 1.3934596754744593, 1.6322037609434554], p10)
    npt.assert_array_almost_equal([0.7987480617157162, 0.7762418395479502, 0.7282996179383272], ct10)
