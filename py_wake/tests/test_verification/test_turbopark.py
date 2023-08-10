import pytest
import matplotlib.pyplot as plt
import numpy as np
from py_wake.literature import Nygaard_2022
from py_wake.site import XRSite
from py_wake.site._site import UniformSite
from py_wake.site.shear import PowerShear
from py_wake.tests import npt
from py_wake.wind_turbines._wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
import xarray as xr
from py_wake.flow_map import XYGrid


@pytest.fixture(scope='module')
def windTurbines():
    u = np.arange(0, 25.5, .5)
    po = [0, 0, 0, 0, 5, 15, 37, 73, 122, 183, 259, 357, 477, 622, 791, 988, 1212, 1469, 1755, 2009, 2176, 2298, 2388, 2447, 2485, 2500, 2500, 2500,
          2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500,
          2500, 2500, 2500, 2500, 2500, 2500, 2500, 0]
    ct = [0, 0, 0, 0, 0.78, 0.77, 0.78, 0.78, 0.77, 0.77, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.77, 0.77, 0.77, 0.76, 0.73, 0.7, 0.68, 0.52, 0.42,
          0.36, 0.31, 0.27, 0.24, 0.22, 0.19, 0.18, 0.16, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06,
          0.05, 0.05, 0.05, 0.04, 0]

    wt1 = WindTurbine(name="Ørsted1", diameter=120, hub_height=100, powerCtFunction=PowerCtTabular(u, po, 'kw', ct))

    u2 = np.arange(0, 27)
    pow2 = [0, 0, 0, 0, 54, 144, 289, 474, 730, 1050, 1417, 1780, 2041, 2199,
            2260, 2292, 2299, 2300, 2300, 2300, 2300, 2300, 2300, 2300, 2300, 2300, 0]
    ct2 = [0, 0, 0, 0, 0.94, 0.82, 0.76, 0.68, 0.86, 0.83, 0.77, 0.68, 0.66, 0.52,
           0.47, 0.41, 0.38, 0.34, 0.27, 0.26, 0.23, 0.22, 0.22, 0.2, 0.16, 0.17, 0]
    wt2 = WindTurbine(name="Ørsted2", diameter=80, hub_height=70, powerCtFunction=PowerCtTabular(u2, pow2, 'kw', ct2))
    return WindTurbines.from_WindTurbine_lst([wt1, wt2])


@pytest.fixture(scope='module')
def gradient_site():
    x_pt = np.arange(-100, 3100, 100)
    Y_pt, X_pt = np.meshgrid(x_pt, x_pt)
    grad = ((X_pt - 5)**2 + (Y_pt)**2) * 10**(-8) + 1
    speedup = grad / grad[1, 1]
    ds = xr.Dataset({'Speedup': (('x', 'y'), speedup), 'P': 1}, coords={'x': x_pt, 'y': x_pt})
    return XRSite(ds=ds, shear=PowerShear(h_ref=90, alpha=.1))


@pytest.fixture()
def kwargs():
    u0 = [6, 10, 14]
    wd = [270]
    ti0 = [0.09, .1, .11]
    y, x = [v.flatten() for v in np.meshgrid(np.arange(4) * 120 * 6, np.arange(4) * 120 * 6)]
    return dict(x=x, y=y, ws=u0, wd=wd, TI=ti0)


def test_example1_single_row(windTurbines, kwargs):

    site = UniformSite(shear=PowerShear(h_ref=90, alpha=.1))

    wfm = Nygaard_2022(site, windTurbines)
    kwargs['x'] = kwargs['x'][::4]
    kwargs['y'] = kwargs['y'][::4]
    sim_res = wfm(**kwargs)

    # pow_waked from matlab
    power_ref = [[495.429647093727, 2201.84387293603, 2500],
                 [165.681333834342, 938.265716039082, 2500],
                 [105.264701735512, 607.531414199904, 2485.83000221364],
                 [71.8241034924664, 446.254939675813, 2408.31291455241]]

    npt.assert_allclose(sim_res.Power.squeeze() / 1000, power_ref, rtol=1e-5)

    # ws_waked from matlab
    ws_eff_ref = [[6.06355050721975, 10.1059175120329, 14.1482845168461],
                  [4.35804371995363, 7.37377085289107, 12.8635577397301],
                  [3.82923165036237, 6.45010832482725, 12.0276667404546],
                  [3.48366810406203, 5.87189558198255, 11.1721433436644]]

    npt.assert_allclose(sim_res.WS_eff.squeeze(), ws_eff_ref, rtol=1e-6)


def test_example1(windTurbines, kwargs):

    site = UniformSite(shear=PowerShear(h_ref=90, alpha=.1))

    wfm = Nygaard_2022(site, windTurbines)
    sim_res = wfm(**kwargs)

    # pow_waked from https://github.com/OrstedRD/TurbOPark/blob/main/TurbOParkExamples.mlx
    power_ref = [[495.429647093727, 2201.84387293603, 2500],
                 [495.429647093727, 2201.84387293603, 2500],
                 [495.429647093727, 2201.84387293603, 2500],
                 [495.429647093727, 2201.84387293603, 2500],
                 [165.681333834342, 938.265716039082, 2500],
                 [165.681333834342, 938.265716039082, 2500],
                 [165.681333834342, 938.265716039082, 2500],
                 [165.681333834342, 938.265716039082, 2500],
                 [105.264701735512, 607.531414199904, 2485.83000221364],
                 [105.264701735512, 607.531414199904, 2485.83000221364],
                 [105.264701735512, 607.531414199904, 2485.83000221364],
                 [105.264701735512, 607.531414199904, 2485.83000221364],
                 [71.8241034924664, 446.254939675813, 2408.31291455241],
                 [71.8241034924664, 446.254939675813, 2408.31291455241],
                 [71.8241034924664, 446.254939675813, 2408.31291455241],
                 [71.8241034924664, 446.254939675813, 2408.31291455241]]

    npt.assert_allclose(sim_res.Power.squeeze() / 1000, power_ref, rtol=1e-5)

    # ws_waked from https://github.com/OrstedRD/TurbOPark/blob/main/TurbOParkExamples.mlx
    ws_eff_ref = [[6.06355050721975, 10.1059175120329, 14.1482845168461],
                  [6.06355050721975, 10.1059175120329, 14.1482845168461],
                  [6.06355050721975, 10.1059175120329, 14.1482845168461],
                  [6.06355050721975, 10.1059175120329, 14.1482845168461],
                  [4.35804371995363, 7.37377085289107, 12.8635577397301],
                  [4.35804371995363, 7.37377085289107, 12.8635577397301],
                  [4.35804371995363, 7.37377085289107, 12.8635577397301],
                  [4.35804371995363, 7.37377085289107, 12.8635577397301],
                  [3.82923165036237, 6.45010832482725, 12.0276667404546],
                  [3.82923165036237, 6.45010832482725, 12.0276667404546],
                  [3.82923165036237, 6.45010832482725, 12.0276667404546],
                  [3.82923165036237, 6.45010832482725, 12.0276667404546],
                  [3.48366810406203, 5.87189558198255, 11.1721433436644],
                  [3.48366810406203, 5.87189558198255, 11.1721433436644],
                  [3.48366810406203, 5.87189558198255, 11.1721433436644],
                  [3.48366810406203, 5.87189558198255, 11.1721433436644]]

    npt.assert_allclose(sim_res.WS_eff.squeeze(), ws_eff_ref, rtol=1e-6)


def test_example2_single_row(windTurbines, gradient_site, kwargs):
    type = np.array([1, 1, 1, 1])
    kwargs['x'] = kwargs['x'][::4]
    kwargs['y'] = kwargs['y'][::4]
    wfm = Nygaard_2022(gradient_site, windTurbines)
    sim_res = wfm(**kwargs, type=type)

    # pow_waked from https://github.com/OrstedRD/TurbOPark/blob/main/TurbOParkExamples.mlx
    power_ref = [[267.408098975358, 1325.91692326387, 2238.805191408],
                 [87.8716269861529, 549.915174301573, 1860.30802639162],
                 [56.3363587519906, 421.529493247156, 1453.25377990821],
                 [50.9865832982385, 379.447075562596, 1216.18489248015]]

    npt.assert_allclose(sim_res.Power.squeeze() / 1000, power_ref, rtol=1e-5)

    # ws_waked from https://github.com/OrstedRD/TurbOPark/blob/main/TurbOParkExamples.mlx
    ws_eff_ref = [[5.85109033776109, 9.75181722960182, 13.6525441214426],
                  [4.37635141095725, 7.29654364961552, 11.3076935877073],
                  [4.02595954168878, 6.71637563917382, 10.0998726719234],
                  [3.94419598700442, 6.48890311114917, 9.45281987051811]]

    npt.assert_allclose(sim_res.WS_eff.squeeze(), ws_eff_ref, rtol=1e-6)


def test_example2(windTurbines, gradient_site, kwargs):
    type = np.array([1, 1, 0, 0] * 4)

    wfm = Nygaard_2022(gradient_site, windTurbines)
    sim_res = wfm(**kwargs, type=type)

    # pow_waked from https://github.com/OrstedRD/TurbOPark/blob/main/TurbOParkExamples.mlx
    power_ref = [[267.408098975358, 1325.91692326387, 2238.805191408],
                 [271.8198199871, 1344.52728661225, 2243.13577732067],
                 [531.934637441146, 2253.03477894046, 2500],
                 [577.513122499193, 2311.97909224055, 2500],
                 [87.8716269861529, 549.915174301573, 1860.30802639162],
                 [90.029536518109, 560.285573640592, 1879.96345820567],
                 [179.445496763605, 1030.67640805119, 2500],
                 [195.741429267846, 1133.76926324207, 2500],
                 [56.3363587519906, 421.529493247156, 1453.25377990821],
                 [57.8305559469565, 426.318770082901, 1476.12985730185],
                 [120.791650152783, 701.169901423848, 2500],
                 [132.472056081865, 765.870852417475, 2500],
                 [50.9865832982385, 379.447075562596, 1216.18489248015],
                 [51.592223376867, 385.115650375366, 1238.96287545131],
                 [94.521082069401, 557.67685933973, 2496.89121383494],
                 [103.321661723677, 606.468580599153, 2500]]

    npt.assert_allclose(sim_res.Power.squeeze() / 1000, power_ref, rtol=1e-5)

    # ws_waked from https://github.com/OrstedRD/TurbOPark/blob/main/TurbOParkExamples.mlx
    ws_eff_ref = [[5.85109033776109, 9.75181722960182, 13.6525441214426],
                  [5.88151599991104, 9.80252666651839, 13.7235373331257],
                  [6.18942978427981, 10.315716307133, 14.4420028299862],
                  [6.34659697413515, 10.5776616235586, 14.808726272982],
                  [4.37635141095725, 7.29654364961552, 11.3076935877073],
                  [4.40032818353454, 7.33705302203356, 11.3830017555773],
                  [4.47086472757053, 7.59525983939998, 13.2633760517144],
                  [4.58382519255162, 7.82537781973676, 13.7197242744724],
                  [4.02595954168878, 6.71637563917382, 10.0998726719234],
                  [4.04256173274396, 6.74226362206974, 10.1628921688756],
                  [3.9876698995182, 6.73423047758535, 12.7262841310504],
                  [4.08583652526119, 6.92565340951916, 13.2610220831519],
                  [3.94419598700442, 6.48890311114917, 9.45281987051811],
                  [3.95541154401606, 6.51954405608306, 9.51488521921339],
                  [3.71960287825919, 6.27819606668872, 12.396373794498],
                  [3.80940471146609, 6.44644338137639, 13.0445573540727]]

    npt.assert_allclose(sim_res.WS_eff.squeeze(), ws_eff_ref, rtol=1e-6)


def test_WS_jlk_flowmap(windTurbines, gradient_site, kwargs):

    wfm = Nygaard_2022(gradient_site, windTurbines)
    sim_res = wfm(**kwargs)
    fm = sim_res.flow_map(XYGrid(x=np.linspace(-100, 3000), y=np.linspace(-100, 3000)), ws=10)
    fm.plot_wake_map()
    if 0:
        plt.show()
