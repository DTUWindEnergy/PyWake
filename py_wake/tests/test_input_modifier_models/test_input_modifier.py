from py_wake.input_modifier_models.input_modifier_model import InputModifierModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.site.xrsite import UniformSite
from py_wake.examples.data.hornsrev1 import V80
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.flow_map import XZGrid
import matplotlib.pyplot as plt
from py_wake.deficit_models.gaussian import ZongGaussianDeficit
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake import np
from py_wake.tests import npt
import pytest
from xarray.core.dataarray import DataArray
from matplotlib.patches import Ellipse
from py_wake.deficit_models.utils import ct2a_mom1d


class DummyFloatingModifier(InputModifierModel):
    def __init__(self, pitch=True, displacement=True):
        self.pitch = pitch
        self.displacement = displacement

    def __call__(self, x_ilk, y_ilk, WD_ilk, WS_eff_ilk, **_):

        theta_ilk = (90 - WD_ilk) / 180 * np.pi
        dw_offset = WS_eff_ilk * 2
        modified_input_dict = {}
        if self.pitch:
            modified_input_dict['tilt_ilk'] = WS_eff_ilk * 2
        if self.displacement:
            modified_input_dict.update({'x_ilk': x_ilk - dw_offset * np.cos(theta_ilk),
                                        'y_ilk': y_ilk - dw_offset * np.sin(theta_ilk)})
        return modified_input_dict


class FloatingV80(V80):
    def plot_yz(self, y, z=None, h=None, types=None, wd=270, yaw=0, tilt=0, normalize_with=1, ax=None):
        """Plot with pitched tower"""
        if z is None:
            z = np.zeros_like(y)
        if types is None:
            types = np.zeros_like(y).astype(int)
        else:
            types = (np.zeros_like(y) + types).astype(int)  # ensure same length as x
        if h is None:
            h = np.zeros_like(y) + self.hub_height(types)
        else:
            h = np.zeros_like(y) + h

        if ax is None:
            ax = plt.gca()
        colors = ['gray', 'k', 'r', 'g', 'k'] * 5

        yaw = np.zeros_like(y) + yaw
        tilt = np.zeros_like(y) + tilt
        y, z, h, D = [v / normalize_with for v in [y, z, h, self.diameter(types)]]
        if isinstance(wd, DataArray):
            wd = wd.values

        for i, (y_, z_, h_, d, t, yaw_, tilt_) in enumerate(
                zip(y, z, h, D, types, yaw, tilt)):
            if len(np.atleast_1d(wd)) == 1:
                wd = np.atleast_1d(wd)[0]
                ty = y_ - np.cos(np.deg2rad(wd)) * d / 20
                ax.plot([ty - h_ * np.sin(np.deg2rad(tilt_)), ty], [z_, z_ + h_], 'k')  # tower (d/20 behind rotor)
                ax.plot([ty, y_], [z_ + h_, z_ + h_], 'k')  # shaft

                circle = Ellipse((y_, h_ + z_), d * np.sin(np.deg2rad(wd - yaw_)),
                                 d, angle=-tilt_, ec=colors[t], fc="None", zorder=32)
                ax.add_artist(circle)
            else:
                ax.plot([y_, y_], [h_ + z_ - d / 2, h_ + z_ + d / 2], 'k')  # rotor
            ax.plot(y_, h_, 'None')

        for t, c in zip(np.unique(types), colors):
            ax.plot([], [], '2', color=c, label=self._names[int(t)])

        for i, (y_, z_, h_, d) in enumerate(zip(y, z, h, D)):
            ax.annotate(i, (y_ + d / 2, z_ + h_ + d / 2), fontsize=7)
        ax.legend(loc=1)
        ax.axis('equal')


def run_floating(wfm_cls, pitch, displacement):
    wfm = wfm_cls(UniformSite(), FloatingV80(), ZongGaussianDeficit(ct2a=ct2a_mom1d),
                  turbulenceModel=CrespoHernandez(ct2a=ct2a_mom1d),
                  deflectionModel=JimenezWakeDeflection(),
                  inputModifierModels=DummyFloatingModifier(pitch=pitch, displacement=displacement))
    sim_res = wfm([0, 300, 600], [0, 0, 0], wd=[270, 280], ws=[5, 10, 15, 20], yaw=0, tilt=0)

    fm_wt2 = sim_res.flow_map(XZGrid(y=0, x=np.linspace(595, 620, 1000), z=70), wd=270, ws=10)
    return sim_res, fm_wt2


@pytest.mark.parametrize('wfm_cls,pitch,displacement,ref', [
    (PropagateDownwind, False, False, [10., 6.51877627, 6.289912]),
    (All2AllIterative, False, False, [10., 6.51877627, 6.289912]),
    (PropagateDownwind, True, False, [10., 7.450013, 7.108338]),
    (All2AllIterative, True, False, [10., 7.450013, 7.108338]),
    (All2AllIterative, False, True, [10., 6.457691, 6.272679]),
    (All2AllIterative, True, True, [10., 7.404978, 7.095028])
])
def test_floating_pitch_modifier(wfm_cls, pitch, displacement, ref):

    sim_res, fm_wt2 = run_floating(wfm_cls, pitch=pitch, displacement=displacement)
    if 0:
        ax1, ax2 = plt.subplots(2, 1)[1]
        sim_res_fixed = run_floating(wfm_cls, pitch=False, displacement=False)[0]
        sim_res_fixed.WS_eff.sel(wd=270, ws=10).squeeze().plot(ax=ax2, label='Fixed')
        sim_res.WS_eff.sel(wd=270, ws=10).squeeze().plot(ax=ax2, label=f'pitch:{pitch}, displacement:{displacement}')
        ax2.legend()
        sim_res.flow_map(XZGrid(y=0), wd=270).plot_wake_map(ax=ax1)
        plt.show()

    # compare WS_eff with ref
    npt.assert_array_almost_equal(sim_res.WS_eff.sel(wd=270, ws=10).squeeze(), ref)
    # compare sim_res, wt2 with flowmap just before wt2
    if 'ws' in sim_res.x.dims:
        i = np.searchsorted(fm_wt2.x, sim_res.x.sel(ws=10, wd=270)[2]) - 1
    else:
        i = np.searchsorted(fm_wt2.x, sim_res.x[2]) - 1
    npt.assert_array_almost_equal(sim_res.WS_eff.sel(ws=10, wd=270, wt=2),
                                  fm_wt2.WS_eff.isel(x=i), 3)
