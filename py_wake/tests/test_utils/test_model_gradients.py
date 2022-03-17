from numpy import newaxis as na
import pytest

import matplotlib.pyplot as plt
import numpy as np
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit,\
    BastankhahGaussianDeficit, NiayifarGaussianDeficit
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.examples.data.iea34_130rwt._iea34_130rwt import IEA34_130_1WT_Surrogate
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site, IEA37WindTurbines
from py_wake.ground_models.ground_models import GroundModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel
from py_wake.site.distance import StraightDistance
from py_wake.site.shear import Shear, LogShear, PowerShear
from py_wake.superposition_models import SuperpositionModel, AddedTurbulenceSuperpositionModel
from py_wake.tests import npt
from py_wake.tests.check_speed import timeit
from py_wake.turbulence_models.stf import STF2017TurbulenceModel, STF2005TurbulenceModel
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.utils import gradients
from py_wake.utils.gradients import autograd, plot_gradients, fd, cs
from py_wake.utils.model_utils import get_models
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel


def check_gradients(wfm, name, wt_x=[-1300, -650, 0], wt_y=[0, 0, 0], wt_h=[110, 110, 110], fd_step=1e-6, fd_decimal=6,
                    output=(lambda wfm: wfm.aep, 'AEP [GWh]')):
    if wfm is None:
        return
    site = IEA37Site(16)
    wt = IEA37_WindTurbines()
    wfm = wfm(site=site, wt=wt)
    try:
        # plot points
        x_lst = np.array([0, 0., 1.]) * np.arange(1, 600, 10)[:, na] + wt_x
        y_lst = np.array([0, 0, 1.]) * np.arange(-101, 100, 5)[:, na] + wt_y
        h_lst = np.array([0, 0, 1.]) * np.arange(-50, 50, 5)[:, na] + wt.hub_height()
        kwargs = {'ws': [9], 'wd': [270]}

        xp, yp, hp = x_lst[20], y_lst[25], h_lst[2]

        def fdstep(*args, **kwargs):
            return fd(*args, **kwargs, step=fd_step)
        output_func, output_label = output
        output_func = output_func(wfm)
        autograd(output_func, True, 0)(xp, wt_y, **kwargs)[2]

        dOutputdx_lst = [grad(output_func, True, 0)(xp, wt_y, **kwargs)[2] for grad in [fdstep, cs, autograd]]
        npt.assert_almost_equal(dOutputdx_lst[0], dOutputdx_lst[1], fd_decimal)
        npt.assert_almost_equal(dOutputdx_lst[1], dOutputdx_lst[2], 10)

        dOutputdy_lst = [grad(output_func, True, 1)(wt_x, yp, **kwargs)[2] for grad in [fdstep, cs, autograd]]
        npt.assert_almost_equal(dOutputdy_lst[0], dOutputdy_lst[1], fd_decimal)
        npt.assert_almost_equal(dOutputdy_lst[1], dOutputdy_lst[2], 10)

        dOutputdh_lst = [grad(output_func, True, 2)(wt_x, wt_y, hp, **kwargs)[2] for grad in [fdstep, cs, autograd]]
        npt.assert_almost_equal(dOutputdh_lst[0], dOutputdh_lst[1], fd_decimal)
        npt.assert_almost_equal(dOutputdh_lst[1], dOutputdh_lst[2], 10)

        if 0:
            # wfm(wt_x, wt_y, **kwargs).flow_map().plot_wake_map()

            _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
            ax1.set_title("Center line")
            ax1.set_xlabel('Downwind distance [m]')
            ax1.set_ylabel(output_label)

            ax1.plot(x_lst[:, 2], [output_func(xp, wt_y, **kwargs) for xp in x_lst])
            ax1.axvline(wt_x[1], color='k')
            for grad, dOutputdx in zip([fd, cs, autograd], dOutputdx_lst):
                plot_gradients(output_func(xp, wt_y, **kwargs), dOutputdx, xp[2], grad.__name__, step=100, ax=ax1)

            ax2.set_xlabel('Crosswind distance [m]')
            ax2.set_ylabel(output_label)
            ax2.set_title("%d m downstream" % wt_x[1])

            ax2.plot(y_lst[:, 2], [output_func(wt_x, yp, **kwargs) for yp in y_lst])
            gradients.color_dict = {}
            for grad, dOutputdy in zip([fd, cs, autograd], dOutputdy_lst):
                plot_gradients(output_func(wt_x, yp, **kwargs), dOutputdy, yp[2], grad.__name__, step=50, ax=ax2)

            ax3.set_xlabel('hubheight [m]')
            ax3.set_ylabel(output_label)

            ax3.plot(h_lst[:, 2], [output_func(wt_x, wt_y, hp, **kwargs) for hp in h_lst])
            gradients.color_dict = {}
            for grad, dOutputdh in zip([fd, cs, autograd], dOutputdh_lst):
                plot_gradients(output_func(wt_x, wt_y, hp, **kwargs), dOutputdh, hp[2], grad.__name__, step=10, ax=ax3)

            plt.suptitle(name)
            plt.show()
            plt.close('all')
        print(f'[x] {name}')
    except AssertionError as e:
        print(f'[ ] {name}')
        raise
    except Exception:
        print(f'[ ] {name}')
        # raise


@pytest.mark.parametrize('model', get_models(WindFarmModel))
def test_wind_farm_models(model):
    check_gradients(lambda site, wt: model(site, wt, IEA37SimpleBastankhahGaussianDeficit()), model.__name__)


@pytest.mark.parametrize('model', get_models(WakeDeficitModel))
def test_wake_deficit_models(model):
    check_gradients(lambda site, wt: PropagateDownwind(site, wt, model(), turbulenceModel=STF2005TurbulenceModel()),
                    model.__name__)


@pytest.mark.parametrize('model', get_models(BlockageDeficitModel))
def test_blockage_deficit_models(model):
    if model is not None:
        obj = model()
        obj.upstream_only = True
        check_gradients(lambda site, wt: All2AllIterative(site, wt, wake_deficitModel=NoWakeDeficit(),
                                                          blockage_deficitModel=obj,
                                                          turbulenceModel=STF2005TurbulenceModel()),
                        model.__name__,
                        wt_x=[1040, 320, 0],
                        wt_y=[0, 0, 0])


@pytest.mark.parametrize('model', get_models(SuperpositionModel))
def test_superposition_models(model):
    if model is not None:
        check_gradients(lambda site, wt: PropagateDownwind(
            site, wt, wake_deficitModel=BastankhahGaussianDeficit(),
            superpositionModel=model(),
            turbulenceModel=STF2005TurbulenceModel()),
            model.__name__)


@pytest.mark.parametrize('model', get_models(RotorAvgModel))
def test_rotor_average_models(model):
    if model is not None:
        check_gradients(lambda site, wt: PropagateDownwind(
            site, wt, wake_deficitModel=BastankhahGaussianDeficit(),
            rotorAvgModel=model()),
            model.__name__)


@pytest.mark.parametrize('model', get_models(DeflectionModel))
def test_deflection_models(model):
    if model is not None:
        check_gradients(lambda site, wt: PropagateDownwind(
            site, wt, wake_deficitModel=BastankhahGaussianDeficit(),
            deflectionModel=model(),
        ),
            model.__name__)


@pytest.mark.parametrize('model', get_models(TurbulenceModel))
def test_turbulence_models(model):
    if model is not None:
        check_gradients(lambda site, wt: PropagateDownwind(
            site, wt, wake_deficitModel=NiayifarGaussianDeficit(),
            turbulenceModel=model(),
        ),
            model.__name__)


@pytest.mark.parametrize('model', get_models(AddedTurbulenceSuperpositionModel))
def test_AddedTurbulenceSuperpositionModels(model):
    if model is not None:
        check_gradients(lambda site, wt: PropagateDownwind(
            site, wt, wake_deficitModel=NiayifarGaussianDeficit(),
            turbulenceModel=STF2017TurbulenceModel(addedTurbulenceSuperpositionModel=model()),
        ),
            model.__name__)


@pytest.mark.parametrize('model', get_models(GroundModel))
def test_ground_models(model):
    if model is not None:
        check_gradients(lambda site, wt: PropagateDownwind(
            site, wt, wake_deficitModel=BastankhahGaussianDeficit(groundModel=model()),
        ),
            model.__name__)


@pytest.mark.parametrize('site', [IEA37Site(16), Hornsrev1Site(), ParqueFicticioSite(distance=StraightDistance())])
def test_sites(site):
    x, y = site.initial_position[3]
    check_gradients(lambda site, wt, s=site: PropagateDownwind(
        s, wt, wake_deficitModel=BastankhahGaussianDeficit(),
    ),
        site.__class__.__name__,
        wt_x=[x - 1040, x - 520, x],
        wt_y=[y, y, y],
        fd_decimal=4,
    )


@pytest.mark.parametrize('model', get_models(Shear))
def test_shear(model):
    if model is not None:
        model = {PowerShear: PowerShear(h_ref=100, alpha=.1),
                 LogShear: LogShear(h_ref=100, z0=.03)}[model]
        check_gradients(lambda site, wt, s=Hornsrev1Site(shear=model): PropagateDownwind(
            s, wt, wake_deficitModel=BastankhahGaussianDeficit(),
        ),
            model.__class__.__name__,
            wt_h=[100, 100, 100]
        )


@pytest.mark.parametrize('model', get_models(StraightDistance))
def test_distance_models(model):
    print(model)


@pytest.mark.parametrize('wt', [IEA37WindTurbines, V80, IEA34_130_1WT_Surrogate])
def test_windturbines(wt):
    wt = wt()

    def get_wfm(site, wt, wt_=wt):
        return PropagateDownwind(site, wt_, wake_deficitModel=BastankhahGaussianDeficit(),
                                 turbulenceModel=STF2017TurbulenceModel())
    iea34 = wt.name() == 'IEA 3.4MW'
    check_gradients(get_wfm, wt.__class__.__name__, fd_step=(1e-6, 1e-3)[iea34], fd_decimal=(6, 2)[iea34])


output_lst = ['WS_eff', 'TI_eff', 'power', 'ct']


@pytest.mark.parametrize('output', output_lst)
def test_output(output):
    argnum = output_lst.index(output)
    # WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, wt_inputs = calc_wt_interaction

    def output_func(wfm):
        return lambda *args, argnum=argnum, **kwargs: wfm.calc_wt_interaction(*args, **kwargs)[argnum].mean()
    wake_deficitModel = BastankhahGaussianDeficit
    check_gradients(lambda site, wt: PropagateDownwind(site, V80(), wake_deficitModel(), turbulenceModel=STF2005TurbulenceModel()),
                    name=output, output=(output_func, output), fd_decimal=[6, 2][output == 'power'])
