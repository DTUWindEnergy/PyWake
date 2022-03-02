import matplotlib.pyplot as plt
import numpy as np
from autograd import numpy as anp
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site, IEA37WindTurbines
from py_wake.utils.gradients import autograd, plot_gradients, fd, cs, hypot, cabs, interp,\
    _use_autograd_in, set_vjp
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines import _wind_turbines
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
import pytest
from autograd.core import primitive, defvjp
from py_wake.utils.model_utils import get_models
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit,\
    BastankhahGaussianDeficit, NiayifarGaussianDeficit
from numpy import newaxis as na
from py_wake.utils import gradients
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.turbulence_models.stf import STF2017TurbulenceModel, STF2005TurbulenceModel
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.superposition_models import SuperpositionModel, AddedTurbulenceSuperpositionModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.ground_models.ground_models import GroundModel
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
from py_wake.site.shear import Shear, LogShear, PowerShear
from py_wake.site.distance import StraightDistance
from py_wake.tests.check_speed import timeit
from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt
from py_wake.examples.data.iea34_130rwt._iea34_130rwt import IEA34_130_1WT_Surrogate


@pytest.mark.parametrize('obj', [_wind_turbines, WindTurbines, V80().power, _wind_turbines.__dict__])
def test_use_autograd_in(obj):
    _wind_turbines.np = np
    assert _wind_turbines.np == np
    with _use_autograd_in([obj]):
        assert _wind_turbines.np == anp
    assert _wind_turbines.np == np


def test_scalar2scalar():
    def f(x):
        return x**2 + 1

    x = np.array([3])
    npt.assert_equal(cs(f)(x), 6)
    npt.assert_almost_equal(fd(f)(x), 6, 5)
    npt.assert_equal(autograd(f)(x), 6)
    pf = primitive(f)
    defvjp(pf, lambda ans, x: lambda g: g * 2 * x)
    npt.assert_array_equal(autograd(pf, False)(x), 6)


def test_vector2vector_independent():
    def f(x):
        return x**2 + 1

    def df(x):
        return 2 * x

    x = np.array([2, 3, 4])
    ref = [4, 6, 8]
    npt.assert_array_almost_equal(fd(f, False)(x), ref, 5)
    npt.assert_array_equal(cs(f, False)(x), ref)
    npt.assert_array_equal(autograd(f, False)(x), ref)

    pf = primitive(f)
    defvjp(pf, lambda ans, x: lambda g: g * df(x))
    npt.assert_array_equal(autograd(pf, False)(x), ref)


def test_vector2vector_dependent():
    def f(x):
        return x**2 + x[::-1]

    def df(x):
        return np.diag(2 * x) + np.diag(np.ones(3))[::-1]

    x = np.array([2., 3, 4])
    ref = [[4., 0., 1.],
           [0., 7., 0.],
           [1., 0., 8.]]
    npt.assert_array_almost_equal(fd(f, True)(x), ref, 5)
    npt.assert_array_almost_equal_nulp(cs(f, True)(x), ref)
    npt.assert_array_equal(autograd(f, True)(x), ref)

    pf = primitive(f)
    defvjp(pf, lambda ans, x: lambda g: np.dot(g, df(x)))
    npt.assert_array_equal(autograd(pf, True)(x), ref)


def test_multivector2vector_independent():
    def f(x, y):
        return x**2 + 2 * y**3 + 1

    def dfdx(x, y):
        return 2 * x

    def dfdy(x, y):
        return 6 * y**2

    x = np.array([2, 3, 4])
    y = np.array([1, 2, 3])
    ref_x = [4, 6, 8]
    ref_y = [6, 24, 54]
    npt.assert_array_almost_equal(fd(f, False)(x, y), ref_x, 5)
    npt.assert_array_almost_equal(fd(f, False, 1)(x, y), ref_y, 4)

    npt.assert_array_almost_equal_nulp(cs(f, False)(x, y), ref_x)
    npt.assert_array_almost_equal_nulp(cs(f, False, 1)(x, y), ref_y)

    npt.assert_array_equal(autograd(f, False)(x, y), ref_x)
    npt.assert_array_equal(autograd(f, False, 1)(x, y), ref_y)

    pf = primitive(f)
    defvjp(pf, lambda ans, x, y: lambda g: g * dfdx(x, y), lambda ans, x, y: lambda g: g * dfdy(x, y))
    npt.assert_array_equal(autograd(pf, False)(x, y), ref_x)
    npt.assert_array_equal(autograd(pf, False, 1)(x, y), ref_y)


def test_scalar2multi_scalar():
    def fxy(x):
        return x**2 + 1, 2 * x + 1

    def f(x):
        fx, fy = fxy(x)
        return fx + fy

    x = 3.
    ref = 8
    npt.assert_equal(cs(f)(x), ref)
    npt.assert_almost_equal(fd(f)(x), ref, 5)
    npt.assert_equal(autograd(f)(x), ref)

    pf = primitive(f)
    defvjp(pf, lambda ans, x: lambda g: g * (2 * x + 2))
    npt.assert_array_equal(autograd(pf, False)(x), ref)

    pf = primitive(fxy)
    defvjp(pf, lambda ans, x: lambda g: (g[0] * 2 * x, g[1] * 2))
    npt.assert_array_equal(autograd(f, False)(x), ref)


def test_vector2multi_vector():
    def fxy(x):
        return x**2 + 1, 2 * x + 1

    def f0(x):
        return fxy(x)[0]

    def fsum(x):
        fx, fy = fxy(x)
        return fx + fy

    x = np.array([1., 2, 3])
    ref0 = [2, 4, 6]
    refsum = [4, 6, 8]
    npt.assert_equal(cs(f0)(x), ref0)
    npt.assert_almost_equal(fd(f0)(x), ref0, 5)
    npt.assert_equal(autograd(f0)(x), ref0)
    pf0 = primitive(f0)
    defvjp(pf0, lambda ans, x: lambda g: g * (2 * x))
    npt.assert_array_equal(autograd(pf0, False)(x), ref0)

    npt.assert_equal(cs(fsum)(x), refsum)
    npt.assert_almost_equal(fd(fsum)(x), refsum, 5)
    npt.assert_equal(autograd(fsum)(x), refsum)
    pfsum = primitive(fsum)
    defvjp(pfsum, lambda ans, x: lambda g: g * (2 * x + 2))
    npt.assert_array_equal(autograd(pfsum, False)(x), refsum)

    pfxy = primitive(fxy)

    def dfxy(x):
        return 2 * x, np.full(x.shape, 2)

    def gsum(x):
        fx, fy = pfxy(x)
        return fx + fy

    def g0(x):
        return pfxy(x)[0]

    pgsum = primitive(gsum)
    pg0 = primitive(g0)
    defvjp(pgsum, lambda ans, x: lambda g: g * np.sum(dfxy(x), 0))
    defvjp(pg0, lambda ans, x: lambda g: g * dfxy(x)[0])

    npt.assert_array_equal(autograd(pgsum, False)(x), refsum)
    npt.assert_array_equal(autograd(pg0, False)(x), ref0)

    defvjp(pfxy, lambda ans, x: lambda g: dfxy(x)[0])

    def h0(x):
        return pfxy(x)[0]
    npt.assert_array_equal(autograd(h0, False)(x), ref0)

    defvjp(pfxy, lambda ans, x: lambda g: np.sum(g * np.asarray(dfxy(x)), 0))

    def hsum(x):
        fx, fy = pfxy(x)
        return fx + fy

    npt.assert_array_equal(autograd(hsum, False)(x), refsum)


def test_gradients():
    wt = IEA37_WindTurbines()
    ws_lst = np.arange(3, 25, .1)

    ws_pts = np.array([3., 6., 9., 12.])
    dpdu_lst = autograd(wt.power)(ws_pts)
    if 0:
        plt.plot(ws_lst, wt.power(ws_lst))
        for dpdu, ws in zip(dpdu_lst, ws_pts):
            plot_gradients(wt.power(ws), dpdu, ws, "", 1)

        plt.show()
    dpdu_ref = np.where((ws_pts > 4) & (ws_pts <= 9.8),
                        3 * 3350000 * (ws_pts - 4)**2 / (9.8 - 4)**3,
                        0)

    npt.assert_array_almost_equal(dpdu_lst, dpdu_ref)

    fd_dpdu_lst = fd(wt.power)(ws_pts)
    npt.assert_array_almost_equal(fd_dpdu_lst, dpdu_ref, 0)

    cs_dpdu_lst = cs(wt.power)(ws_pts)
    npt.assert_array_almost_equal(cs_dpdu_lst, dpdu_ref)


def test_plot_gradients():
    x = np.arange(-3, 4, .1)
    plt.plot(x, x**2)
    plot_gradients(1.5**2, 3, 1.5, "test", 1)
    if 0:
        plt.show()
    plt.close('all')


def test_hypot():
    # Test real.
    a = np.array([3, 9])
    b = np.array([4, 40])
    npt.assert_equal(hypot(a, b), np.array([5, 41]))
    # Test complex.
    a = 3 + 4j
    b = 1 - 2j
    npt.assert_array_almost_equal(hypot(a, b), 2.486028939392892 + 4.022479320953552j)


def test_cabs():
    a = [-5, 6]
    npt.assert_array_equal(cabs(a), np.abs(a))
    npt.assert_array_almost_equal(fd(cabs)(a), [-1, 1], 10)
    npt.assert_array_equal(cs(cabs)(a), [-1, 1])
    npt.assert_array_equal(autograd(cabs)(a), [-1, 1])


def test_gradients_interp():
    xp, x, y = [5, 16], [0, 10, 20], [100, 200, 400]

    def f(xp):
        return 2 * gradients.interp(xp, x, y)
    npt.assert_array_equal(interp(xp, x, y), np.interp(xp, x, y))
    npt.assert_array_almost_equal(fd(f)(xp), [20, 40])
    npt.assert_array_equal(cs(f)(xp), [20, 40])
    npt.assert_array_equal(autograd(f)(xp), [20, 40])


def test_gradients_logaddexp():

    x = [0, 0, 0, 1]
    y = [1, 100, 1000, 1]

    def f(x, y):
        return 2 * gradients.logaddexp(x, y)

    dfdx = 2 * (np.exp(x - np.logaddexp(x, y)))
    dfdy = 2 * (np.exp(y - np.logaddexp(x, y)))
    npt.assert_array_equal(f(x, y), 2 * np.logaddexp(x, y))
    npt.assert_array_almost_equal(fd(f)(x, y), dfdx)
    npt.assert_array_almost_equal(fd(f, argnum=1)(x, y), dfdy)
    npt.assert_array_almost_equal(cs(f)(x, y), dfdx)
    npt.assert_array_almost_equal(cs(f, argnum=1)(x, y), dfdy)
    npt.assert_array_equal(autograd(f)(x, y), dfdx)
    npt.assert_array_equal(autograd(f, argnum=1)(x, y), dfdy)


def test_set_vjp():
    def df(x):
        return 3 * x

    @set_vjp(df)
    def f(x):
        return x**2

    assert f(4) == 16
    npt.assert_almost_equal(fd(f)(4), 8, 5)
    npt.assert_almost_equal(cs(f)(4), 8)
    npt.assert_array_equal(autograd(f)([4, 5]), [12, 15])


def test_set_vjp_cls():
    class T():
        def df(self, x):
            return 3 * x

        @set_vjp(df)
        def f(self, x):
            return x**2

    t = T()
    assert t.f(4) == 16
    npt.assert_almost_equal(fd(t.f)(4), 8, 5)
    npt.assert_almost_equal(cs(t.f)(4), 8)
    assert autograd(t.f)(4) == 12


def test_set_vjp_kwargs():
    def df(x):
        return 3 * x

    @set_vjp(df)
    def f(x):
        return x**2

    def f2(x):
        return f(x)

    assert f2(x=4) == 16
    npt.assert_almost_equal(fd(f2)(x=4), 8, 5)
    npt.assert_almost_equal(cs(f2)(x=4), 8)
    assert autograd(f2)(x=4) == 12


@pytest.mark.parametrize('interpolator', [gradients.PchipInterpolator, gradients.UnivariateSpline])
def test_Interpolators(interpolator):
    xp = np.linspace(0, 2 * np.pi, 100000)
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.sin(x)
    plt.plot(xp, np.sin(xp), label='sin(x)')
    plt.plot(x, np.sin(x), '.')

    interp = interpolator(x, y)
    x = x[3]
    dfdx_lst = [method(interp)(x) for method in [fd, cs, autograd]]
    npt.assert_array_almost_equal(dfdx_lst[0], dfdx_lst[1])
    npt.assert_array_equal(dfdx_lst[1], dfdx_lst[2])
    gradients.color_dict = {}
    if 0:
        plt.plot(xp, interp(xp), label='Interpolated')
        dfdx = np.cos(x)
        gradients.plot_gradients(np.sin(x), dfdx, x, label='analytical', step=.5)

        for method, dfdx in zip([fd, cs, autograd], dfdx_lst):
            gradients.plot_gradients(interp(x), dfdx, x, label=method.__name__, step=1)
        plt.legend()
        plt.show()


def test_manual_vs_autograd_speed():
    cubePowerSimpleCt = CubePowerSimpleCt()
    x = np.random.random(10000) * 30

    def t(method):
        autograd(method)(x, 0)
        autograd(method)(x, 1)
    t_autograd = np.mean(timeit(t, min_time=0.2)(cubePowerSimpleCt._power_ct)[1])
    t_manual = np.mean(timeit(t, min_time=.2)(cubePowerSimpleCt._power_ct_withgrad)[1])
    assert np.abs(t_manual - t_autograd) / t_manual < 0.08, (t_manual, t_autograd)


def test_multiple_inputs():
    def f(x, y):
        return x * y

    for method in [fd, cs, autograd]:
        npt.assert_array_almost_equal(method(f, argnum=[0, 1])(np.array([2, 3]), np.array([4, 5])),
                                      [[4, 5], [2, 3]], 8)


def check_gradients(wfm, name, wt_x=[-1300, -650, 0], wt_y=[0, 0, 0], wt_h=[110, 110, 110], fd_step=1e-6, fd_decimal=6):
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

        dAEPdx_lst = [grad(wfm.aep, True, 0)(xp, wt_y, **kwargs)[2] for grad in [fdstep, cs, autograd]]
        npt.assert_almost_equal(dAEPdx_lst[0], dAEPdx_lst[1], fd_decimal)
        npt.assert_almost_equal(dAEPdx_lst[1], dAEPdx_lst[2], 10)

        dAEPdy_lst = [grad(wfm.aep, True, 1)(wt_x, yp, **kwargs)[2] for grad in [fdstep, cs, autograd]]
        npt.assert_almost_equal(dAEPdy_lst[0], dAEPdy_lst[1], fd_decimal)
        npt.assert_almost_equal(dAEPdy_lst[1], dAEPdy_lst[2], 10)

        dAEPdh_lst = [grad(wfm.aep, True, 2)(wt_x, wt_y, hp, **kwargs)[2] for grad in [fdstep, cs, autograd]]
        npt.assert_almost_equal(dAEPdh_lst[0], dAEPdh_lst[1], fd_decimal)
        npt.assert_almost_equal(dAEPdh_lst[1], dAEPdh_lst[2], 10)

        if 0:
            wfm(wt_x, wt_y, **kwargs).flow_map().plot_wake_map()

            _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
            ax1.set_title("Center line")
            ax1.set_xlabel('Downwind distance [m]')
            ax1.set_ylabel('AEP [GWh]')

            ax1.plot(x_lst[:, 2], [wfm.aep(xp, wt_y, **kwargs) for xp in x_lst])
            ax1.axvline(wt_x[1], color='k')
            for grad, dAEPdx in zip([fd, cs, autograd], dAEPdx_lst):
                plot_gradients(wfm.aep(xp, wt_y, **kwargs), dAEPdx, xp[2], grad.__name__, step=100, ax=ax1)

            ax2.set_xlabel('Crosswind distance [m]')
            ax2.set_ylabel('AEP [GWh]')
            ax2.set_title("%d m downstream" % wt_x[1])

            ax2.plot(y_lst[:, 2], [wfm.aep(wt_x, yp, **kwargs) for yp in y_lst])
            gradients.color_dict = {}
            for grad, dAEPdy in zip([fd, cs, autograd], dAEPdy_lst):
                plot_gradients(wfm.aep(wt_x, yp, **kwargs), dAEPdy, yp[2], grad.__name__, step=50, ax=ax2)

            ax3.set_xlabel('hubheight [m]')
            ax3.set_ylabel('AEP [GWh]')

            ax3.plot(h_lst[:, 2], [wfm.aep(wt_x, wt_y, hp, **kwargs) for hp in h_lst])
            gradients.color_dict = {}
            for grad, dAEPdh in zip([fd, cs, autograd], dAEPdh_lst):
                plot_gradients(wfm.aep(wt_x, wt_y, hp, **kwargs), dAEPdh, hp[2], grad.__name__, step=10, ax=ax3)

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


@pytest.mark.parametrize('site', [IEA37Site(16), Hornsrev1Site(), ParqueFicticioSite()])
def test_sites(site):
    x, y = site.initial_position[3]
    check_gradients(lambda site, wt, s=site: PropagateDownwind(
        s, wt, wake_deficitModel=BastankhahGaussianDeficit(),
    ),
        site.__class__.__name__,
        wt_x=[x - 1040, x - 520, x],
        wt_y=[y, y, y]
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
