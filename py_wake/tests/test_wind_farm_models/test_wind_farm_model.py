import pytest
from py_wake import np
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit
from py_wake.deficit_models.noj import NOJ

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, wt9_y, wt9_x, wt16_x, wt16_y
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.wind_turbines import WindTurbines
from py_wake.tests import npt
from py_wake.utils.gradients import autograd, cs, fd
from py_wake.utils.profiling import timeit, profileit
import matplotlib.pyplot as plt
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.flow_map import XYGrid
import os
from py_wake import examples
from py_wake.literature.iea37_case_study1 import IEA37CaseStudy1
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.site._site import UniformSite
from py_wake.deficit_models.utils import ct2a_mom1d
import warnings


def test_yaw_wrong_name():
    wfm = NOJ(Hornsrev1Site(), V80())
    for k in ['yaw_ilk', 'Yaw']:
        with pytest.raises(ValueError, match=r'Custom \*yaw\*\-keyword arguments not allowed'):
            wfm([0], [0], **{k: [[[30]]]})


def test_yaw_dimensions():
    wf_model = NOJ(Hornsrev1Site(), V80(), ct2a=ct2a_mom1d)

    x, y = wt9_x, wt9_y

    I, L, K = 9, 360, 23
    for yaw in [45,
                np.broadcast_to(45, (I,)),
                np.broadcast_to(45, (I, L)),
                np.broadcast_to(45, (I, L, K)),
                ]:
        sim_res_all_wd = wf_model(x, y, yaw=yaw)
        if len(np.shape(yaw)) > 1:
            yaw1 = yaw[:, 1:2]
        else:
            yaw1 = yaw
        sim_res_1wd = wf_model(x, y, wd=1, yaw=yaw1)

        npt.assert_almost_equal(sim_res_all_wd.WS_eff.sel(wt=1, wd=1, ws=10), 9.70670076)
        npt.assert_almost_equal(sim_res_1wd.WS_eff.sel(wt=1, ws=10), 9.70670076)


def test_calc_wt_interaction_parallel_results():
    x, y = wt16_x, wt16_y
    wfm = NOJ(Hornsrev1Site(), V80())
    WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, *_ = wfm(x, y, wd_chunks=3, ws_chunks=2, n_cpu=None,
                                                        return_simulationResult=False)
    WS_ref, TI_ref, power_ref, ct_ref, *_ = wfm(x, y, return_simulationResult=False)
    npt.assert_array_equal(WS_eff_ilk, WS_ref)
    npt.assert_array_equal(TI_eff_ilk, np.broadcast_to(TI_ref, (16, 360, 23)))
    npt.assert_array_equal(power_ilk, power_ref)
    npt.assert_array_equal(ct_ilk, ct_ref)


def test_chunks_results():
    x, y = wt16_x, wt16_y
    wfm = NOJ(Hornsrev1Site(), V80())

    sim_res_ref = profileit(wfm)(x, y)[0]
    sim_res = profileit(wfm)(x, y, wd_chunks=3, ws_chunks=2)[0]

    npt.assert_array_equal(sim_res_ref.aep(), sim_res.aep())
    # assert mem < mem_ref
    # assert t < t_ref * 6


@pytest.mark.parametrize('wd', [np.arange(360), np.arange(0, 360, 22.5), np.arange(3) * 2])
def test_aep_chunks_results(wd):
    x, y = wt16_x, wt16_y
    wfm = NOJ(Hornsrev1Site(), V80())

    aep_ref = wfm.aep(x, y, wd=wd)
    aep = wfm.aep(x, y, wd_chunks=3, ws_chunks=2, wd=wd)

    npt.assert_array_almost_equal(aep, aep_ref, 10)


def test_aep_time_chunks_results():
    x, y = wt16_x, wt16_y
    wfm = NOJ(Hornsrev1Site(), V80())

    d = np.load(os.path.dirname(examples.__file__) + "/data/time_series.npz")
    wd, ws = [d[k][:6 * 24] for k in ['wd', 'ws']]

    aep_ref = wfm.aep(x, y, wd=wd, ws=ws, time=True)
    aep = wfm(x, y, wd_chunks=3, ws_chunks=2, wd=wd, ws=ws, time=True).aep().sum()
    npt.assert_array_almost_equal(aep, aep_ref, 10)


@pytest.mark.parametrize('rho', [1.225, 1.3, np.full((16,), 1.3), np.full((16, 360), 1.3),
                                 np.full((16, 360, 23), 1.3), np.full((360), 1.3), np.full((360, 23), 1.3)])
def test_aep_chunks_input_dims(rho):
    x, y = wt16_x, wt16_y
    wfm = NOJ(Hornsrev1Site(), V80())

    aep1 = wfm(x, y, wd_chunks=3, ws_chunks=2, Air_density=rho).aep().sum()
    aep_ref = wfm(x, y, Air_density=rho).aep().sum()
    aep2 = wfm.aep(x, y, wd_chunks=3, ws_chunks=2, Air_density=rho)

    npt.assert_array_almost_equal(aep1, aep_ref, 10)
    npt.assert_array_almost_equal(aep2, aep_ref, 10)


@pytest.mark.parametrize('wrt_arg', ['x', 'y', 'h',
                                     ['x', 'y'], ['x', 'h'],
                                     'wd', 'ws'])
def test_aep_gradients_function1(wrt_arg):
    wfm = IEA37CaseStudy1(16, deflectionModel=JimenezWakeDeflection())
    x, y = wfm.site.initial_position[np.array([0, 2, 5, 8, 14])].T
    kwargs = {'x': x, 'y': y, 'h': x * 0 + wfm.windTurbines.hub_height(),
              'wd': [0], 'ws': 9.8, 'yaw': np.arange(1, 6).reshape((5, 1, 1)) * 5, 'tilt': 0}

    dAEP_autograd = wfm.aep_gradients(gradient_method=autograd, wrt_arg=wrt_arg)(**kwargs)
    dAEP_cs = wfm.aep_gradients(gradient_method=cs, wrt_arg=wrt_arg)(**kwargs)
    dAEP_fd = wfm.aep_gradients(gradient_method=fd, wrt_arg=wrt_arg)(**kwargs)

    if 0:
        ax1, ax2 = plt.subplots(1, 2)[1]
        wfm(**kwargs).flow_map(XYGrid(resolution=100)).plot_wake_map(ax=ax1)
        ax2.set_title(wrt_arg)
        ax2.plot(dAEP_autograd.flatten(), '.', label='autograd')
        ax2.plot(dAEP_cs.flatten(), '.', label='cs')
        ax2.plot(dAEP_fd.flatten(), '.', label='fd')
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(dAEP_autograd, dAEP_cs, 14)
    npt.assert_array_almost_equal(dAEP_autograd, dAEP_fd, 5)
    npt.assert_array_equal(dAEP_autograd, wfm.aep_gradients(gradient_method=autograd, wrt_arg=wrt_arg, **kwargs))


@pytest.mark.parametrize('wrt_arg', ['x', 'y', 'h',
                                     ['x', 'y'], ['x', 'h'],
                                     'wd', 'ws', 'yaw'])
def test_aep_gradients_function2(wrt_arg):
    wfm = IEA37CaseStudy1(16, deflectionModel=JimenezWakeDeflection())
    x, y = wfm.site.initial_position[np.array([0, 2, 5, 8, 14])].T
    kwargs = {'x': x, 'y': y, 'h': x * 0 + wfm.windTurbines.hub_height(),
              'wd': [0], 'ws': 9.8, 'yaw': np.arange(1, 6).reshape((5, 1, 1)) * 5, 'tilt': 0}

    dAEP_autograd = wfm.aep_gradients(gradient_method=autograd, wrt_arg=wrt_arg, **kwargs)
    dAEP_cs = wfm.aep_gradients(gradient_method=cs, wrt_arg=wrt_arg, **kwargs)
    dAEP_fd = wfm.aep_gradients(gradient_method=fd, wrt_arg=wrt_arg, **kwargs)

    if 0:
        ax1, ax2 = plt.subplots(1, 2)[1]
        wfm(**kwargs).flow_map(XYGrid(resolution=100)).plot_wake_map(ax=ax1)
        ax2.set_title(wrt_arg)
        ax2.plot(dAEP_autograd.flatten(), '.', label='autograd')
        ax2.plot(dAEP_cs.flatten(), '.', label='cs')
        ax2.plot(dAEP_fd.flatten(), '.', label='fd')
        plt.legend()
        plt.show()

    npt.assert_array_almost_equal(dAEP_autograd, dAEP_cs, 14)
    npt.assert_array_almost_equal(dAEP_autograd, dAEP_fd, 5)
    npt.assert_array_equal(dAEP_autograd, wfm.aep_gradients(gradient_method=autograd, wrt_arg=wrt_arg, **kwargs))


def test_aep_gradients_parallel():
    wfm = IEA37CaseStudy1(16, deflectionModel=JimenezWakeDeflection())
    x, y = wfm.site.initial_position[np.array([0, 2, 5, 8, 14])].T
    kwargs = {'x': x, 'y': y, 'h': x * 0 + wfm.windTurbines.hub_height(), 'yaw': 0, 'tilt': 0, 'wd': None, 'ws': [10]}
    dAEP_ref = timeit(lambda: wfm.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], **kwargs))()[0]

    dAEP_autograd = timeit(lambda: wfm.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], n_cpu=2, **kwargs),
                           min_runs=2)()[0]

    npt.assert_array_almost_equal(dAEP_ref, dAEP_autograd, 8)

    # print(t_par, t_seq)
    # assert min(t_par)<min(t_seq) # not faster on 4 CPU laptop


def test_aep_gradients_chunks():
    wfm = IEA37CaseStudy1(16, deflectionModel=JimenezWakeDeflection())
    x, y = wfm.site.initial_position[np.array([0, 2, 5, 8, 14])].T
    kwargs = {'x': x, 'y': y, 'h': x * 0 + wfm.windTurbines.hub_height(), 'wd': None, 'ws': [10], 'yaw': 0, 'tilt': 0}
    dAEP_ref = wfm.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], **kwargs)

    dAEP_autograd = wfm.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], wd_chunks=2, **kwargs)

    npt.assert_array_almost_equal(dAEP_ref, dAEP_autograd, 8)


def test_aep_gradients_with_types():
    # Generating the powerCtCurves (the problem does not occur when using V80())
    wts = WindTurbines.from_WindTurbine_lst([V80(), V80()])
    wfm = NOJ(UniformSite(), wts)

    x = [0, 500]
    y = [0, 0]
    t = [0, 0]

    daep_autograd = wfm.aep_gradients(autograd, x=x, y=y, type=t)
    daep_fd = wfm.aep_gradients(fd, x=x, y=y, type=t)
    daep_cs = wfm.aep_gradients(cs, x=x, y=y, type=t)
    npt.assert_array_almost_equal(daep_autograd, daep_cs, 10)
    npt.assert_array_almost_equal(daep_autograd, daep_fd)


@pytest.mark.parametrize('n_wt, shape,dims', [(16, (16,), ('wt',)),
                                              (12, (12,), ('wt',)),
                                              (12, (16,), ('wd',)),
                                              (16, (16, 16), ('wt', 'wd')),
                                              (12, (12, 16), ('wt', 'wd')),
                                              (12, (12, 16, 23), ('wt', 'wd', 'ws')),
                                              (16, (16, 23), ('wd', 'ws')),

                                              ])
def test_wt_kwargs_dimensions(n_wt, shape, dims):
    site = Hornsrev1Site(16)
    x, y = site.initial_position[:n_wt].T
    wfm = PropagateDownwind(site, IEA37_WindTurbines(), wake_deficitModel=IEA37SimpleBastankhahGaussianDeficit())

    sim_res = wfm(x, y,  # wind turbine positions
                  wd=np.linspace(0, 337.5, 16),
                  Air_density=np.full(shape, 1.225))
    assert sim_res.Air_density.dims == dims


@pytest.mark.parametrize('ws,time', [([7], False),
                                     ([7.0, 8, 9, 10, 11], True)])
def test_wake_model_two_turbine_types(ws, time):
    site = IEA37Site(16)
    wt = WindTurbines.from_WindTurbine_lst([IEA37_WindTurbines(), IEA37_WindTurbines()])
    wake_model = NOJ(site, wt)

    # n_cpu > 1 does not work when type is used, i.e. more than one wtg type. Reason is attempt to broadcast type (1d)
    # to the parameters which have shape reflecting multiple time steps
    wake_model(
        x=[0, 0, 1, 1],
        y=[0, 1, 0, 1],
        type=[0, 0, 1, 1],
        ws=ws,
        wd=[269.0, 270, 273, 267, 268],
        time=time,
        n_cpu=2,
    )


def test_wd_dependent_wt_positions():
    wfm = IEA37CaseStudy1(16)
    sim_res = wfm(x=[[-100, 0, 100], [100, 0, -100]], y=[[0, 100, 0],
                                                         [0, -100, 0]], wd=[0, 90, 180], WS=[[[5]], [[10]]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ws_eff = sim_res.flow_map().WS_eff.interp(
            x=('z', [-100, -300, 100]), y=('z', [-300, 100, 300]), wd=('z', [0, 90, 180])).squeeze().values
    npt.assert_array_equal(ws_eff[0], ws_eff)
    if 0:
        for wd, ax in zip(sim_res.wd, plt.subplots(3, 1, figsize=(4, 10))[1]):
            sim_res.flow_map(wd=wd).plot_wake_map(ax=ax)
        plt.show()


def test_ws_dependent_wt_positions():
    wfm = IEA37CaseStudy1(16)
    sim_res = wfm(x=[[[0, 0, 0]], [[100, 200, 300]]], y=np.zeros((2, 1, 3)), wd=[270], ws=[8, 9, 10])
    ws_eff = sim_res.flow_map().WS_eff.interp(x=400, y=0).squeeze()
    npt.assert_array_almost_equal(ws_eff, [4.06513, 4.117601, 3.830711])
    if 0:
        for ws, ax in zip(sim_res.ws, plt.subplots(3, 1, figsize=(4, 10))[1]):
            sim_res.flow_map(ws=ws).plot_wake_map(ax=ax)
        plt.show()


def test_time_dependent_wt_positions():
    wfm = IEA37CaseStudy1(16)
    sim_res = wfm(x=[[-100, 0, 100], [100, 0, -100]], y=[[0, 100, 0],
                                                         [0, -100, 0]], wd=[0, 90, 180], ws=[10, 10, 10], time=True)
    ws_eff = sim_res.flow_map(time=[0, 1, 2]).WS_eff.interp(
        x=('time', [-100, -300, 100]), y=('time', [-300, 100, 300])).squeeze()
    npt.assert_array_equal(ws_eff[0], ws_eff)
    if 0:
        for t, ax in zip(sim_res.time.values, plt.subplots(3, 1, figsize=(4, 10))[1]):
            sim_res.flow_map(time=t).plot_wake_map(ax=ax)
        plt.show()
