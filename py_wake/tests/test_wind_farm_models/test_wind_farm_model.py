import pytest
from py_wake import np
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.deficit_models.noj import NOJ, NOJDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, HornsrevV80, wt9_y, wt9_x, wt16_x, wt16_y
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.tests import npt
from py_wake.utils.gradients import autograd, cs, fd
from py_wake.utils.profiling import timeit, profileit
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
import matplotlib.pyplot as plt
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.flow_map import XYGrid
import os
from py_wake import examples


def test_yaw_wrong_name():
    wfm = NOJ(Hornsrev1Site(), V80())
    for k in ['yaw_ilk', 'Yaw']:
        with pytest.raises(ValueError, match=r'Custom \*yaw\*\-keyword arguments not allowed'):
            wfm([0], [0], **{k: [[[30]]]})


def test_yaw_dimensions():
    site = Hornsrev1Site()
    windTurbines = HornsrevV80()
    wake_deficitModel = NOJDeficit()

    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=wake_deficitModel)

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
    site = Hornsrev1Site()
    x, y = wt16_x, wt16_y
    wt = HornsrevV80()
    wfm = NOJ(site, wt)
    WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, *_ = wfm.calc_wt_interaction(x, y, wd_chunks=3, ws_chunks=2, n_cpu=None)
    WS_ref, TI_ref, power_ref, ct_ref, *_ = wfm.calc_wt_interaction(x, y)
    npt.assert_array_equal(WS_eff_ilk, WS_ref)
    npt.assert_array_equal(TI_eff_ilk, np.broadcast_to(TI_ref, (16, 360, 23)))
    npt.assert_array_equal(power_ilk, power_ref)
    npt.assert_array_equal(ct_ilk, ct_ref)


def test_chunks_results():
    site = Hornsrev1Site()
    x, y = wt16_x, wt16_y
    wt = HornsrevV80()
    wfm = NOJ(site, wt)

    sim_res_ref, t_ref, mem_ref = profileit(wfm)(x, y)
    sim_res, t, mem = profileit(wfm)(x, y, wd_chunks=3, ws_chunks=2)

    npt.assert_array_equal(sim_res_ref.aep(), sim_res.aep())
    # assert mem < mem_ref
    # assert t < t_ref * 6


@pytest.mark.parametrize('wd', [np.arange(360), np.arange(0, 360, 22.5), np.arange(3) * 2])
def test_aep_chunks_results(wd):
    site = Hornsrev1Site()
    x, y = wt16_x, wt16_y
    wt = HornsrevV80()
    wfm = NOJ(site, wt)

    aep_ref = wfm.aep(x, y, wd=wd)
    aep = wfm.aep(x, y, wd_chunks=3, ws_chunks=2, wd=wd)

    npt.assert_array_almost_equal(aep, aep_ref, 10)


def test_aep_time_chunks_results():
    site = Hornsrev1Site()
    x, y = wt16_x, wt16_y
    wt = HornsrevV80()
    wfm = NOJ(site, wt)

    d = np.load(os.path.dirname(examples.__file__) + "/data/time_series.npz")
    wd, ws = [d[k][:6 * 24] for k in ['wd', 'ws']]

    aep_ref = wfm.aep(x, y, wd=wd, ws=ws, time=True)
    aep = wfm(x, y, wd_chunks=3, ws_chunks=2, wd=wd, ws=ws, time=True).aep().sum()
    npt.assert_array_almost_equal(aep, aep_ref, 10)


@pytest.mark.parametrize('rho', [1.225, 1.3, np.full((16,), 1.3), np.full((16, 360), 1.3),
                                 np.full((16, 360, 23), 1.3), np.full((360), 1.3), np.full((360, 23), 1.3)])
def test_aep_chunks_input_dims(rho):
    site = Hornsrev1Site()
    x, y = wt16_x, wt16_y
    wt = HornsrevV80()
    wfm = NOJ(site, wt)

    aep1 = wfm(x, y, wd_chunks=3, ws_chunks=2, Air_density=rho).aep().sum()
    aep_ref = wfm(x, y, Air_density=rho).aep().sum()
    aep2 = wfm.aep(x, y, wd_chunks=3, ws_chunks=2, Air_density=rho)

    npt.assert_array_almost_equal(aep1, aep_ref, 10)
    npt.assert_array_almost_equal(aep2, aep_ref, 10)


@pytest.mark.parametrize('wrt_arg', ['x', 'y', 'h',
                                     ['x', 'y'], ['x', 'h'],
                                     # 'wd', 'ws'
                                     'yaw'
                                     ])
def test_aep_gradients_function(wrt_arg):
    site = Hornsrev1Site()
    iea37_site = IEA37Site(16)

    wt = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site, wt, deflectionModel=JimenezWakeDeflection())
    x, y = iea37_site.initial_position[np.array([0, 2, 5, 8, 14])].T
    kwargs = {'x': x, 'y': y, 'h': x * 0 + wt.hub_height(),
              'wd': [0], 'ws': 9.8, 'yaw': np.arange(1, 6).reshape((5, 1, 1)) * 5}
    dAEP_autograd = wfm.aep_gradients(gradient_method=autograd, wrt_arg=wrt_arg)(**kwargs)
    dAEP_cs = wfm.aep_gradients(gradient_method=cs, wrt_arg=wrt_arg)(**kwargs)
    dAEP_fd = wfm.aep_gradients(gradient_method=fd, wrt_arg=wrt_arg)(**kwargs)

    if 0:
        ax1, ax2 = plt.subplots(1, 2)[1]
        wfm(**kwargs).flow_map(XYGrid(resolution=100)).plot_wake_map(ax=ax1)
        ax2.set_title(wrt_arg)
        ax2.plot(dAEP_autograd.flatten(), '.')
        plt.show()

    npt.assert_array_almost_equal(dAEP_autograd, dAEP_cs, 15)
    npt.assert_array_almost_equal(dAEP_autograd, dAEP_fd, 6)
    npt.assert_array_equal(dAEP_autograd, wfm.aep_gradients(gradient_method=autograd, wrt_arg=wrt_arg, **kwargs))


def test_aep_gradients_parallel():
    site = Hornsrev1Site()
    iea37_site = IEA37Site(16)

    wt = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site, wt, deflectionModel=JimenezWakeDeflection())
    x, y = iea37_site.initial_position[np.array([0, 2, 5, 8, 14])].T
    kwargs = {'x': x, 'y': y, 'h': x * 0 + wt.hub_height(), 'wd': None, 'ws': [10]}
    dAEP_ref, t_seq = timeit(lambda: wfm.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], **kwargs))()

    dAEP_autograd, t_par = timeit(lambda: wfm.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], n_cpu=2, **kwargs),
                                  min_runs=2)()

    npt.assert_array_almost_equal(dAEP_ref, dAEP_autograd, 8)

    # print(t_par, t_seq)
    # assert min(t_par)<min(t_seq) # not faster on 4 CPU laptop


def test_aep_gradients_chunks():
    site = Hornsrev1Site()
    iea37_site = IEA37Site(16)

    wt = IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site, wt, deflectionModel=JimenezWakeDeflection())
    x, y = iea37_site.initial_position[np.array([0, 2, 5, 8, 14])].T
    kwargs = {'x': x, 'y': y, 'h': x * 0 + wt.hub_height(), 'wd': None, 'ws': [10]}
    dAEP_ref = wfm.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], **kwargs)

    dAEP_autograd = wfm.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], wd_chunks=2, **kwargs)

    npt.assert_array_almost_equal(dAEP_ref, dAEP_autograd, 8)


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
    wf_model = IEA37SimpleBastankhahGaussian(site, IEA37_WindTurbines())
    sim_res = wf_model(x, y,  # wind turbine positions
                       wd=np.linspace(0, 337.5, 16),
                       Air_density=np.full(shape, 1.225))
    assert sim_res.Air_density.dims == dims
