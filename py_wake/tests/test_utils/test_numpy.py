import os

from autograd.numpy.numpy_boxes import ArrayBox
from numpy import newaxis as na
import pytest

from py_wake import np
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussianDeficit
from py_wake.deflection_models.deflection_model import DeflectionModel
from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.ground_models.ground_models import GroundModel
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel
from py_wake.site._site import Site
from py_wake.site.distance import StraightDistance
from py_wake.site.shear import Shear
from py_wake.superposition_models import SuperpositionModel, AddedTurbulenceSuperpositionModel
from py_wake.tests import npt
from py_wake.turbulence_models.stf import STF2005TurbulenceModel, STF2017TurbulenceModel
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.utils.gradients import autograd
from py_wake.utils.model_utils import get_models
from py_wake.utils.numpy_utils import Numpy32
from py_wake.utils.profiling import profileit
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.examples.data.hornsrev1 import Hornsrev1Site


@pytest.mark.parametrize('v,dtype,dtype32', [(5., float, np.float32),
                                             (5 + 0j, complex, np.complex64)])
def test_32bit_precision(v, dtype, dtype32):

    assert np.array(v).dtype == dtype
    with Numpy32():
        assert np.array(v).dtype == dtype32
    assert np.array(v).dtype == dtype


def test_members():
    import numpy
    with Numpy32():
        assert np.pi == numpy.pi
        npt.assert_array_equal(np.r_[3, 4], numpy.r_[3, 4])
        np.array([3]).dtype
        np.iscomplexobj([3])


def test_autograd32():
    def f(x):
        assert isinstance(x, ArrayBox)
        assert x.dtype == np.float32
        return x**2

    with Numpy32():
        autograd(f)([1, 2])


def test_speed_mem():
    if os.name == 'posix':
        pytest.xfail("Memory tests behave differently on Linux")

    def f(x):
        return (x**2).sum()

    N = 1000
    x = np.arange(N, dtype=float) * np.arange(1024**2 / 8)[:, na] + 1
    t64, mem64 = profileit(f)(x)[1:]

    with Numpy32():
        x = np.asarray(x)
        t32, mem32 = profileit(f)(x)[1:]

    assert t32 / t64 < .6
    assert mem32 / mem64 < .6

# def test_speed():
#     site = Hornsrev1Site()
#     windTurbines = V80()
#     wfm = BastankhahGaussian(site, windTurbines)
#     x, y = square(200, 5 * 80)
#     timeit(wfm.__call__, verbose=1)(x, y)
#     with Numpy32():
#         timeit(wfm.__call__, verbose=1)(x, y)

# def test_speed_gradients():
#     site = Hornsrev1Site()
#     windTurbines = V80()
#     wfm = BastankhahGaussian(site, windTurbines)
#     x, y = square(200, 5 * 80)
#     with Numpy32():
#         timeit(wfm.aep_gradients, verbose=1)(autograd, wrt_arg=['x', 'y'], wd_chunks=4, x=x, y=y)
#     timeit(wfm.aep_gradients, verbose=1)(autograd, wrt_arg=['x', 'y'], wd_chunks=4, x=x, y=y)


def check_numpy32(wfm, name):
    if hasattr(wfm.site, 'initial_position'):
        x, y = wfm.site.initial_position[:16].T
    else:
        x, y = IEA37Site(16).initial_position.T
    h = wfm.windTurbines.hub_height() + np.arange(len(x))
    kwargs = {'x': x, 'y': y, 'h': h, 'wd': [0, 90, 180, 270], 'ws': [8, 9, 10]}
    with Numpy32():
        aep_32 = wfm(**kwargs).aep()
    assert aep_32.dtype.name == 'float32'
    aep_64 = wfm(**kwargs).aep()
    assert aep_64.dtype.name == 'float64'
    npt.assert_allclose(aep_32, aep_64, rtol=0.01, atol=0.03)


@pytest.mark.parametrize('model_type,model',
                         [(mt.__name__, m) for mt in [WindFarmModel,
                                                      WakeDeficitModel,
                                                      BlockageDeficitModel,
                                                      SuperpositionModel,
                                                      RotorAvgModel,
                                                      DeflectionModel,
                                                      TurbulenceModel,
                                                      AddedTurbulenceSuperpositionModel,
                                                      GroundModel,
                                                      Site,
                                                      Shear,
                                                      StraightDistance,
                                                      ] for m in get_models(mt)])
def test_all_models(model_type, model):
    if model is None:
        return
    d = {'WindFarmModel': (PropagateDownwind, All2AllIterative)[model_type == 'BlockageDeficitModel'],
         'WakeDeficitModel': IEA37SimpleBastankhahGaussianDeficit,
         'TurbulenceModel': STF2005TurbulenceModel}
    if model_type == 'AddedTurbulenceSuperpositionModel':
        d['TurbulenceModel'] = STF2017TurbulenceModel(addedTurbulenceSuperpositionModel=model())
    elif model_type == 'GroundModel':
        d['WakeDeficitModel'] = IEA37SimpleBastankhahGaussianDeficit(groundModel=model())
    elif model_type in ['Site', 'Shear']:
        pass
    else:
        d[model_type] = model

    site = IEA37Site(16)
    if model_type == 'Site':
        site = model()
    elif model_type == 'Shear':
        site = Hornsrev1Site(shear=model())
    elif model_type == 'StraightDistance':
        site = ParqueFicticioSite(distance=model())

    def get(k):
        try:
            return d[k]()
        except TypeError:
            return d[k]

    kwargs = {k: get(v) for k, v in [('wake_deficitModel', 'WakeDeficitModel'),
                                     ('turbulenceModel', 'TurbulenceModel'),
                                     ('blockage_deficitModel', 'BlockageDeficitModel'),
                                     ('superpositionModel', 'SuperpositionModel'),
                                     ('rotorAvgModel', 'RotorAvgModel'),
                                     ('deflectionModel', 'DeflectionModel'),
                                     ('turbulenceModel', 'TurbulenceModel')] if v in d}

    wt = IEA37_WindTurbines()
    wfm = d['WindFarmModel'](site, wt, **kwargs)
    check_numpy32(wfm, model.__name__)
