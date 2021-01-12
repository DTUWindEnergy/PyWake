
from py_wake.deficit_models.deficit_model import BlockageDeficitModel
from py_wake.deficit_models.fuga import FugaDeficit
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit
from py_wake.site._site import Site
from py_wake.site.xrsite import XRSite
from py_wake.superposition_models import SuperpositionModel
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.turbulence_models.turbulence_model import SqrMaxSum
from py_wake.utils.model_utils import cls_in, get_models, get_signature, get_model_input
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.tests import npt


def test_get_models():
    assert cls_in(FugaDeficit, get_models(BlockageDeficitModel))
    assert cls_in(SelfSimilarityDeficit, get_models(BlockageDeficitModel))

    assert [n.__name__ for n in get_models(SuperpositionModel)] == ['LinearSum', 'SquaredSum', 'MaxSum', 'WeightedSum']


def test_get_signature():
    assert get_signature(NOJDeficit) == "NOJDeficit(k=0.1, use_effective_ws=False)"
    print()
    assert get_signature(NOJDeficit, indent_level=1) == """NOJDeficit(
    k=0.1,
    use_effective_ws=False)"""
    assert (get_signature(STF2017TurbulenceModel) ==
            "STF2017TurbulenceModel(addedTurbulenceSuperpositionModel=LinearSum())")
    assert (get_signature(STF2017TurbulenceModel, {'addedTurbulenceSuperpositionModel': SqrMaxSum}) ==
            "STF2017TurbulenceModel(addedTurbulenceSuperpositionModel=SqrMaxSum())")
    assert(get_signature(XRSite) ==
           "XRSite(ds, initial_position=None, interp_method='linear', shear=None, distance=StraightDistance(), default_ws=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], bounds='check')")


def test_get_model_input():
    site, windTurbines = IEA37Site(16), IEA37_WindTurbines()
    wfm = IEA37SimpleBastankhahGaussian(site, windTurbines)
    args = get_model_input(wfm, [1000], [0], ws=10, wd=270)
    npt.assert_array_almost_equal(args['dw_ijl'], [[[1000]]])
    npt.assert_array_almost_equal(args['hcw_ijl'], [[[0]]])
    npt.assert_array_almost_equal(args['dh_ijl'], [[[0]]])
    npt.assert_array_almost_equal(args['yaw_ilk'], [[[0]]])
    npt.assert_array_almost_equal(args['WS_ilk'], [[[10]]])
    npt.assert_array_almost_equal(args['ct_ilk'], [[[8 / 9]]])
