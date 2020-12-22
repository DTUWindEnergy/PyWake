from py_wake.utils.get_models import get_models, cls_in, get_signature
from py_wake.deficit_models.deficit_model import BlockageDeficitModel
from py_wake.deficit_models.fuga import FugaDeficit
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.turbulence_models.turbulence_model import SqrMaxSum
from py_wake.site._site import Site
from py_wake.site.xrsite import XRSite
from py_wake.superposition_models import SuperpositionModel


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
