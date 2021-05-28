from py_wake.wind_farm_models.engineering_models import PropagateDownwind

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.deficit_models.noj import NOJ
import pytest


def test_yaw_wrong_name():
    wfm = NOJ(Hornsrev1Site(), V80())
    for k in ['yaw_ilk', 'Yaw']:
        with pytest.raises(ValueError, match=r'Custom \*yaw\*\-keyword arguments not allowed'):
            wfm([0], [0], **{k: [[[30]]]})
