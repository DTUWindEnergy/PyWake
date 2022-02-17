import pytest

import numpy as np
from py_wake.deficit_models.noj import NOJ, NOJDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, HornsrevV80, wt9_y, wt9_x
from py_wake.tests import npt
from py_wake.wind_farm_models.engineering_models import PropagateDownwind


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
