from py_wake.deficit_models.noj import NOJ
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.tests.test_files import tfp
from py_wake.wind_farm_models.wind_farm_model import SimulationResult


def test_save_load():
    site = UniformSite([1], ti=0)
    windTurbines = IEA37_WindTurbines()
    wfm = NOJ(site, windTurbines)

    sim_res1 = wfm([0], [0], wd=270)

    sim_res1.save(tfp + "tmp.nc")

    sim_res2 = SimulationResult.load(tfp + 'tmp.nc', wfm)
    for sim_res in [sim_res1, sim_res2]:

        npt.assert_almost_equal(sim_res.aep().sum(), 3.35 * 24 * 365 / 1000)
        npt.assert_almost_equal(sim_res.aep(normalize_probabilities=True).sum(), 3.35 * 24 * 365 / 1000)

        npt.assert_equal(sim_res.aep().data.sum(), wfm.aep([0], [0], wd=270))
        npt.assert_almost_equal(sim_res.aep(normalize_probabilities=True).sum(),
                                wfm.aep([0], [0], wd=270, normalize_probabilities=True))
        npt.assert_almost_equal(sim_res.aep(with_wake_loss=False).sum(),
                                wfm.aep([0], [0], wd=270, with_wake_loss=False))
