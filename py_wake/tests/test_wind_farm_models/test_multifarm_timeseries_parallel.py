import pytest
from py_wake import NOJ
from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windturbine
from py_wake.wind_turbines import WindTurbines
from py_wake.examples.data.iea37 import iea37_path
from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt


@pytest.fixture
def two_turbines() -> WindTurbines:
    iea37_wtg_file = iea37_path + 'iea37-335mw.yaml'
    name, hub_height, diameter, power_ct_func = read_iea37_windturbine(iea37_wtg_file)

    name_2 = '5MW'
    hub_height_2 = 115.0
    diameter_2 = 140.0

    power_ct_func_2 = CubePowerSimpleCt(
        ws_cutin=power_ct_func.ws_cutin,
        ws_cutout=power_ct_func.ws_cutout,
        ws_rated=power_ct_func.ws_rated,
        power_rated=5000000.0,
        power_unit='W',
        ct=power_ct_func.ct,
        ct_idle=power_ct_func.ct_idle,
        # additional_models=[]
    )
    return WindTurbines(
        names=[name, name_2],
        diameters=[diameter, diameter_2],
        hub_heights=[hub_height, hub_height_2],
        powerCtFunctions=[power_ct_func, power_ct_func_2],
    )


def test_wake_model_two_turbine_types(two_turbines):
    site = IEA37Site(16)
    wake_model = NOJ(site, two_turbines)

    # n_cpu > 1 does not work when type is used, i.e. more than one wtg type. Reason is attempt to broadcast type (1d)
    # to the parameters which have shape reflecting multiple time steps
    wake_model(
        x=[0, 0, 1, 1],
        y=[0, 1, 0, 1],
        type=[0, 0, 1, 1],
        ws=[7.0, 8, 9, 10],
        wd=[269.0, 270, 273, 267],
        time=[0, 1, 2, 3],
        n_cpu=2,
    )

def test_time_chunks(two_turbines):
    site = IEA37Site(16)
    wake_model = NOJ(site, two_turbines)

    # WindFarmModel __call__ reports to support "time_chunks" as a parameter, but when entering
    # EngineeringWindFarmModel.calc_wt_interaction, this is not supporting time_chunks and it fails
    # Looks like using wd_chunks with time does the trick, but maybe make it explicit such that using "time_chunks" works
    wake_model(
        x=[0, 0, 1, 1],
        y=[0, 1, 0, 1],
        type=[0, 0, 1, 1],
        ws=[7.0, 8, 9, 10],
        wd=[269.0, 270, 273, 267],
        time=[0, 1, 2, 3],
        time_chunks=2,
    )