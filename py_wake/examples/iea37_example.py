from py_wake.wake_models.noj import NOJ
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
import time
from py_wake.aep_calculator import AEPCalculator


def run_NOJ():
    windTurbines = IEA37_WindTurbines()
    site = IEA37Site(64)
    wake_model = NOJ(site, windTurbines)
    aep_calculator = AEPCalculator(wake_model)

    x, y = site.initial_position.T
    print(aep_calculator.calculate_AEP(x, y).sum())


def main():
    if __name__ == '__main__':
        run_NOJ()


main()
