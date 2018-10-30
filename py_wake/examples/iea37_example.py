import numpy as np
from py_wake.aep._aep import AEP

from py_wake.wake_models.noj import NOJ
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37_Site
import time


def run_NOJ():
    windTurbines = IEA37_WindTurbines()
    site = IEA37_Site(64)
    wake_model = NOJ(windTurbines)
    aep = AEP(site, windTurbines, wake_model)

    x, y = site.initial_position.T
    t = time.time()
    N = 5
    for _ in range(N):
        print(aep.calculate_AEP(x, y).sum())
    print((time.time() - t) / N)


def main():
    if __name__ == '__main__':
        run_NOJ()


main()
