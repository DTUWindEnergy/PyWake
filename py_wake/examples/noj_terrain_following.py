from py_wake import NOJ
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines

from py_wake.examples.data.ParqueFicticio._parque_ficticio import ParqueFicticioSite


def main():
    if __name__ == '__main__':
        site = ParqueFicticioSite()
        windTurbines = IEA37_WindTurbines()
        wf_model = NOJ(site, windTurbines)
        x, y = site.initial_position.T
        print(wf_model(x, y).aep())


main()
