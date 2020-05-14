from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm
import numpy as np


def main():
    if __name__ == '__main__':

        for n_wt in [9, 16, 36, 64]:
            x, y, aep_ref = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)
            if 0:
                import matplotlib.pyplot as plt
                plt.plot(x, y, '2k')
                for i, (x_, y_) in enumerate(zip(x, y)):
                    plt.annotate(i, (x_, y_))
                plt.axis('equal')
                plt.show()
            site = IEA37Site(n_wt)
            windTurbines = IEA37_WindTurbines(iea37_path + 'iea37-335mw.yaml')
            wake_model = IEA37SimpleBastankhahGaussian(site, windTurbines)

            aep = wake_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep(normalize_probabilities=True)

            # Compare to reference results provided for IEA task 37
            print(n_wt, aep_ref[0] * 1e-3, aep)


main()
