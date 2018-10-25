import numpy as np
from py_wake.aep._aep import AEP
from py_wake.examples.data.hornsrev_setup import wt_x, wt_y, HornsrevV80,\
    HornsrevSite
from py_wake.wake_models.noj import NOJ


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt

        plt.plot(wt_x, wt_y, '2k')
        for i, (x_, y_) in enumerate(zip(wt_x, wt_y)):
            plt.annotate(i, (x_, y_))
        plt.axis('equal')
        plt.show()
        wt = HornsrevV80()
        aep = AEP(HornsrevSite(), wt, NOJ(wt))
        import time
        t = time.time()
        print('AEP', aep.calculate_AEP(wt_x, wt_y)[0].sum())
        print('Computation time:', time.time() - t)


main()
