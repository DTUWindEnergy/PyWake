from py_wake.examples.data.hornsrev1 import wt_x, wt_y, HornsrevV80,\
    Hornsrev1Site
from py_wake.wake_models.noj import NOJ
from py_wake.aep_calculator import AEPCalculator


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        wt = HornsrevV80()
        site = Hornsrev1Site()
        wt.plot(wt_x, wt_y)
        aep_calculator = AEPCalculator(NOJ(site, wt))
        print('AEP', aep_calculator.calculate_AEP(wt_x, wt_y)[0].sum())
        plt.show()


main()
