from py_wake.examples.data.hornsrev1 import wt_x, wt_y, HornsrevV80,\
    Hornsrev1Site
from py_wake import NOJ


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        wt = HornsrevV80()
        site = Hornsrev1Site()
        wt.plot(wt_x, wt_y)
        wf_model = NOJ(site, wt)
        aep = wf_model(wt_x, wt_y).aep()
        plt.title('AEP: %.1fGWh' % aep.sum())
        plt.show()


main()
