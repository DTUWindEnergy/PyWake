from py_wake import np
from py_wake.utils.plotting import setup_plot


def ct2a_madsen(ct, ct2ap=np.array([0.2460, 0.0586, 0.0883])):
    """
    BEM axial induction approximation by
    Madsen, H. A., Larsen, T. J., Pirrung, G. R., Li, A., and Zahle, F.: Implementation of the blade element momentum model on a polar grid and its aeroelastic load impact, Wind Energ. Sci., 5, 1–27, https://doi.org/10.5194/wes-5-1-2020, 2020.
    """
    # Evaluate with Horner's rule.
    # ct2a_ilk = ct2ap[2] * ct_ilk**3 + ct2ap[1] * ct_ilk**2 + ct2ap[0] * ct_ilk
    return ct * (ct2ap[0] + ct * (ct2ap[1] + ct * ct2ap[2]))


def ct2a_mom1d(ct):
    """
    1D momentum, CT = 4a(1-a), with CT forced to below 1.
    """
    return 0.5 * (1. - np.sqrt(1. - np.minimum(1, ct)))


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        ct = np.linspace(0, 1.5)
        plt.plot(ct, ct2a_mom1d(ct), label='1d momentum')
        plt.plot(ct, ct2a_madsen(ct), label='Madsen')
        setup_plot(xlabel='Ct', ylabel='Induction factor, a')
        plt.show()


main()
