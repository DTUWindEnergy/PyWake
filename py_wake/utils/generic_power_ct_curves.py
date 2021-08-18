import numpy as np
from numpy import newaxis as na
from scipy.optimize.zeros import newton
import matplotlib.pyplot as plt


def standard_power_ct_curve(power_norm, diameter, turbulence_intensity=.1,
                            rho=1.225, max_cp=.49, constant_ct=.8,
                            gear_loss_const=.01, gear_loss_var=.014, generator_loss=0.03, converter_loss=.03,
                            wsp_lst=np.arange(0.1, 25, .1)):
    """Generate standard power curve, extracted from WETB (original extracted from excel sheet made by Kenneth Thomsen)

    Parameters
    ----------
    power_norm : int or float
        Nominal power [kW]
    diameter : int or float
        Rotor diameter [m]
    turbulence_intensity : float
        Turbulence intensity [%]
    rho : float, optional
        Density of air [kg/m^3], default is 1.225
    max_cp : float, optional
        Maximum power coefficient
    constant_ct : float, optional
        Ct value in constant-ct region
    gear_loss_const : float, optional
        Constant gear loss [% of power_norm in kW]
    gear_loss_var : float, optional
        Variable gear loss [%]
    generator_loss : float, optional
        Generator loss [%]
    converter_loss : float, optional
        converter loss [%]
    """

    area = (diameter / 2) ** 2 * np.pi

    sigma_lst = wsp_lst * turbulence_intensity

    p_aero = .5 * rho * area * wsp_lst ** 3 * max_cp / 1000

    # calc power - gear, generator and conv loss
    gear_loss = gear_loss_const * power_norm + gear_loss_var * p_aero
    p_gear = p_aero - gear_loss
    p_gear[p_gear < 0] = 0
    p_generator_loss = generator_loss * p_gear
    p_gen = p_gear - p_generator_loss
    p_converter_loss = converter_loss * p_gen
    p_raw = p_gen - p_converter_loss
    p_raw[p_raw > power_norm] = power_norm

    if turbulence_intensity > 0:
        sigma = sigma_lst[:, na]
        ndist = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(wsp_lst - wsp_lst[:, na]) ** 2 / (2 * sigma ** 2))
#         for i in range(len(wsp_lst)):
#             ndist[i, i * 2 + 1:] = 0
        power_lst = (ndist * p_raw).sum(1) / ndist.sum(1)
    else:
        power_lst = p_raw

    p_gen = power_lst / (1 - converter_loss)
    p_gear = p_gen / (1 - generator_loss)
    p_aero = (p_gear + gear_loss_const * power_norm) / (1 - gear_loss_var)
    cp = p_aero * 1000 / (.5 * rho * area * wsp_lst ** 3)
    # cp is too high at low ws due to constant loss, so limit to 16/27
    cp = np.minimum(cp, 16 / 27)

    def cp2ct(cp):
        # solve cp = 4 * a * (1 - a)**2 for a
        y = 27.0 / 16.0 * cp
        a = 2.0 / 3.0 * (1 - np.cos(np.arctan2(2 * np.sqrt(y * (1.0 - y)), 1 - 2 * y) / 3.0))
        return 4 * a * (1 - a)

    ct_lst = cp2ct(cp)

    # scale ct, such that the constant region (~cut-in to rated) equals <constant_ct>
    # First part (~0-2m/s) is constant at 8/9 due to cp limit and must be disregarded
    ct_below_lim = np.where(ct_lst < 8 / 9 - 1e-6)[0][0]
    # find index of most constant ct after the disregarded 8/9 region
    constant_ct_idx = ct_below_lim + np.argmin(np.abs(np.diff(ct_lst[ct_below_lim:])))
    if ct_lst[constant_ct_idx] < cp2ct(max_cp):
        # if TI is high, then there is no constant region
        constant_ct_idx = 0
    f = constant_ct / ct_lst[constant_ct_idx]
    ct_lst = np.minimum(8 / 9, ct_lst * f)
    return wsp_lst, power_lst, ct_lst


def main():
    if __name__ == '__main__':

        import matplotlib.pyplot as plt
        u = np.linspace(0., 30, 500)
        ax1 = plt.gca()
        ax2 = plt.twinx()
        for ti in [0, .1, .2, .25, .3, .5]:
            u, p, ct = standard_power_ct_curve(10000, 178.3, ti)
            ax1.plot(u, p, label='TI=%s' % ti)
            ax2.plot(u, ct, '--')
        ax1.legend(loc='center right')
        plt.show()


main()
