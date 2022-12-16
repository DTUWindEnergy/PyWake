import numpy as np
from py_wake.site.distance import StraightDistance
from numpy import newaxis as na
from py_wake.utils.grid_interpolator import GridInterpolator


class ISONoiseModel:

    def __init__(self, src_x, src_y, src_h, freqs, sound_power_level, elevation_function=None):
        """ISONoise model based on

        DSF/ISO/DIS 9613-2
        Acoustics – Attenuation of sound during propagation – Part 2:
        Engineering method for the prediction of sound pressure levels outdoors

        DS/ISO 9613-1:1993
        Akustik. Måling og beskrivelse af ekstern støj. Lydudbredelsesdæmpning udendørs. Del 1:
        Metode til beregning af luftabsorption

        The code is a vectorized version of the implementaion made by Camilla Marie Nyborg <cmny@dtu.dk>

        The model models the sound pressure level from a number of sound sources at a number of receivers taking into
        account
        - spherical geometrical spreading
        - ground reflection/absorption
        - atmospheric absorption (DS/ISO 9613-1:1993)


        Parameters
        ----------
        src_x : array_like
            x coordinate of sound sources
        src_y : array_like
            y coordinate of sound sources
        src_h : array_like or float
            height of sound sources (typically wind turbine hub height)
        freqs : array_like
            Frequencies of sound_power_level
        sound_power_level : array_like
            Emitted sound power level, dim=(n_sources, n_freq)

        """

        if elevation_function:
            src_h += elevation_function(src_x, src_y)
        self.elevation_function = elevation_function
        self.src_x, self.src_y = src_x, src_y
        self.src_h = np.zeros_like(src_x) + src_h
        self.n_src = len(src_x)
        self.distance = StraightDistance()
        self.freqs = freqs
        self.sound_power_level = sound_power_level

    def ground_eff(self, ground_distance_ij, distance_ij, ground_type):
        # Ground effects ISO

        freqs_init = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 8000.0])

        src_h_ij = self.src_h[:, na]
        rec_h_ij = self.distance.dst_h_j[na]

        hei_check_ij = 30.0 * (src_h_ij + rec_h_ij)

        q_ij = np.where(ground_distance_ij <= hei_check_ij, 0, 1 - hei_check_ij / ground_distance_ij)

        Gs = Gr = Gm = ground_type

        # for f=63Hz Am =-3*q, for other frequencies Am = -3 * q * (1-Gm)
        fak = np.where(freqs_init == 63.0, 1, (1 - Gm))
        Am_fij = -3.0 * q_ij[na] * fak[:, na, na]

        # 125 Hz
        a_source_ij = 1.5 + 3.0 * np.exp(-0.12 * (src_h_ij - 5.0)**2) * (1.0 - np.exp(-ground_distance_ij / 50.0)) + \
            5.7 * np.exp(-0.09 * src_h_ij**2) * \
            (1.0 - np.exp(-2.8 * (10.0**(-6.0)) * (ground_distance_ij**2)))
        a_rec_ij = 1.5 + 3.0 * np.exp(-0.12 * (rec_h_ij - 5.0)**2) * (1.0 - np.exp(-ground_distance_ij / 50.0)) + \
            5.7 * np.exp(-0.09 * rec_h_ij**2) * (1.0 - np.exp(-2.8 * (10.0**(-6.0)) * (ground_distance_ij**2)))

        # 250 Hz
        b_source_ij = 1.5 + 8.6 * np.exp(-0.09 * src_h_ij**2) * (1.0 - np.exp(-ground_distance_ij / 50.0))
        b_rec_ij = 1.5 + 8.6 * np.exp(-0.09 * rec_h_ij**2) * (1.0 - np.exp(-ground_distance_ij / 50.0))

        # 500 Hz
        c_source_ij = 1.5 + 14.0 * np.exp(-0.46 * src_h_ij**2) * (1.0 - np.exp(-ground_distance_ij / 50.0))
        c_rec_ij = 1.5 + 14.0 * np.exp(-0.46 * rec_h_ij**2) * (1.0 - np.exp(-ground_distance_ij / 50.0))

        # 1000 Hz
        d_source_ij = 1.5 + 5.0 * np.exp(-0.9 * src_h_ij**2) * (1.0 - np.exp(-ground_distance_ij / 50.0))
        d_rec_ij = 1.5 + 5.0 * np.exp(-0.9 * rec_h_ij**2) * (1.0 - np.exp(-ground_distance_ij / 50.0))

        zeros = np.zeros_like(ground_distance_ij)
        As_fij = np.array([zeros - 1.5,  # 63Hz
                           -1.5 + Gs * a_source_ij,  # 125Hz
                           -1.5 + Gs * b_source_ij,  # 250Hz
                           -1.5 + Gs * c_source_ij,  # 500Hz
                           -1.5 + Gs * d_source_ij,  # 1000Hz
                           zeros - 1.5 * (1.0 - Gs),  # 2000Hz
                           zeros - 1.5 * (1.0 - Gs)  # 8000Hz
                           ])

        Ar_fij = np.array([zeros - 1.5,  # 63 Hz
                           -1.5 + Gr * a_rec_ij,  # 125 Hz
                           -1.5 + Gr * b_rec_ij,  # 250 Hz
                           -1.5 + Gr * c_rec_ij,  # 500 Hz
                           -1.5 + Gr * d_rec_ij,  # 1000 Hz
                           zeros - 1.5 * (1.0 - Gr),  # 2000 Hz
                           zeros - 1.5 * (1.0 - Gr)  # 8000 Hz
                           ])

        # Interpolation to other frequencies are actually not following the standard completely.
        ip = GridInterpolator([freqs_init], np.moveaxis([As_fij, Ar_fij, Am_fij], 0, -1))

        # interpolate to freq and sum up As+Ar+Am
        Agr_ijf = np.moveaxis(ip(np.array([self.freqs]).T).sum(-1), 0, -1)

        # Area of sphere: 4pi R^2
        # 10*log_10(4pi) ~ 11
        # 10 * log_10(R^2) = 20 * log_10(R) =
        # reference area = 1.0 m^2
        Adiv_ij = 20.0 * np.log10(distance_ij / 1.0) + 11.0  # The geometrical spreading (divergence)

        ISO_ground_ijf = - Adiv_ij[:, :, na] - Agr_ijf

        return ISO_ground_ijf

    def atmab(self, distance_ij, T0, RH0):
        # Atmospheric absorption
        T_0 = 293.15
        T_01 = 273.16
        T = T0 + 273.15
        p_s0 = 1.0

        psat = p_s0 * 10.0**(-6.8346 * (T_01 / T)**1.261 + 4.6151)
        h = p_s0 * (RH0) * (psat / p_s0)
        F_rO = 1.0 / p_s0 * (24.0 + 4.04 * (10**4.0) * h * (0.02 + h) / (0.391 + h))
        F_rN = 1.0 / p_s0 * (T_0 / T)**(0.5) * (9.0 + 280 * h *
                                                np.exp(-4.17 * ((T_0 / T)**(1.0 / 3.0) - 1.0)))
        alpha_ps = self.freqs**2.0 / p_s0 * (1.84 * (10**(-11)) * (T / T_0)**(0.5) + (T / T_0)**(-5.0 / 2.0) *
                                             (0.01275 * np.exp(-2239.1 / T) / (F_rO + self.freqs**2.0 / F_rO) +
                                              0.1068 * np.exp(-3352 / T) / (F_rN + self.freqs**2.0 / F_rN)))

        ISO_alpha = alpha_ps[na, na] * 20.0 / np.log(10.0) * distance_ij[:, :, na]
        return ISO_alpha

    def transmission_loss(self, rec_x, rec_y, rec_h, ground_type, Temp, RHum):
        # transmission loss = ground effects + atmospheric absorption
        rec_h = np.zeros_like(rec_x) + rec_h
        if self.elevation_function:
            rec_h += self.elevation_function(rec_x, rec_y)
        self.distance.setup(self.src_x, self.src_y, self.src_h, [rec_x, rec_y, rec_h])
        ground_distance_ij = np.sqrt(self.distance.dx_ij**2 + self.distance.dy_ij**2)
        distance_ij = np.sqrt(self.distance.dx_ij**2 + self.distance.dy_ij**2 + self.distance.dh_ij**2)

        atm_abs_ijf = self.atmab(distance_ij, T0=Temp, RH0=RHum)  # The atmospheric absorption term
        ground_eff_ijf = self.ground_eff(ground_distance_ij, distance_ij, ground_type)

        return ground_eff_ijf - atm_abs_ijf  # Delta_SPL

    def __call__(self, rec_x, rec_y, rec_h, Temp, RHum, ground_type):
        """Calculate the sound pressure level at a list of reveicers

        Parameters
        ----------
        rec_x : array_like
            x coordinate, [m], of receivers
        rec_y : array_like
            y coordinate, [m], of receivers
        rec_h : array_like or float
            height, [m], of receivers (typically 2m)
        Temp : float
            Temperature (deg celcius)
        RHum : float
            Relative humidity (%)
        ground_type : float
            Factor describing the amount of ground absorption, 0=hard reflective ground,
            e.g. paving, water, ice and concrete. 1= soft porous absorbing ground,
            e.g. grass, trees, other vegetation and farming land

        Returns
        -------
        total_spl : array_like
            Total sound pressure level at each receiver
        spl : array_like
            Sound pressure level of freqs, dim=(n_receiver, n_freq)
        """

        # Computing total transmission loss
        Delta_SPL_ijf = self.transmission_loss(rec_x, rec_y, rec_h, ground_type, Temp, RHum)
        sound_power_ijxxf = self.sound_power_level[:, na]
        I, J, F = Delta_SPL_ijf.shape
        shape = [I, J] + [1] * (len(sound_power_ijxxf.shape) - len(Delta_SPL_ijf.shape)) + [F]
        Delta_SPL_ijxxf = Delta_SPL_ijf.reshape(shape)

        # Add negative transmission loss to emitted sound and sum over sources to get sound pressure level
        spl_jxxf = 10.0 * np.log10(np.sum(10.0**((self.sound_power_level[:, na] + Delta_SPL_ijxxf) / 10.0), axis=0))

        # Sum over frequencies to get total sound pressure level
        total_spl_jxx = 10.0 * np.log10(np.sum(10.0**(spl_jxxf / 10.0), axis=-1))
        return total_spl_jxx, spl_jxxf


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from py_wake.deficit_models.gaussian import ZongGaussian
        from py_wake.flow_map import XYGrid
        from py_wake.turbulence_models.crespo import CrespoHernandez

        from py_wake.site._site import UniformSite
        from py_wake.examples.data.swt_dd_142_4100_noise.swt_dd_142_4100 import SWT_DD_142_4100
        from py_wake.utils.layouts import rectangle
        from py_wake.utils.plotting import setup_plot

        wt = SWT_DD_142_4100()
        wfm = ZongGaussian(UniformSite(), wt, turbulenceModel=CrespoHernandez())
        x, y = rectangle(5, 5, 5 * wt.diameter())
        sim_res = wfm(x, y, wd=270, ws=8, mode=0)
        nm = sim_res.noise_model()
        ax1, ax2, ax3 = plt.subplots(1, 3, figsize=(16, 4))[1]
        sim_res.flow_map().plot_wake_map(ax=ax1)

        ax1.plot([x[0]], [1000], '.', label='Receiver 1')
        ax1.plot([x[-1]], [1000], '.', label='Receiver 2')
        ax1.legend()
        total_sp_jlk, spl_jlkf = nm(rec_x=[x[0], x[-1]], rec_y=[1000, 1000], rec_h=2, Temp=20, RHum=80, ground_type=0.0)
        ax2.plot(nm.freqs, spl_jlkf[0, 0, 0], label='Receiver 1')
        ax2.plot(nm.freqs, spl_jlkf[1, 0, 0], label='Receiver 2')
        setup_plot(xlabel='Frequency [Hz]', ylabel='Sound pressure level [dB]', ax=ax2)
        plt.tight_layout()

        nmap = sim_res.noise_map(grid=XYGrid(x=np.linspace(-1000, 5000, 100), y=np.linspace(-1000, 1000, 50), h=2))
        nmap['Total sound pressure level'].squeeze().plot(ax=ax3)
        wt.plot(x, y, ax=ax3)

        plt.show()


main()
