import numpy as np
from abc import ABC, abstractmethod


class SuperpositionModel(ABC):
    @abstractmethod
    def calc_effective_WS(self, WS_xxx, deficit_jxxx):
        """Calculate effective wind speed

        This method must be overridden by subclass

        Parameters
        ----------
        WS_xxx : array_like
            Local wind speed. xxx optionally includes destination turbine/site, wind directions, wind speeds
        deficit_jxxx : array_like
            deficit caused by source turbines(j) on xxx (see above)

        Returns
        -------
        WS_eff_xxx : array_like
            Effective wind speed for xxx (see WS_xxx)

        """


class SquaredSum(SuperpositionModel):
    def calc_effective_WS(self, WS_xxx, deficit_jxxx):
        return WS_xxx - np.sqrt(np.sum(deficit_jxxx**2, 0))


class LinearSum(SuperpositionModel):
    def calc_effective_WS(self, WS_xxx, deficit_jxxx):
        return WS_xxx - np.sum(deficit_jxxx, 0)


class MaxSum(SuperpositionModel):
    def calc_effective_WS(self, WS_xxx, deficit_jxxx):
        return WS_xxx - np.max(deficit_jxxx, 0)


class WeightedSum(SuperpositionModel):
    """
    Implemented according to the paper by:
    Haohua Zong and Fernando Porté-Agel
    A momentum-conserving wake superposition method for wind farm power prediction
    J. Fluid Mech. (2020), vol. 889, A8; doi:10.1017/jfm.2020.77
    """

    def calc_effective_WS(self, WS_xxx, centerline_deficit_jxxx,
                          convection_velocity_jxxx,
                          sigma_sqr_jxxx, cw_jxxx, hcw_jxxx, dh_jxxx):
        Ws = WS_xxx
        usc = centerline_deficit_jxxx
        uc = convection_velocity_jxxx
        sigma_sqr = sigma_sqr_jxxx
        cw = cw_jxxx
        hcw = hcw_jxxx
        dh = dh_jxxx
        # Determine non-centreline deficit ratio
        # Local deficit
        us = usc * np.exp(-1 / (2 * sigma_sqr) * cw**2)
        # Total cross-wind integrated deficit
        us_int = usc * 2 * np.pi * sigma_sqr
        # Combined qunatities
        Uc = Ws.copy()
        Uc_star = Ws.copy()
        Us = np.zeros_like(Ws)
        Us_int = np.zeros_like(Ws)

        # Initialize
        count = 0
        Uc_star = 10 * Uc
        # Iterate until combined convection velocity converges
        while (np.max(np.abs((Uc - Uc_star) / Uc_star)) > 1e-3) and (count < 10):
            # Initialize combined convection velocity
            if count == 0:
                Uc = np.max(uc, 0)
            else:
                Uc = Uc_star
            # Initialize and avoid division by zero
            ucn = np.ones_like(uc)
            Inz = Uc != 0
            ucn[:, Inz] = uc[:, Inz] / Uc[Inz]

            # Combined local deficit
            Us = np.sum(ucn * us, 0)
            # Combined deficit integrated to infinity in cross-wind direction
            Us_int = np.sum(ucn * us_int, 0)

            # Calculate the integral of Us**2
            sum1, sum2 = np.zeros_like(Us), np.zeros_like(Us)
            n_wt = us.shape[0]
            for j in np.arange(0, n_wt):
                sum1 += (ucn[j] * usc[j])**2 * np.pi * sigma_sqr[j]
            if n_wt > 0:
                sum2 = np.zeros_like(Us)
                for j in range(n_wt):
                    for k in np.arange(j + 1, n_wt):
                        cross_sigma_jk = np.zeros_like(Ws)
                        s1, s2 = sigma_sqr[j], sigma_sqr[k]
                        # Distance between turbines
                        w2w_hcw = np.abs(hcw[j] - hcw[k])
                        w2w_dh = np.abs(dh[j] - dh[k])
                        cross_sigma_jk = 2 * np.exp(-(w2w_hcw**2 + w2w_dh**2) /
                                                    (2 * (s1 + s2))) * np.pi * s1 * s2 / (s1 + s2)
                        sum2 += 2 * (ucn[j] * usc[j]) * (ucn[k] * usc[k]) * cross_sigma_jk

            # Avoid division by zero
            Us_int[Us_int == 0] = 1
            # Update combined convection velocity
            Uc_star = Ws - (sum1 + sum2) / Us_int

            count += 1

        return Ws - Us
