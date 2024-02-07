"""
Created on 24/03/2023

Description: Implementation of the Minimalistic Prediction Model developped by
    Jens N. Sørensen and Gunner C. Larsen

@author: David Fournely and Ariadna Garcia Montes

simplified and generalized by Mads M Pedersen
"""
from py_wake import np
from scipy.special import gamma, gammainc

from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
from py_wake.utils.layouts import farm_area
from py_wake.utils import fuga_utils
from py_wake.tests import npt
import os


class CorrectionFactorCalibrated():
    def __init__(self):
        import joblib
        self.PolynomialFeature_model = joblib.load(os.path.dirname(__file__) + '/PolynomialFeature_model')
        self.Regression_model = joblib.load(os.path.dirname(__file__) + '/Regression_model')

    def __call__(self, Uh0, sr, Nturb):
        a_input = [[Uh0, sr, np.sqrt(Nturb)]]
        xdata = self.PolynomialFeature_model.fit_transform(a_input)
        return self.Regression_model.predict(xdata)[0]


class MinimalisticPredictionModel():
    """Sørensen, J.N.; Larsen, G.C.
    A Minimalistic Prediction Model to Determine Energy Production and Costs of Offshore Wind Farms.
    Energies 2021, 14, 448. https://doi.org/10.3390/en14020448"""

    def __init__(self, correction_factor, latitude, CP, Uin, Uout):
        """
        Parameters
        ----------
        correction_factor : int, float or function
            Finite-size wind farm corrrection which multiplied with sqrt(Nturb) gives
            the number of wind turbines exposed to the free wind
        latitude : int or float
            latitude [deg] used to calculate the coriolis parameter
        CP : float, optional
            Wind turbine power coefficient
        Uin : int or float, optional
            Wind turbine cut-in wind speed
        Uout : int or float, optional
            Wind turbine cut-out wind speed
        rho : float, optional
            Air density
        """

        self.CP = CP
        self.Uin = Uin
        self.Uout = Uout
        omega = 2 * np.pi / (24 * 60 * 60)  # earth rotation speed
        self.f = 2 * omega * np.sin(np.deg2rad(latitude))
        self.correction_factor = correction_factor

    def predict(self, Pg, CT, D, H, z0, Aw, kw, Nturb, Area):
        """
        Inputs:
            Pg    - [W] Nameplate capacity (generator power)
            CT    - [-] Thrust coefficient
            D     - [m] Rotor diameter
            H     - [m] Tower height
            z0    - [m] roughness length
            Aw    - [m/s] Weibull scale parameter
            kw    - [-] Weibull shape parameter
            Nturb - [-] Number of turbines
            Area  - [m2] Area of wind farm

        Outputs:
            power - [Wh] Annual energy production of the wind farm
            ws_eff - [m/s] Effective mean wind speed including wakes
        """

        kappa = 0.4  # [-] Von Karman constant
        Uin, Uout = self.Uin, self.Uout

        # factor defined by Frandsen, should be used instead of f in eq 13 and 19 (typos in paper)
        fm = self.f * np.exp(4)
        delta = np.log(H / z0)  # eq 19

        # Mean spacing between wt in diameters, eq 8
        S = np.sqrt(Area) / (D * (np.sqrt(Nturb) - 1))

        # Rated wind speed [m/s], eq 4
        Ur = (8 * Pg / (1.225 * np.pi * D**2 * self.CP))**(1 / 3)

        # Power modeled as P = alpha * U^3 + beta, eq 1
        alpha = Pg / (Ur**3 - Uin**3)  # [(m/s)^-3] eq 2
        beta = -Pg * Uin**3 / (Ur**3 - Uin**3)  # [-], eq 2

        Uh0 = Aw * gamma(1 + 1 / kw)  # [m/s] Mean velocity at hub height
        Ctau = np.pi * CT / (8 * S * S)  # [-] Wake parameter, rotor ct smeared on WT area
        nu = np.sqrt(0.5 * Ctau) * D / (kappa**2 * H) * delta  # [-] wake eddy viscosity

        # Finite-size wind farm corrrection, section 2.5
        correction_factor = self.correction_factor
        if hasattr(correction_factor, '__call__'):
            correction_factor = correction_factor(Uh0, S, Nturb)
        Nfree = correction_factor * np.sqrt(Nturb)  # Number of wt exposed to the free wind

        # Geostrophic wind speed
        G_last = Uh0
        for n in range(10):
            G = Uh0 * (1 + np.log(G_last / (fm * H)) / delta)
            dG = abs(G - G_last)
            if dG < 1e-5:
                break
            G_last = G

        gam = np.log(G / (fm * H))  # eq 19

        # Mean velocity at hub height without wake effects from geostrophic wind
        Uh0 = G / (1 + gam / kappa * np.sqrt((kappa / delta)**2))  # eq 13, ct=0

        # Power without wake effects, eq 16 modified by
        # - add gamma(1 + 3 / kw) to cancel out normalization in scipy's gammainc
        # - gammainc terms swapped (typo in paper)
        def get_Py(Aw, Aw_out):  # Yearly power
            return alpha * Aw**3 * gamma(1 + 3 / kw) * (gammainc(1 + 3 / kw, (Ur / Aw)**kw) - gammainc(1 + 3 / kw, (Uin / Aw)**kw)) +\
                beta * (np.exp(-(Uin / Aw)**kw) - np.exp(-(Ur / Aw)**kw)) + \
                Pg * (np.exp(-(Ur / Aw)**kw) - np.exp(-(Uout / Aw_out)**kw))

        P_y = get_Py(Aw, Aw)

        # Mean velocity at hub height with wake effects
        z0_lo = z0  # / (1 - D / (2 * H))**(nu / (1 + nu))  # ???
        Uh = G / (1 + gam * np.sqrt(Ctau + (kappa / np.log(H / z0_lo))**2) / kappa)

        # eq 18. The paper states 3/2 instead of 3.2 which is either a typo or an initial guess
        # eps2 corresponds to eps(Uout) in paper and eps2(Ur)=eps1
        eps1 = (1 + gam / delta) / (1 + gam / kappa * np.sqrt(Ctau + (kappa / delta)**2))
        eps2 = (1 + gam / delta) / (1 + gam / kappa * np.sqrt(Ctau * (Ur / Uh)**3.2 + (kappa / delta)**2))

        # Power production with wake effects
        P_WFy = get_Py(eps1 * Aw, eps2 * Aw)

        power = ((Nturb - Nfree) * P_WFy + Nfree * P_y)
        ws_eff = ((Nturb - Nfree) * Uh + Nfree * Uh0) / Nturb
        return power, ws_eff


class MinimalisticWindFarmModel(WindFarmModel, MinimalisticPredictionModel):
    def __init__(self, site, windTurbines, correction_factor, latitude,
                 max_cp=None, ws_cutin=None, ws_cutout=None):
        """Minimalistic wind farm model that wraps the MinimalPredictionModel:

        Sørensen, J.N.; Larsen, G.C.
        A Minimalistic Prediction Model to Determine Energy Production and Costs of Offshore Wind Farms.
        Energies 2021, 14, 448. https://doi.org/10.3390/en14020448

        The model requires a minimal set of inputs

        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        correction_factor : int, float or function
            Finite-size wind farm corrrection which multiplied with sqrt(Nturb) gives
            the number of wind turbines exposed to the free wind
        latitude : int or float
            latitude [deg] used to calculate the coriolis parameter
        max_cp : float, optional
            Wind turbine power coefficient. Must be specified or exist in the windTurbine
        ws_cutin : int or float, optional
            Wind turbine cut-in wind speed. Defaults to 4 if not specified and not present in the windTurbine
        ws_cutout : int or float, optional
            Wind turbine cut-out wind speed. Defaults to 25 if not specified and not present in the windTurbine
        """
        WindFarmModel.__init__(self, site, windTurbines)
        max_cp = max_cp or windTurbines.max_cp
        ws_cutin = ws_cutin or getattr(windTurbines, 'ws_cutin', 4)
        ws_cutout = ws_cutout or getattr(windTurbines, 'ws_cutout', 25)

        MinimalisticPredictionModel.__init__(self, correction_factor, latitude, CP=max_cp,
                                             Uin=ws_cutin, Uout=ws_cutout)

    def calc_wt_interaction(self, x_ilk, y_ilk, h_i=None, type_i=0,
                            wd=None, ws=None, time=False,
                            n_cpu=1, wd_chunks=None, ws_chunks=None, **kwargs):

        rated_power = self.windTurbines.power([8, 12, 16]).max()
        ct_max = self.windTurbines.ct([6, 8, 12, 16]).max()
        area = farm_area(wt_x=x_ilk.mean((1, 2)), wt_y=y_ilk.mean((1, 2)))

        # Create LocalWind_omni with only one wind speed and wind direction
        localWind_omni = self.site.local_wind(x_ilk, y_ilk, h_i, wd=0, ws=10)
        localWind_omni['P_ilk'][:] = 1

        TI_ilk = kwargs.get('TI_ilk', localWind_omni['TI_ilk'])
        z0 = fuga_utils.z0(np.mean(TI_ilk), zref=np.mean(self.windTurbines.hub_height()), zeta0=0)[0]

        power_sector, ws_eff_sector = np.array([
            self.predict(Pg=rated_power,
                         CT=ct_max,
                         D=self.windTurbines.diameter(),
                         H=self.windTurbines.hub_height(),
                         z0=z0,
                         Aw=A_w,
                         kw=k_w,
                         Nturb=len(x_ilk),
                         Area=area)
            for A_w, k_w in zip(self.site.ds.Weibull_A.values[:-1], self.site.ds.Weibull_k.values[:-1])]).T

        f = self.site.ds.Sector_frequency.values[:-1]
        power = np.sum(power_sector * f)
        ws_eff = np.sum(f * ws_eff_sector)

        I, L, K = len(x_ilk), 1, 1
        WS_eff_ilk = np.full((I, L, K), ws_eff)
        power_ilk = np.full((I, L, K), power / len(x_ilk))
        TI_eff_ilk = localWind_omni['TI_ilk']
        ct_ilk = np.full((I, L, K), ct_max)
        kwargs_ilk = {'type_i': type_i, **kwargs}

        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind_omni, kwargs_ilk


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.hornsrev1 import Hornsrev1Site
        from py_wake.deficit_models.noj import NOJ
        from py_wake.wind_turbines.generic_wind_turbines import SimpleGenericWindTurbine
        from py_wake.deficit_models.noj import NOJLocal

        wt = SimpleGenericWindTurbine(name='Simple', diameter=80, hub_height=70, power_norm=2000)

        ti = fuga_utils.ti(z0=0.0001, zref=wt.hub_height(), zeta0=0)[0]

        site = Hornsrev1Site(ti=ti)
        x, y = site.initial_position.T
        for wfm, eff, ref in [(MinimalisticWindFarmModel(site, wt, 3, 55), 1, 547.179521),
                              (MinimalisticWindFarmModel(site, wt, CorrectionFactorCalibrated(), 55), .91, 586.5399377399355),
                              (NOJ(site, wt, k=0.032), .91, None),
                              (NOJLocal(site, wt), .91, None),
                              ]:
            res = wfm.aep(x, y) * eff
            print(res)
            if ref:
                npt.assert_allclose(res, ref, rtol=0.001)

        print(.160 * .412 * 24 * 365)  # reference from https://energynumbers.info/capacity-factors-at-danish-offshore-wind-farms


main()
