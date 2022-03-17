import numpy as np
from numpy import newaxis as na
from py_wake.deficit_models import DeficitModel
from py_wake.deficit_models import BlockageDeficitModel
from py_wake.ground_models.ground_models import NoGround
from py_wake.utils.gradients import hypot, cabs
from py_wake.deficit_models.utils import a0


class VortexDipole(BlockageDeficitModel):
    """
    The vorticity originating from a wind turbine can be represented by a
    vortex dipole line (see Appendix B in [2]). The induction estimated by
    such a representation is very similar to the results given by the more
    complex vortex cylinder model in the far-field r/R > 6 [1,2]. The
    implementation follows the relationships given in [1,2]. This script is
    an adapted version of the one published by Emmanuel Branlard:
    https://github.com/ebranlard/wiz/blob/master/wiz/VortexDoublet.py
    References:
        [1] Emmanuel Branlard et al 2020 J. Phys.: Conf. Ser. 1618 062036
        [2] Branlard, E, Meyer Forsting, AR. Wind Energy. 2020; 23: 2068– 2086.
            https://doi.org/10.1002/we.2546
    """

    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']

    def __init__(self, sct=1.0, limiter=1e-10, exclude_wake=True, superpositionModel=None, groundModel=NoGround(),
                 upstream_only=False):
        DeficitModel.__init__(self, groundModel=groundModel)
        BlockageDeficitModel.__init__(self, upstream_only=upstream_only, superpositionModel=superpositionModel)
        # limiter to avoid singularities
        self.limiter = limiter
        # coefficient for scaling the effective forcing
        self.sct = sct
        # if used in a wind farm simulation, set deficit in wake region to
        # zero, as here the wake model is active
        self.exclude_wake = exclude_wake

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        """
        The analytical relationships can be found in [1,2].
        """
        R_il = (D_src_il / 2)
        # radial distance
        r_ijlk = hypot(dw_ijlk, cw_ijlk)
        # circulation/strength of vortex dipole Eq. (1) in [1]
        gammat_ilk = WS_ilk * 2. * a0(ct_ilk * self.sct)
        # Eq. (2) in [1], induced velocities away from centreline, however
        # here it is simplified. Effectively the equations are the same as for
        # a Rankine Half Body.
        # avoid devision by zero
        r_ijlk = np.where((r_ijlk / R_il[:, na, :, na]) < self.limiter, np.inf, r_ijlk)
        # deficit
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            np.warnings.filterwarnings('ignore', r'invalid value encountered in power')
            deficit_ijlk = gammat_ilk[:, na] / 4. * R_il[:, na, :, na]**2 * (-dw_ijlk / r_ijlk**3)

        if self.exclude_wake:
            # indices on rotor plane and in wake region
            iw = ((dw_ijlk / R_il[:, na, :, na] >= -self.limiter) &
                  (cabs(cw_ijlk) <= R_il[:, na, :, na])) * np.full(deficit_ijlk.shape, True)
            deficit_ijlk = np.where(iw, 0., deficit_ijlk)
            # Close to the rotor the induced velocities become unphysical and are
            # limited to the induction in the rotor plane estimated by BEM.
            ilim = deficit_ijlk > gammat_ilk[:, na] / 2.
            deficit_ijlk = np.where(ilim, gammat_ilk[:, na] / 2. * np.sign(deficit_ijlk), deficit_ijlk)

        return deficit_ijlk


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.superposition_models import LinearSum
        from py_wake.wind_farm_models import All2AllIterative
        from py_wake.deficit_models.no_wake import NoWakeDeficit
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()
        vd = VortexDipole()

        plt.figure()
        noj_vd = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=vd)
        flow_map = noj_vd(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
        clevels = np.array([.6, .7, .8, .9, .95, .98, .99, .995, .998, .999, 1., 1.01, 1.02]) * 10.
        flow_map.plot_wake_map(levels=clevels)
        plt.title('Vortex Dipole')
        plt.ylabel("Crosswind distance [y/R]")
        plt.xlabel("Downwind distance [x/R]")
        plt.show()

        # run wind farm simulation
        sim_res = noj_vd(x, y, wd=[0, 30, 45, 60, 90], ws=[5, 10, 15])

        # calculate AEP
        aep = sim_res.aep().sum()

        # plot wake map
        plt.figure()
        print(noj_vd)
        flow_map = sim_res.flow_map(wd=0, ws=10)
        flow_map.plot_wake_map(levels=clevels, plot_colorbar=False)
        plt.title('Vortex Dipole model, AEP: %.3f GWh' % aep)
        plt.show()


main()
