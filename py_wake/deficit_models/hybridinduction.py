import numpy as np
from numpy import newaxis as na
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit2020
from py_wake.deficit_models.vortexdipole import VortexDipole
from py_wake.deficit_models import BlockageDeficitModel


class HybridInduction(BlockageDeficitModel):
    """
    The idea behind this model originates from [2,3], which advocates to
    combine near-rotor and farfield approximations of a rotor's induced
    velocities. Whereas in [1,2] the motivation is to reduce the computational
    effort, here the already very fast self-similar model [1] is combined with
    the vortex dipole approximation in the far-field, as the self-similar one
    is optimized for the near-field (r/R > 6, x/R < 1) and misses the
    acceleration around the wake for x/R > 0. The combination of both allows
    capturing the redistribution of energy by blockage. Location at which to
    switch from near-rotor to far-field can be altered though by setting
    switch_radius.
    References:
        [1] N. Troldborg, A.R. Meyer Fortsing, Wind Energy, 2016
        [2] Emmanuel Branlard et al 2020 J. Phys.: Conf. Ser. 1618 062036
        [3] Branlard, E, Meyer Forsting, AR. Wind Energy. 2020; 23: 2068– 2086.
            https://doi.org/10.1002/we.2546
    """

    args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']

    def __init__(self, switch_radius=6.,
                 near_rotor=SelfSimilarityDeficit2020(), far_field=VortexDipole(), superpositionModel=None):
        BlockageDeficitModel.__init__(self, superpositionModel=superpositionModel)
        self.switch_radius = switch_radius
        self.near_rotor = near_rotor
        self.far_field = far_field

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):

        # deficit given by near-rotor model
        dnr_ijlk = self.near_rotor.calc_deficit(WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk)

        # deficit given by far-field model
        dff_ijlk = self.far_field.calc_deficit(WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk)

        # apply deficits in specified regions
        R_il = (D_src_il / 2)
        # radial distance from rotor centre
        r_ijlk = np.hypot(dw_ijlk, cw_ijlk)
        rcut_ijlk = (R_il * self.switch_radius)[:, na, :, na] * np.ones_like(dff_ijlk)
        # region where to apply the far-field deficit
        iff = (r_ijlk > rcut_ijlk) | (dw_ijlk > 0)
        deficit_ijlk = dnr_ijlk.copy()
        deficit_ijlk[iff] = dff_ijlk[iff]

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
        hi = HybridInduction()

        plt.figure()
        noj_hi = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                  superpositionModel=LinearSum(), blockage_deficitModel=hi)
        flow_map = noj_hi(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
        clevels = np.array([.6, .7, .8, .9, .95, .98, .99, .995, .998, .999, 1., 1.01, 1.02]) * 10.
        flow_map.plot_wake_map(levels=clevels)
        plt.title('Vortex Dipole (far-field) + Self-Similar (near-rotor)')
        plt.ylabel("Crosswind distance [y/R]")
        plt.xlabel("Downwind distance [x/R]")
        plt.show()

        # run wind farm simulation
        sim_res = noj_hi(x, y, wd=[0, 30, 45, 60, 90], ws=[5, 10, 15])

        # calculate AEP
        aep = sim_res.aep().sum()

        # plot wake map
        plt.figure()
        print(noj_hi)
        flow_map = sim_res.flow_map(wd=0, ws=10)
        flow_map.plot_wake_map(levels=clevels, plot_colorbar=False)
        plt.title('Vortex Dipole (far-field) + Self-Similar (near-rotor), AEP: %.3f GWh' % aep)
        plt.show()


main()
