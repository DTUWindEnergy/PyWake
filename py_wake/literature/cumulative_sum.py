import numpy as np
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.superposition_models import CumulativeWakeSum
from py_wake.deficit_models.gaussian import NiayifarGaussianDeficit
from py_wake.rotor_avg_models import CGIRotorAvg
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine


class CumulativeWake(PropagateDownwind):
    """
    Wind farm model used in:
    Majid Bastankhah, Bridget L. Welch, Luis A. Martínez-Tossas, Jennifer King and Paul Fleming
    Analytical solution for the cumulative wake of wind turbines in wind farms
    J. Fluid Mech. (2021), vol. 911, A53, doi:10.1017/jfm.2020.1037
    """

    def __init__(self, site, windTurbines):

        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=NiayifarGaussianDeficit(ct2a=ct2a_mom1d, a=[0.31, 0.], ceps=.2,
                                                                             use_effective_ws=True, use_effective_ti=True,
                                                                             rotorAvgModel=CGIRotorAvg(21)),
                                   superpositionModel=CumulativeWakeSum(),
                                   turbulenceModel=CrespoHernandez(c=[0.66, 0.83, 0.03, -0.32]))


# NREL 5MW data extracted from paper
ct = np.array([
    [2.4967177242888408, 1.3884210526315786],
    [3.0021881838074402, 1.2684210526315787],
    [3.5010940919037203, 1.1652631578947366],
    [3.9999999999999996, 1.0799999999999998],
    [4.4989059080962805, 1.0105263157894733],
    [4.99781181619256, 0.9526315789473682],
    [5.496717724288842, 0.9052631578947368],
    [6.002188183807442, 0.8652631578947366],
    [6.501094091903721, 0.8315789473684209],
    [7, 0.8031578947368418],
    [7.49890590809628, 0.7789473684210524],
    [7.99124726477024, 0.7726315789473682],
    [8.496717724288839, 0.7726315789473682],
    [9.00218818380744, 0.7715789473684208],
    [9.494529540481402, 0.7621052631578944],
    [10.000000000000004, 0.7463157894736839],
    [10.49890590809628, 0.7305263157894736],
    [10.99781181619256, 0.7178947368421049]])
cp = np.array([
    [2.5016501650165015, 0.2354029062087188],
    [3.0033003300330035, 0.46208718626155887],
    [3.5049504950495054, 0.5476882430647292],
    [4.006600660066006, 0.5772787318361956],
    [4.5016501650165015, 0.5836195508586526],
    [5.003300330033003, 0.5788639365918098],
    [5.498349834983495, 0.5704095112285338],
    [5.999999999999998, 0.5593130779392339],
    [6.495049504950494, 0.548216644649934],
    [7.003300330033001, 0.5365918097754294],
    [7.498349834983497, 0.5260237780713343],
    [7.999999999999998, 0.5228533685601058],
    [8.5016501650165, 0.5223249669749009],
    [8.996699669966993, 0.5217965653896963],
    [9.491749174917492, 0.5175693527080582],
    [9.99339933993399, 0.5096433289299869],
    [10.495049504950492, 0.5017173051519155],
    [10.996699669966995, 0.4948480845442537]])


class nrel5mw(WindTurbine):
    def __init__(self, method='linear', rho=1.225, D=126., z0=90.):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        WindTurbine.__init__(self, name='nrel5mw', diameter=D, hub_height=z0,
                             powerCtFunction=PowerCtTabular(cp[:, 0], cp[:, 1] * 0.5 * rho * cp[:, 0]**3 * D**2 / 4. * np.pi, 'w',
                                                            ct[:, 1], method=method))
