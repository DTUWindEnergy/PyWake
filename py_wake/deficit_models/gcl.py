import numpy as np
from py_wake.deficit_models.deficit_model import WakeDeficitModel
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.superposition_models import LinearSum
from py_wake.rotor_avg_models.rotor_avg_model import RotorCenter
na = np.newaxis


def my_power(term, factor):
    with np.warnings.catch_warnings():
        # if term is 0, exp(log(0))=0 as expected for a positive factor
        np.warnings.filterwarnings('ignore', r'divide by zero encountered in log')
        return np.exp(factor * np.log(term))


def get_r96(D, CT, TI):
    """Computes the wake radius at 9.6D downstream location of a turbine from empirical relation

    .. math::
        R_{9.6D} = a_1 \\exp (a_2 C_T^2 + a_3 C_T + a_4)  (b_1  TI + b_2)  D

    Inputs
    ----------
    D: float
        Wind turbine diameter
    CT: float
        Outputs WindTurbine object's thrust coefficient
    TI: float
        Ambient turbulence intensity
    pars: list
        GCL Model parameters [a1, a2, a3, a4, b1, b2]

    Returns
    -------
    R96: float
        Wake radius at 9.6D downstream location
    """
    a1, a2, a3, a4, b1, b2 = [0.435449861, 0.797853685, -0.124807893, 0.136821858, 15.6298, 1.0]
    R96 = a1 * (np.exp(a2 * CT * CT + a3 * CT + a4)) * (b1 * TI + b2) * D

    return R96


def get_Rw(x, R, TI, CT):
    """Computes the wake radius at a location.
    [1]-eq.3

    .. math::
        R_w = \\left(\\frac{105  c_1^2 }{2 \\pi}\\right)^{0.2} (C_T A (x + x_0))^{1/3}

    with A, the area, and x_0 and c_1 defined as

    .. math::
        x_0 = \\frac{9.6 D}{\\left(\\frac{2 R_96}{k D} \\right)^3 - 1}

        c_1 = \\left(\\frac{k D}{2}\\right)^{5/2}
              \\left(\\frac{105}{2 \\pi} \\right)^{-1/2}
              (C_T A x_0)^{-5/6}

    with k and m defined as

    .. math::
        k = \\sqrt{\\frac{m + 1}{2}}

        m = \\frac{1}{\\sqrt{1 - C_T}}

    Inputs
    ----------
    x: float or ndarray
        Distance between turbines and wake location in the wind direction
    R: float
        Wind turbine radius
    TI: float
        Ambient turbulence intensity
    CT: float
        Outputs WindTurbine object's thrust coefficient

    Returns
    -------
    Rw: float or ndarray
        Wake radius at a location
    """
    D = 2.0 * R
    Area = np.pi * R * R

    m = 1.0 / (np.sqrt(1.0 - CT))
    k = np.sqrt((m + 1.0) / 2.0)

    R96 = get_r96(D, CT, TI)
    x0 = (9.6 * D) / (my_power(2.0 * R96 / (k * D), 3.0) - 1.0)
    xx0 = x + x0
    term1 = my_power(k * D / 2.0, 2.5)
    term2 = my_power(105.0 / (2.0 * np.pi), -0.5)
    term3 = my_power(CT * Area * x0, -5.0 / 6.0)
    c1 = term1 * term2 * term3

    Rw = my_power(105.0 * c1 * c1 / (2.0 * np.pi), 0.2) * my_power(CT * Area * xx0, 1.0 / 3.0)

    Rw = np.where(x + x0 <= 0., 0., Rw)
    return Rw, xx0, c1


def get_dU(x, r, R, CT, TI):
    """Computes the wake velocity deficit at a location

    Inputs
    ----------
    x: float
        Distance between turbines and wake location in the wind direction
    r: float
        Radial distance between the turbine and the location
    R: float
        Wake producing turbine's radius [m]
    CT: float
        Outputs WindTurbine object's thrust coefficient
    TI: float
        Ambient turbulence intensity [-]
    order: int, optional

    Returns
    -------
    dU: float
        Wake velocity deficit at a location
    """

    CT = np.maximum(CT, np.finfo(float).eps)
    Area = np.pi * R * R
    Rw, xx0, c1 = get_Rw(x, R, TI, CT)
    c1s = c1 * c1

    term10 = (1 / 9)
    term20 = my_power(CT * Area / (xx0 * xx0), 1. / 3.)

    term310 = my_power(r, 1.5)
    term320 = 1.0 / np.sqrt(3. * c1s * CT * Area * xx0)
    term30 = term310 * term320
    term41 = my_power(35. / (2. * np.pi), .3)
    term42 = my_power(3. * c1s, -0.2)
    term40 = term41 * term42
    t4 = term30 - term40
    dU1 = -term10 * term20 * t4 * t4

    dU = dU1

    dU = np.where((Rw < r) | (x <= 0), 0, dU)
    return dU


class GCLDeficit(WakeDeficitModel):
    """
    Implemented according to:
            Larsen, G. C. (2009). A simple stationary semi-analytical wake model.
            Risoe National Laboratory for Sustainable Energy,
            Technical University of Denmark. Denmark.
            Forskningscenter Risoe. Risoe-R, No. 1713(EN)

    Description:
        based on an analytical solution of the thin shear layer approximation of the NS equations.
        The wake flow fields are assumed rotationally symmetric, and the rotor inflow fields
        are consistently assumed uniform.
        The effect of expansion is approximately accounted for by imposing suitable
        empirical downstream boundary conditions on the wake expansion that depend
        on the rotor thrust and the ambient turbulence conditions, respectively.
    """

    def __init__(self, use_effective_ws=False, use_effective_ti=False):
        self.use_effective_ws = use_effective_ws
        self.use_effective_ti = use_effective_ti
        self.args4deficit = ['WS_ilk', 'D_src_il', 'dw_ijlk', 'cw_ijlk', 'ct_ilk', 'TI_ilk']
        if use_effective_ws:
            self.args4deficit.append('WS_eff_ilk', )
        if use_effective_ti:
            self.args4deficit.append('TI_eff_ilk')

    def wake_radius(self, dw_ijlk, D_src_il, TI_ilk, ct_ilk, **kwargs):
        if self.use_effective_ti:
            TI_ilk = kwargs['TI_eff_ilk']
        with np.warnings.catch_warnings():
            # if term is 0, exp(log(0))=0 as expected for a positive factor
            np.warnings.filterwarnings('ignore', r'invalid value encountered in log')
            return get_Rw(x=dw_ijlk, R=(D_src_il / 2)[:, na, :, na], TI=TI_ilk[:, na], CT=ct_ilk[:, na])[0]

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, TI_ilk, **kwargs):
        if self.use_effective_ws:
            WS_ilk = kwargs['WS_eff_ilk']
        if self.use_effective_ti:
            TI_ilk = kwargs['TI_eff_ilk']
        eps = 1e-10
        dw_ijlk_gt0 = np.maximum(dw_ijlk, eps)
        R_src_il = D_src_il / 2.
        dU = -get_dU(x=dw_ijlk_gt0, r=cw_ijlk, R=R_src_il[:, na, :, na],
                     CT=ct_ilk[:, na], TI=TI_ilk[:, na])
        return WS_ilk[:, na] * dU * (dw_ijlk > eps)


class GCL(PropagateDownwind):
    def __init__(self, site, windTurbines, rotorAvgModel=RotorCenter(), superpositionModel=LinearSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=None):
        PropagateDownwind.__init__(self, site, windTurbines, wake_deficitModel=GCLDeficit(),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel,
                                   groundModel=groundModel)


class GCLLocal(PropagateDownwind):
    def __init__(self, site, windTurbines, rotorAvgModel=RotorCenter(), superpositionModel=LinearSum(),
                 deflectionModel=None, turbulenceModel=None, groundModel=None):

        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=GCLDeficit(use_effective_ws=True, use_effective_ti=True),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel,
                                   groundModel=groundModel)


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.turbulence_models import GCLTurbulence
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        wf_model = GCL(site, windTurbines)
        wf_model_local = GCLLocal(site, windTurbines, turbulenceModel=GCLTurbulence())
        # run wind farm simulation
        sim_res = wf_model(x, y)
        sim_res_local = wf_model_local(x, y)
        # calculate AEP
        aep = sim_res.aep().sum()
        aep_local = sim_res_local.aep().sum()

        # plot wake map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
        levels = np.arange(0, 10.5, 0.5)
        print(wf_model)
        flow_map = sim_res.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map(levels=levels, ax=ax1, plot_colorbar=False)
        flow_map.plot_windturbines(ax=ax1)
        ax1.set_title('Original Larsen, AEP: %.2f GWh' % aep)

        # plot wake map
        print(wf_model_local)
        flow_map = sim_res_local.flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map(levels=levels, ax=ax2, plot_colorbar=False)
        flow_map.plot_windturbines(ax=ax2)
        ax2.set_title('Local Larsen, AEP: %.2f GWh' % aep_local)

        plt.figure()
        flow_map.plot_ti_map(levels=np.arange(0, 1, .01))
        plt.title('TI map for GCLLocal with GCL turbulence model')
        plt.show()


main()
