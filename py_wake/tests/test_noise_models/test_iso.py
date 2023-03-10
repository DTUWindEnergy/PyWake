import numpy as np
from py_wake.noise_models.iso import ISONoiseModel
from py_wake.tests import npt
from numpy import newaxis as na
from py_wake.deficit_models.gaussian import BastankhahGaussian, BastankhahGaussianDeficit
from py_wake.examples.data.swt_dd_142_4100_noise.swt_dd_142_4100 import SWT_DD_142_4100
from py_wake.site._site import UniformSite
from py_wake.utils.layouts import rectangle
from py_wake.utils.plotting import setup_plot
from py_wake.flow_map import XYGrid
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind


def test_iso_noise_model():

    Temp = 20  # Temperature
    RHum = 80.0  # Relative humidity

    rec_x = [2000, 3000, 4000]  # receiver xy-position
    rec_y = [0, 0, 0]

    source_x = [0, 0]  # source xy-position
    source_y = [0, 500]

    freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])

    # If the source and receiver are placed in complex
    # terrain then the height of the terrain should be added to these heights.
    z_hub = 100  # source height
    z_rec = 2  # receiver height

    ground_type = 1  # Ranging from 0 to 1 (hard to soft ground)

    sound_power_level = np.array([[89.4, 93.6, 97.2, 98.6, 101., 102.3, 96.7, 84.1],
                                  [89.2, 93.2, 96.2, 97.6, 100., 101.3, 95.7, 83.1]])
    iso = ISONoiseModel(src_x=source_x, src_y=source_y, src_h=z_hub, freqs=freqs, sound_power_level=sound_power_level)

    Delta_SPL = iso.transmission_loss(rec_x, rec_y, z_rec, ground_type, Temp, RHum)
    delta_SPL_ref = [[[-74.18868997998351, -82.6237111823142, -85.106899362965, -84.78075941279131, -87.47936788035022,
                       -95.0531580482448, -119.90461076531315, -216.18076560019855],
                      [-77.78341234414849, -86.43781608834009, -89.6588031834609, -91.0544347291193, -96.14098255652445,
                       -107.56227940740277, -144.8146479691885, -289.1327626666617],
                      [-79.65387282494626, -89.23269240205617, -93.19182772534491, -96.30991899366798, -103.78536068849834,
                       -119.05570214846978, -168.71394345143312, -361.09322135290586]],
                     [[-74.4562086405804, -82.90475257479693, -85.43331381258002, -85.21311520726341, -88.05865461649006,
                       -95.86918353475227, -121.48366996751929, -220.71586737235828],
                      [-77.90553623031623, -86.56901853514375, -89.82054728871552, -91.28744741509202, -96.47283823614013,
                         -108.0533933516973, -145.81906793231144, -292.12576375254764],
                      [-79.70589475152278, -89.3092674680563, -93.29138286495427, -96.46309784962982, -104.01291072740128,
                         -119.40308086820056, -169.44754252350324, -363.32306335554176]]]
    npt.assert_array_almost_equal(delta_SPL_ref, Delta_SPL[:, :, 0, 0], 8)
    total_spl, spl = iso(rec_x, rec_y, z_rec, ground_type=ground_type, Temp=Temp, RHum=RHum)
    npt.assert_array_almost_equal([23.068816073636583, 18.034141554900494, 15.043308370722233],
                                  total_spl[:, 0, 0])


def test_iso_noise_map():
    wt = SWT_DD_142_4100()
    wfm = BastankhahGaussian(UniformSite(), wt, ct2a=ct2a_mom1d)
    x, y = rectangle(5, 5, 5 * wt.diameter())
    sim_res = wfm(x, y, wd=270, ws=8, mode=0)

    nm = sim_res.noise_model()
    total_spl_jlk, spl_jlkf = nm(rec_x=[x[0], x[-1]], rec_y=[1000, 1000], rec_h=2, Temp=20, RHum=80, ground_type=0.0)
    sim_res.noise_map()  # cover default grid
    nmap = sim_res.noise_map(grid=XYGrid(x=np.linspace(-1000, 4000, 100), y=np.linspace(-1000, 1000, 50), h=2))

    npt.assert_array_almost_equal(total_spl_jlk.squeeze(), [34.87997084, 29.56191044])
    npt.assert_array_almost_equal(
        spl_jlkf.squeeze(),
        [[2.21381744e+01, 2.60932044e+01, 2.88772888e+01, 2.84115953e+01, 2.82813953e+01, 2.55784584e+01,
          7.29692738e+00, -5.38230264e+01],
         [1.78459253e+01, 2.16729775e+01, 2.40686797e+01, 2.29153778e+01, 2.21888885e+01, 1.89631674e+01,
          -3.52596583e-02, -6.18125090e+01]])
    npt.assert_array_almost_equal(nmap['Total sound pressure level'].interp(x=[x[0], x[-1]], y=1000),
                                  total_spl_jlk, 2)
    npt.assert_array_almost_equal(nmap['Sound pressure level'].interp(x=[x[0], x[-1]], y=1000),
                                  spl_jlkf, 1)

    if 0:
        import matplotlib.pyplot as plt
        ax1, ax2, ax3 = plt.subplots(1, 3, figsize=(16, 4))[1]
        sim_res.flow_map().plot_wake_map(ax=ax1)

        ax1.plot([x[0]], [1000], '.', label='Receiver 1')
        ax1.plot([x[-1]], [1000], '.', label='Receiver 2')
        ax1.legend()

        ax2.plot(nm.freqs, spl_jlkf[0, 0, 0], label='Receiver 1')
        ax2.plot(nm.freqs, spl_jlkf[1, 0, 0], label='Receiver 2')
        setup_plot(xlabel='Frequency [Hz]', ylabel='Sound pressure level [dB]', ax=ax2)
        plt.tight_layout()

        nmap['Total sound pressure level'].squeeze().plot(ax=ax3)
        wt.plot(x, y, ax=ax3)

        plt.show()
