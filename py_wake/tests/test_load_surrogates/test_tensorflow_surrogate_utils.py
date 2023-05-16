from py_wake.utils.tensorflow_surrogate_utils import TensorFlowModel
from py_wake.examples.data import example_data_path
from py_wake import np
from numpy import newaxis as na
import pytest
from py_wake.tests import npt


def test_TensorflowSurrogate():
    surrogate = TensorFlowModel.load_h5(example_data_path +
                                        "iea34_130rwt/one_turbine/electrical_power_operating.h5")

    assert surrogate.input_names == ['ws', 'ti', 'shear']
    assert surrogate.output_names[0] == "generator_servo inpvec   2  2: pelec [w]"
    # assert surrogate.input_space == {'ti': (6.94521e-05, 0.5157860142),
    #                                  'ws': (4.0056388753, 24.9807585868),
    #                                  'shear': (-0.0997070313, 0.4994140625)}
    assert surrogate.metadata['wind_speed_cut_in'] == 4.0
    assert surrogate.metadata['wind_speed_cut_out'] == 25.0

#
# def test_bounds_warning():
#     surrogate = TensorFlowModel.load_h5(example_data_path +
#                                         "iea34_130rwt/one_turbine/electrical_power_operating.h5")
#
#     import warnings
#     warnings.filterwarnings('error')
#     with pytest.raises(UserWarning, match='Input, ws, with value, 3.0 outside range 4.0056388753-24.9807585868'):
#         surrogate.predict_output(np.array([3., .1, .1])[na])
#
#     with pytest.raises(UserWarning, match='Input, ws, with value, 25.0 outside range 4.0056388753-24.9807585868'):
#         surrogate.predict_output(np.array([25., .1, .1])[na])
#
#     npt.assert_almost_equal(surrogate.predict_output(np.array([25., .1, .1])[na], bounds='ignore'), 3399991.35441946)
