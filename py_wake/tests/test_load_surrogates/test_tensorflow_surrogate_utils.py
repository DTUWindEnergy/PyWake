from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
from py_wake.examples.data import example_data_path
import numpy as np
from numpy import newaxis as na
import pytest
from py_wake.tests import npt


def test_TensorflowSurrogate():
    surrogate = TensorflowSurrogate(example_data_path + "iea34_130rwt/one_turbine/electrical_power", 'operating')

    assert surrogate.input_channel_names == ['ws', 'ti', 'shear']
    assert surrogate.output_channel_name == "generator_servo inpvec   2  2: pelec [w]"
    assert surrogate.input_space == {'ti': (6.94521e-05, 0.5157860142),
                                     'ws': (4.0056388753, 24.9807585868),
                                     'shear': (-0.0997070313, 0.4994140625)}
    assert surrogate.wind_speed_cut_in == 4.0
    assert surrogate.wind_speed_cut_out == 25.0


def test_bounds_warning():
    surrogate = TensorflowSurrogate(example_data_path + "iea34_130rwt/one_turbine/electrical_power", 'operating')
    import warnings
    warnings.filterwarnings('error')
    with pytest.raises(UserWarning, match='Input, ws, with value, 3.0 outside range 4.0056388753-24.9807585868'):
        surrogate.predict_output(np.array([3., .1, .1])[na])

    with pytest.raises(UserWarning, match='Input, ws, with value, 25.0 outside range 4.0056388753-24.9807585868'):
        surrogate.predict_output(np.array([25., .1, .1])[na])

    assert surrogate.predict_output(np.array([25., .1, .1])[na], bounds='ignore') == 3399991.2
