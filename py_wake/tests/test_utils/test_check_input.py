import pytest
from py_wake.utils.check_input import check_input
import numpy as np


def test_check_input():
    input_space = [(0, 1), (100, 200)]

    with pytest.raises(ValueError, match="Input, index_0, with value, 2 outside range 0-1"):
        check_input(input_space, np.array([(2, 150)]).T)

    with pytest.raises(ValueError, match="Input, index_1, with value, 50 outside range 100-200"):
        check_input(input_space, np.array([(1, 50)]).T)

    with pytest.raises(ValueError, match="Input, wd, with value, 250 outside range 100-200"):
        check_input(input_space, np.array([(1, 250)]).T, ['ws', 'wd'])

    check_input(input_space, np.array([(1, 200)]).T, ['ws', 'wd'])
