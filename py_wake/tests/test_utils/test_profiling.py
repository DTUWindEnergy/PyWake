import time
import numpy as np
from py_wake.utils.profiling import profileit
from py_wake.tests import npt


def test_profile_it():
    def f(t, mem):
        start = time.time()
        a = np.arange(1024 * 1024 * mem, dtype=np.byte)
        while time.time() < t + start:
            pass
    for _ in range(3):
        r, t, m = profileit(f)(.1, 100)
        npt.assert_allclose(t, 0.1, rtol=.2)
        npt.assert_allclose(m, 100, rtol=.2)
