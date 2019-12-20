
import numpy as np
# , haar2d_decomp
import pywt
from numpy.testing import assert_, assert_allclose, assert_raises

from dwt import haar_decomposition, haar_recomposition


def test_haar_1D_transform():
    level = 1
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    x =[10,13,25,26,29,21,7,15] 
    assert isinstance(level, int), "provided {} is not int".format(level)

    if level == 1:
        cA_expect, cD_expect = pywt.dwt(x, 'db1')
        cA, cD = haar_decomposition(x)

        print(cA, cA_expect)
        print(cD, cD_expect)

        assert_allclose(cA, cA_expect, rtol=1e-5)
        # assert_allclose(cD, cD_expect, rtol=1e-5)

        x_rec = haar_recomposition(cA, cD)
        print(x_rec)
        assert_allclose(x_rec, x, rtol=1e-5)

    if level > 1:
        coeffs_expect = pywt.wavedec(x, 'db1', level)
        print(coeffs_expect)
        coeffs = haar_decomposition(x, level=level)
        print('\n')
        print(coeffs)
        # assert_allclose(coeffs, coeffs_expect, rtol=1e-5)

        x_rec = haar_recomposition(coeffs)
        print(x_rec)
        assert_allclose(x_rec, x, rtol=1e-5)


if __name__ == "__main__":
    test_haar_1D_transform()
