import pytest

import numpy as np
import pywt
from pytorch_wavelets import DWTForward, DWTInverse
import torch


@pytest.mark.parametrize("wave, J", [
    ('db1', 1), ('db1', 3), ('db3', 1), ('db3', 2),
    ('db3', 3), ('bior2.4', 2)
])
def test_equal(wave, J):
    x = torch.randn(5, 4, 64, 64)
    dwt = DWTForward(4, J, wave)
    iwt = DWTInverse(4, wave)
    yl, yh = dwt(x)
    x2 = iwt((yl, yh))

    coeffs = pywt.wavedec2(x, wave, level=J, axes=(-2,-1), mode='reflect')
    np.testing.assert_array_almost_equal(x.detach(), x2.detach(), decimal=4)
    np.testing.assert_array_almost_equal(yl.detach(), coeffs[0], decimal=4)
    for j in range(J):
        for b in range(3):
            np.testing.assert_array_almost_equal(
                coeffs[J-j][b], yh[j][:,:,b].detach(), decimal=4)
