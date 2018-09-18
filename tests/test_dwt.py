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


@pytest.mark.parametrize("wave, J, j", [
    ('db1', 1, 0), ('db1', 2, 1), ('db2', 2, 0), ('db3', 3, 2)
])
def test_commutativity(wave, J, j):
    C = 3
    Y = torch.randn(4, C, 128, 128, requires_grad=True)
    dwt = DWTForward(C=C,J=J, wave=wave)
    iwt = DWTInverse(C=C, wave=wave)
    coeffs = dwt(Y)
    coeffs_zero = dwt(torch.zeros_like(Y))
    # Set level j LH to be nonzero
    coeffs_zero[1][j][:,:,0] = coeffs[1][j][:,:,0]
    ya = iwt(coeffs_zero)
    # Set level j HL to also be nonzero
    coeffs_zero[1][j][:,:,1] = coeffs[1][j][:,:,1]
    yab = iwt(coeffs_zero)
    # Set level j LH to be nonzero
    coeffs_zero[1][j][:,:,0] = torch.zeros_like(coeffs[1][j][:,:,0])
    yb = iwt(coeffs_zero)
    # Set level j HH to also be nonzero
    coeffs_zero[1][j][:,:,2] = coeffs[1][j][:,:,2]
    ybc = iwt(coeffs_zero)
    # Set level j HL to be nonzero
    coeffs_zero[1][j][:,:,1] = torch.zeros_like(coeffs[1][j][:,:,1])
    yc = iwt(coeffs_zero)
    np.testing.assert_array_almost_equal((ya+yb).detach(), yab.detach(), decimal=4)
    np.testing.assert_array_almost_equal((yc+yb).detach(), ybc.detach(), decimal=4)
