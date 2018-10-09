import pytest

import numpy as np
import pywt
from pytorch_wavelets import DWTForward, DWTInverse
import torch
PRECISION_DECIMAL = 3

HAVE_GPU = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(HAVE_GPU is False,
                                reason='Need a gpu to test cuda')


@pytest.mark.parametrize("wave, J", [
    ('db1', 1), ('db1', 3), ('db3', 1), ('db3', 2),
    ('db3', 3), ('bior2.4', 2)
])
def test_equal(wave, J):
    x = torch.randn(5, 4, 64, 64).cuda()
    dwt = DWTForward(4, J, wave).cuda()
    iwt = DWTInverse(4, wave).cuda()
    yl, yh = dwt(x)
    x2 = iwt((yl, yh))

    coeffs = pywt.wavedec2(
        x.detach().cpu(), wave, level=J, axes=(-2,-1), mode='reflect')
    np.testing.assert_array_almost_equal(
        x.detach().cpu(), x2.detach().cpu(), decimal=4)
    np.testing.assert_array_almost_equal(
        yl.detach().cpu(), coeffs[0], decimal=4)
    for j in range(J):
        for b in range(3):
            np.testing.assert_array_almost_equal(
                coeffs[J-j][b], yh[j][:,:,b].detach().cpu(), decimal=4)


@pytest.mark.parametrize("wave, J", [
    ('db1', 1), ('db1', 3), ('db3', 1), ('db3', 2),
    ('db3', 3), ('bior2.4', 2)
])
def test_ok(wave, J):
    x = torch.randn(5, 4, 64, 64)
    dwt = DWTForward(4, J, wave)
    iwt = DWTInverse(4, wave)
    yl, yh = dwt(x)
    x2 = iwt((yl, yh))
    # Can have data errors sometimes
    assert yl.is_contiguous()
    for j in range(J):
        assert yh[j].is_contiguous()
    assert x2.is_contiguous()
