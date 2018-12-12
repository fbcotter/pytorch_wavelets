import numpy as np
import pytest
import pywt
from pytorch_wavelets import DWTForward, DWTInverse
import torch
from contextlib import contextmanager


HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')


@contextmanager
def set_double_precision():
    old_prec = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        yield
    finally:
        torch.set_default_dtype(old_prec)


@pytest.mark.parametrize("wave, J", [
    ('db1', 1), ('db1', 3), ('db3', 1), ('db3', 2),
    ('db3', 3), ('bior2.4', 2)
])
def test_equal(wave, J):
    x = torch.randn(5, 4, 64, 64).to(dev)
    dwt = DWTForward(J=J, wave=wave).to(dev)
    iwt = DWTInverse(wave=wave).to(dev)
    yl, yh = dwt(x)
    x2 = iwt((yl, yh))

    # Test the forward and inverse worked
    np.testing.assert_array_almost_equal(x.cpu(), x2.detach(), decimal=4)
    # Test it is the same as doing the PyWavelets wavedec with reflection
    # padding
    coeffs = pywt.wavedec2(x.cpu().numpy(), wave, level=J, axes=(-2,-1),
                           mode='reflect')
    np.testing.assert_array_almost_equal(yl.cpu(), coeffs[0], decimal=4)
    for j in range(J):
        for b in range(3):
            np.testing.assert_array_almost_equal(
                coeffs[J-j][b], yh[j][:,:,b].cpu(), decimal=4)


@pytest.mark.parametrize("wave, J", [
    ('db1', 1), ('db1', 3), ('db3', 1), ('db3', 2),
    ('db3', 3), ('bior2.4', 2)
])
def test_equal_double(wave, J):
    with set_double_precision():
        x = torch.randn(5, 4, 64, 64).to(dev)
        assert x.dtype == torch.float64
        dwt = DWTForward(J=J, wave=wave).to(dev)
        iwt = DWTInverse(wave=wave).to(dev)

    yl, yh = dwt(x)
    x2 = iwt((yl, yh))

    # Test the forward and inverse worked
    np.testing.assert_array_almost_equal(x.cpu(), x2.detach(), decimal=7)
    # Test it is the same as doing the PyWavelets wavedec with reflection
    # padding
    coeffs = pywt.wavedec2(x.cpu().numpy(), wave, level=J, axes=(-2,-1),
                           mode='reflect')
    np.testing.assert_array_almost_equal(yl.cpu(), coeffs[0], decimal=7)
    for j in range(J):
        for b in range(3):
            np.testing.assert_array_almost_equal(
                coeffs[J-j][b], yh[j][:,:,b].cpu(), decimal=7)


@pytest.mark.parametrize("wave, J", [
    ('db1', 1), ('db1', 3), ('db3', 1), ('db3', 2),
    ('db3', 3), ('bior2.4', 2)
])
def test_ok(wave, J):
    x = torch.randn(5, 4, 64, 64).to(dev)
    dwt = DWTForward(J=J, wave=wave).to(dev)
    iwt = DWTInverse(wave=wave).to(dev)
    yl, yh = dwt(x)
    x2 = iwt((yl, yh))
    # Can have data errors sometimes
    assert yl.is_contiguous()
    for j in range(J):
        assert yh[j].is_contiguous()
    assert x2.is_contiguous()


@pytest.mark.parametrize("wave, J, j", [
    ('db1', 1, 0), ('db1', 2, 1), ('db2', 2, 0), ('db3', 3, 2)
])
def test_commutativity(wave, J, j):
    # Test the commutativity of the dwt
    C = 3
    Y = torch.randn(4, C, 128, 128, requires_grad=True, device=dev)
    dwt = DWTForward(J=J, wave=wave).to(dev)
    iwt = DWTInverse(wave=wave).to(dev)

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

    np.testing.assert_array_almost_equal(
        (ya+yb).detach().cpu(), yab.detach().cpu(), decimal=4)
    np.testing.assert_array_almost_equal(
        (yc+yb).detach().cpu(), ybc.detach().cpu(), decimal=4)
