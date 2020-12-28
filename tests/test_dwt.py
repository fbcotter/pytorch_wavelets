import numpy as np
import pytest
import pywt
from pytorch_wavelets import DWTForward, DWTInverse
import torch
from contextlib import contextmanager

PREC_FLT = 3
PREC_DBL = 7

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


@pytest.mark.parametrize("wave, J, mode", [
    ('db1', 1, 'zero'),
    ('db1', 3, 'zero'),
    ('db3', 1, 'symmetric'),
    ('db3', 2, 'reflect'),
    ('db2', 3, 'periodization'),
    ('db2', 3, 'periodic'),
    ('db4', 2, 'zero'),
    ('db3', 3, 'symmetric'),
    ('bior2.4', 2, 'periodization'),
    ('bior2.4', 2, 'periodization'),
])
def test_ok(wave, J, mode):
    x = torch.randn(5, 4, 64, 64).to(dev)
    dwt = DWTForward(J=J, wave=wave, mode=mode).to(dev)
    iwt = DWTInverse(wave=wave, mode=mode).to(dev)
    yl, yh = dwt(x)
    x2 = iwt((yl, yh))
    # Can have data errors sometimes
    assert yl.is_contiguous()
    for j in range(J):
        assert yh[j].is_contiguous()
    assert x2.is_contiguous()


@pytest.mark.parametrize("wave, J, mode", [
    ('db1', 1, 'zero'),
    ('db1', 3, 'zero'),
    ('db3', 1, 'symmetric'),
    ('db3', 2, 'reflect'),
    ('db2', 3, 'periodization'),
    ('db2', 3, 'periodic'),
    ('db4', 2, 'zero'),
    ('db3', 3, 'symmetric'),
    ('bior2.4', 2, 'periodization'),
    ('bior2.4', 2, 'periodization')])
def test_equal(wave, J, mode):
    x = torch.randn(5, 4, 64, 64).to(dev)
    dwt = DWTForward(J=J, wave=wave, mode=mode).to(dev)
    iwt = DWTInverse(wave=wave, mode=mode).to(dev)
    yl, yh = dwt(x)
    x2 = iwt((yl, yh))

    # Test the forward and inverse worked
    np.testing.assert_array_almost_equal(x.cpu(), x2.detach().cpu(), decimal=PREC_FLT)
    # Test it is the same as doing the PyWavelets wavedec with reflection
    # padding
    coeffs = pywt.wavedec2(x.cpu().numpy(), wave, level=J, axes=(-2,-1),
                           mode=mode)
    np.testing.assert_array_almost_equal(yl.cpu(), coeffs[0], decimal=PREC_FLT)
    for j in range(J):
        for b in range(3):
            np.testing.assert_array_almost_equal(
                coeffs[J-j][b], yh[j][:,:,b].cpu(), decimal=PREC_FLT)


@pytest.mark.parametrize("size", [
    (64, 64), (127, 127), (126, 127), (100, 99), (99, 100)])
def test_equal_oddshape(size):
    wave = 'db3'
    J = 3
    mode = 'symmetric'
    x = torch.randn(5, 4, *size).to(dev)
    dwt1 = DWTForward(J=J, wave=wave, mode=mode).to(dev)
    iwt1 = DWTInverse(wave=wave, mode=mode).to(dev)

    yl1, yh1 = dwt1(x)
    x1 = iwt1((yl1, yh1))

    # Test it is the same as doing the PyWavelets wavedec
    coeffs = pywt.wavedec2(x.cpu().numpy(), wave, level=J, axes=(-2,-1), mode=mode)
    X = pywt.waverec2(coeffs, wave, mode=mode)
    np.testing.assert_array_almost_equal(X, x1.detach().cpu(), decimal=PREC_FLT)
    np.testing.assert_array_almost_equal(yl1.cpu(), coeffs[0], decimal=PREC_FLT)
    for j in range(J):
        for b in range(3):
            np.testing.assert_array_almost_equal(
                coeffs[J-j][b], yh1[j][:,:,b].cpu(), decimal=PREC_FLT)


@pytest.mark.parametrize("size", [
    (64, 64), (127, 127), (126, 127), (100, 99), (99, 100)])
def test_equal_oddshape2(size):
    wave = 'db3'
    J = 3
    mode = 'periodization'
    x = torch.randn(5, 4, *size).to(dev)
    dwt1 = DWTForward(J=J, wave=wave, mode=mode).to(dev)
    iwt1 = DWTInverse(wave=wave, mode=mode).to(dev)

    yl1, yh1 = dwt1(x)
    x1 = iwt1((yl1, yh1))

    # Test it is the same as doing the PyWavelets wavedec
    coeffs = pywt.wavedec2(x.cpu().numpy(), wave, level=J, axes=(-2,-1), mode=mode)
    X = pywt.waverec2(coeffs, wave, mode=mode)
    np.testing.assert_array_almost_equal(X, x1.detach().cpu(), decimal=PREC_FLT)
    np.testing.assert_array_almost_equal(yl1.cpu(), coeffs[0], decimal=PREC_FLT)
    for j in range(J):
        for b in range(3):
            np.testing.assert_array_almost_equal(
                coeffs[J-j][b], yh1[j][:,:,b].cpu(), decimal=PREC_FLT)


@pytest.mark.parametrize("wave, J, mode", [
    ('db1', 1, 'zero'),
    ('db1', 3, 'zero'),
    ('db3', 1, 'symmetric'),
    ('db3', 2, 'reflect'),
    ('db2', 3, 'periodization'),
    ('db2', 3, 'periodic'),
    ('db4', 2, 'zero'),
    ('db3', 3, 'symmetric'),
    ('bior2.4', 2, 'periodization'),
    ('bior2.4', 2, 'periodization')])
def test_equal_double(wave, J, mode):
    with set_double_precision():
        x = torch.randn(5, 4, 64, 64).to(dev)
        assert x.dtype == torch.float64
        dwt = DWTForward(J=J, wave=wave, mode=mode).to(dev)
        iwt = DWTInverse(wave=wave, mode=mode).to(dev)

    yl, yh = dwt(x)
    x2 = iwt((yl, yh))

    # Test the forward and inverse worked
    np.testing.assert_array_almost_equal(x.cpu(), x2.detach().cpu(), decimal=PREC_DBL)
    coeffs = pywt.wavedec2(x.cpu().numpy(), wave, level=J, axes=(-2,-1), mode=mode)
    np.testing.assert_array_almost_equal(yl.cpu(), coeffs[0], decimal=7)
    for j in range(J):
        for b in range(3):
            np.testing.assert_array_almost_equal(
                coeffs[J-j][b], yh[j][:,:,b].cpu(), decimal=PREC_DBL)


@pytest.mark.parametrize("wave, J, j", [
    ('db1', 1, 0),
    ('db1', 2, 1),
    ('db2', 2, 0),
    ('db3', 3, 2)
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
        (ya+yb).detach().cpu(), yab.detach().cpu(), decimal=PREC_FLT)
    np.testing.assert_array_almost_equal(
        (yc+yb).detach().cpu(), ybc.detach().cpu(), decimal=PREC_FLT)


# Test gradients
@pytest.mark.parametrize("wave, J, mode", [
    ('db1', 1, 'zero'),
    ('db1', 3, 'zero'),
    ('db3', 1, 'symmetric'),
    ('db3', 2, 'reflect'),
    ('db2', 3, 'periodization'),
    ('db4', 2, 'zero'),
    ('bior2.4', 2, 'periodization'),
    ('db1', 1, 'zero'),
    ('db1', 3, 'zero'),
    ('db2', 3, 'periodization'),
    ('db4', 2, 'zero'),
    ('bior2.4', 2, 'periodization')
])
def test_gradients_fwd(wave, J, mode):
    """ Gradient of forward function should be inverse function with filters
    swapped """
    im = np.random.randn(5,6,128, 128).astype('float32')
    imt = torch.tensor(im, dtype=torch.float32, requires_grad=True, device=dev)

    wave = pywt.Wavelet(wave)
    fwd_filts = (wave.dec_lo, wave.dec_hi)
    inv_filts = (wave.dec_lo[::-1], wave.dec_hi[::-1])
    dwt = DWTForward(J=J, wave=fwd_filts, mode=mode).to(dev)
    iwt = DWTInverse(wave=inv_filts, mode=mode).to(dev)

    yl, yh = dwt(imt)

    # Test the lowpass
    ylg = torch.randn(*yl.shape, device=dev)
    yl.backward(ylg, retain_graph=True)
    zeros = [torch.zeros_like(yh[i]) for i in range(J)]
    ref = iwt((ylg, zeros))
    np.testing.assert_array_almost_equal(imt.grad.detach().cpu(), ref.cpu(),
                                         decimal=PREC_FLT)

    # Test the bandpass
    for j, y in enumerate(yh):
        imt.grad.zero_()
        g = torch.randn(*y.shape, device=dev)
        y.backward(g, retain_graph=True)
        hps = [zeros[i] for i in range(J)]
        hps[j] = g
        ref = iwt((torch.zeros_like(yl), hps))
        np.testing.assert_array_almost_equal(imt.grad.detach().cpu(), ref.cpu(),
                                             decimal=PREC_FLT)


# Test gradients
@pytest.mark.parametrize("wave, J, mode", [
    ('db1', 1, 'zero'),
    ('db1', 3, 'zero'),
    ('db3', 1, 'symmetric'),
    ('db3', 2, 'reflect'),
    ('db2', 3, 'periodization'),
    ('db4', 2, 'zero'),
    #  ('db3', 3, 'symmetric', False, False),
    ('bior2.4', 2, 'periodization'),
    ('db1', 1, 'zero'),
    ('db1', 3, 'zero'),
    #  ('db3', 1, 'symmetric', True, True),
    #  ('db3', 2, 'reflect', False, True),
    ('db2', 3, 'periodization'),
    ('db4', 2, 'zero'),
    #  ('db3', 3, 'symmetric', True, False),
    ('bior2.4', 2, 'periodization')
])
def test_gradients_inv(wave, J, mode):
    """ Gradient of inverse function should be forward function with filters
    swapped """
    wave = pywt.Wavelet(wave)
    fwd_filts = (wave.dec_lo, wave.dec_hi)
    inv_filts = (wave.dec_lo[::-1], wave.dec_hi[::-1])
    dwt = DWTForward(J=J, wave=fwd_filts, mode=mode).to(dev)
    iwt = DWTInverse(wave=inv_filts, mode=mode).to(dev)

    # Get the shape of the pyramid
    temp = torch.zeros(5,6,128,128).to(dev)
    l, h = dwt(temp)
    # Create our inputs
    yl = torch.randn(*l.shape, requires_grad=True, device=dev)
    yh = [torch.randn(*h[i].shape, requires_grad=True, device=dev)
          for i in range(J)]
    y = iwt((yl, yh))

    # Test the gradients
    yg = torch.randn(*y.shape, device=dev)
    y.backward(yg, retain_graph=True)
    dyl, dyh = dwt(yg)

    # test the lowpass
    np.testing.assert_array_almost_equal(yl.grad.detach().cpu(), dyl.cpu(),
                                         decimal=PREC_FLT)

    # Test the bandpass
    for j in range(J):
        np.testing.assert_array_almost_equal(yh[j].grad.detach().cpu(),
                                             dyh[j].cpu(),
                                             decimal=PREC_FLT)
