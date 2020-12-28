import numpy as np
import pytest
import pywt
from pytorch_wavelets import DWT1DForward, DWT1DInverse
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
    x = torch.randn(5, 4, 64).to(dev)
    dwt = DWT1DForward(J=J, wave=wave, mode=mode).to(dev)
    iwt = DWT1DInverse(wave=wave, mode=mode).to(dev)
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
    x = torch.randn(5, 4, 64).to(dev)
    dwt = DWT1DForward(J=J, wave=wave, mode=mode).to(dev)
    yl, yh = dwt(x)

    # Test it is the same as doing the PyWavelets wavedec with reflection padding
    coeffs = pywt.wavedec(x.cpu().numpy(), wave, level=J, mode=mode)
    np.testing.assert_array_almost_equal(yl.cpu(), coeffs[0], decimal=PREC_FLT)
    for j in range(J):
        np.testing.assert_array_almost_equal(coeffs[J-j], yh[j].cpu(), decimal=PREC_FLT)

    # Test the forward and inverse worked
    iwt = DWT1DInverse(wave=wave, mode=mode).to(dev)
    x2 = iwt((yl, yh))
    np.testing.assert_array_almost_equal(x.cpu(), x2.detach().cpu(), decimal=PREC_FLT)


@pytest.mark.parametrize("length, mode", [
    (64, 'symmetric'),
    (64, 'periodization'),
    (127, 'symmetric'),
    (127, 'periodization'),
    (99, 'symmetric'),
    (99, 'periodization'),
])
def test_equal_oddshape(length, mode):
    wave = 'db3'
    J = 3
    x = torch.randn(5, 4, length).to(dev)
    dwt1 = DWT1DForward(J=J, wave=wave, mode=mode).to(dev)
    iwt1 = DWT1DInverse(wave=wave, mode=mode).to(dev)

    yl1, yh1 = dwt1(x)
    x1 = iwt1((yl1, yh1))

    # Test it is the same as doing the PyWavelets wavedec
    coeffs = pywt.wavedec(x.cpu().numpy(), wave, level=J, mode=mode)
    X = pywt.waverec(coeffs, wave, mode=mode)
    np.testing.assert_array_almost_equal(X, x1.detach().cpu(), decimal=PREC_FLT)
    np.testing.assert_array_almost_equal(yl1.cpu(), coeffs[0], decimal=PREC_FLT)
    for j in range(J):
        np.testing.assert_array_almost_equal(coeffs[J-j], yh1[j].cpu(), decimal=PREC_FLT)


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
        x = torch.randn(5, 4, 100).to(dev)
        assert x.dtype == torch.float64
        dwt = DWT1DForward(J=J, wave=wave, mode=mode).to(dev)
        iwt = DWT1DInverse(wave=wave, mode=mode).to(dev)

    yl, yh = dwt(x)
    x2 = iwt((yl, yh))

    # Test the forward and inverse worked
    np.testing.assert_array_almost_equal(x.cpu(), x2.detach().cpu(), decimal=PREC_DBL)
    coeffs = pywt.wavedec(x.cpu().numpy(), wave, level=J, mode=mode)
    np.testing.assert_array_almost_equal(yl.cpu(), coeffs[0], decimal=7)
    for j in range(J):
        np.testing.assert_array_almost_equal(coeffs[J-j], yh[j].cpu(), decimal=PREC_DBL)


# Test gradients
@pytest.mark.parametrize("wave, J, mode", [
    ('db1', 1, 'zero'),
    ('db1', 3, 'zero'),
    ('db3', 1, 'symmetric'),
    ('db2', 2, 'symmetric'),
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
    im = np.random.randn(5, 6, 128).astype('float32')
    imt = torch.tensor(im, dtype=torch.float32, requires_grad=True, device=dev)

    wave = pywt.Wavelet(wave)
    fwd_filts = (wave.dec_lo, wave.dec_hi)
    inv_filts = (wave.dec_lo[::-1], wave.dec_hi[::-1])
    dwt = DWT1DForward(J=J, wave=fwd_filts, mode=mode).to(dev)
    iwt = DWT1DInverse(wave=inv_filts, mode=mode).to(dev)

    yl, yh = dwt(imt)

    # Test the lowpass
    ylg = torch.randn(*yl.shape, device=dev)
    yl.backward(ylg, retain_graph=True)
    zeros = [torch.zeros_like(yh[i]) for i in range(J)]
    ref = iwt((ylg, zeros))
    if (imt.grad.detach().cpu() - ref.cpu()).abs().sum() > 1e-3:
        import pdb; pdb.set_trace()
    np.testing.assert_array_almost_equal(imt.grad.detach().cpu(), ref.cpu(), decimal=PREC_FLT)

    # Test the bandpass
    for j, y in enumerate(yh):
        imt.grad.zero_()
        g = torch.randn(*y.shape, device=dev)
        y.backward(g, retain_graph=True)
        hps = [zeros[i] for i in range(J)]
        hps[j] = g
        ref = iwt((torch.zeros_like(yl), hps))
        np.testing.assert_array_almost_equal(imt.grad.detach().cpu(), ref.cpu(), decimal=PREC_FLT)


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
def test_gradients_inv(wave, J, mode):
    """ Gradient of inverse function should be forward function with filters
    swapped """
    wave = pywt.Wavelet(wave)
    fwd_filts = (wave.dec_lo, wave.dec_hi)
    inv_filts = (wave.dec_lo[::-1], wave.dec_hi[::-1])
    dwt = DWT1DForward(J=J, wave=fwd_filts, mode=mode).to(dev)
    iwt = DWT1DInverse(wave=inv_filts, mode=mode).to(dev)

    # Get the shape of the pyramid
    temp = torch.zeros(5,6,128).to(dev)
    l, h = dwt(temp)
    # Create our inputs
    yl = torch.randn(*l.shape, requires_grad=True, device=dev)
    yh = [torch.randn(*h[i].shape, requires_grad=True, device=dev) for i in range(J)]
    y = iwt((yl, yh))

    # Test the gradients
    yg = torch.randn(*y.shape, device=dev)
    y.backward(yg, retain_graph=True)
    dyl, dyh = dwt(yg)

    # test the lowpass
    np.testing.assert_array_almost_equal(yl.grad.detach().cpu(), dyl.cpu(), decimal=PREC_FLT)

    # Test the bandpass
    for j in range(J):
        np.testing.assert_array_almost_equal(yh[j].grad.detach().cpu(), dyh[j].cpu(), decimal=PREC_FLT)

