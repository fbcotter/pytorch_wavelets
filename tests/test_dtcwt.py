import pytest

import numpy as np
from Transform2d_np import Transform2d as Transform2d_np
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import datasets
import torch
import py3nvml
PRECISION_DECIMAL = 3

HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')


def setup():
    global barbara, barbara_t
    global bshape, bshape_half
    global ch
    py3nvml.grab_gpus(1, gpu_fraction=0.5)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_half = bshape[:]
    bshape_half[1] //= 2
    barbara_t = torch.unsqueeze(
        torch.tensor(barbara, dtype=torch.float32, device=dev), dim=0)
    ch = barbara_t.shape[1]


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_simple():
    xfm = DTCWTForward(J=3).to(dev)
    Yl, Yh = xfm(barbara_t)
    assert len(Yl.shape) == 4
    assert len(Yh) == 3
    assert Yh[0].shape[-1] == 2


def test_specific_wavelet():
    xfm = DTCWTForward(J=3, biort='antonini', qshift='qshift_06').to(dev)
    Yl, Yh = xfm(barbara_t)
    assert len(Yl.shape) == 4
    assert len(Yh) == 3
    assert Yh[0].shape[-1] == 2


def test_odd_rows():
    xfm = DTCWTForward(J=3).to(dev)
    Yl, Yh = xfm(barbara_t[:,:,:509])


def test_odd_cols():
    xfm = DTCWTForward(J=3).to(dev)
    Yl, Yh = xfm(barbara_t[:,:,:,:509])


def test_odd_rows_and_cols():
    xfm = DTCWTForward(J=3).to(dev)
    Yl, Yh = xfm(barbara_t[:,:,:509,:509])


@pytest.mark.parametrize("J, o_before_c", [
    (1,False),(1,True),(2, False), (2,True),
    (3, False),(3, True),(4, False), (4, True),
    (5, False), (5, True)
])
def test_fwd(J, o_before_c):
    X = 100*np.random.randn(3, 5, 100, 100)
    xfm = DTCWTForward(J=J, o_before_c=o_before_c).to(dev)
    Yl, Yh = xfm(torch.tensor(X, dtype=torch.float32, device=dev))
    f1 = Transform2d_np()
    yl, yh = f1.forward(X, nlevels=J)

    np.testing.assert_array_almost_equal(
        Yl.cpu(), yl, decimal=PRECISION_DECIMAL)
    for i in range(len(yh)):
        if o_before_c:
            np.testing.assert_array_almost_equal(
                Yh[i][...,0].cpu().transpose(2,1), yh[i].real,
                decimal=PRECISION_DECIMAL)
            np.testing.assert_array_almost_equal(
                Yh[i][...,1].cpu().transpose(2,1), yh[i].imag,
                decimal=PRECISION_DECIMAL)
        else:
            np.testing.assert_array_almost_equal(
                Yh[i][...,0].cpu(), yh[i].real, decimal=PRECISION_DECIMAL)
            np.testing.assert_array_almost_equal(
                Yh[i][...,1].cpu(), yh[i].imag, decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("J, o_before_c", [
    (1,False),(1,True),(2, False), (2,True),
    (3, False),(3, True),(4, False), (4, True),
    (5, False), (5, True)
])
def test_fwd_skip_hps(J, o_before_c):
    X = 100*np.random.randn(3, 5, 100, 100)
    # Randomly turn on/off the highpass outputs
    hps = np.random.binomial(size=J, n=1,p=0.5).astype('bool')
    xfm = DTCWTForward(J=J, skip_hps=hps, o_before_c=o_before_c).to(dev)
    Yl, Yh = xfm(torch.tensor(X, dtype=torch.float32, device=dev))
    f1 = Transform2d_np()
    yl, yh = f1.forward(X, nlevels=J)

    np.testing.assert_array_almost_equal(
        Yl.cpu(), yl, decimal=PRECISION_DECIMAL)
    for j in range(J):
        if hps[j]:
            assert Yh[j].shape == torch.Size([0])
        else:
            if o_before_c:
                np.testing.assert_array_almost_equal(
                    Yh[j][...,0].cpu().transpose(2,1), yh[j].real,
                    decimal=PRECISION_DECIMAL)
                np.testing.assert_array_almost_equal(
                    Yh[j][...,1].cpu().transpose(2,1), yh[j].imag,
                    decimal=PRECISION_DECIMAL)
            else:
                np.testing.assert_array_almost_equal(
                    Yh[j][...,0].cpu(), yh[j].real, decimal=PRECISION_DECIMAL)
                np.testing.assert_array_almost_equal(
                    Yh[j][...,1].cpu(), yh[j].imag, decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("J, o_before_c", [
    (1,False),(1,True),(2, False), (2,True),
    (3, False),(3, True),(4, False), (4, True),
    (5, False), (5, True)
])
def test_inv(J, o_before_c):
    Yl = 100*np.random.randn(3, 5, 64, 64)
    Yhr = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yhi = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yh1 = [yhr + 1j*yhi for yhr, yhi in zip(Yhr, Yhi)]
    if o_before_c:
        Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1),
                            dtype=torch.float32, device=dev).transpose(1,2)
               for yhr, yhi in zip(Yhr, Yhi)]
    else:
        Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1),
                            dtype=torch.float32, device=dev)
               for yhr, yhi in zip(Yhr, Yhi)]

    ifm = DTCWTInverse(J=J, o_before_c=o_before_c).to(dev)
    X = ifm((torch.tensor(Yl, dtype=torch.float32, device=dev), Yh2))
    f1 = Transform2d_np()
    x = f1.inverse(Yl, Yh1)

    np.testing.assert_array_almost_equal(
        X.cpu(), x, decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("J, o_before_c", [
    (1,False),(1,True),(2, False), (2,True),
    (3, False),(3, True),(4, False), (4, True),
    (5, False), (5, True)
])
def test_inv_skip_hps(J, o_before_c):
    hps = np.random.binomial(size=J, n=1,p=0.5).astype('bool')
    Yl = 100*np.random.randn(3, 5, 64, 64)
    Yhr = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yhi = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yh1 = [yhr + 1j*yhi for yhr, yhi in zip(Yhr, Yhi)]
    if o_before_c:
        Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1),
                            dtype=torch.float32, device=dev).transpose(1,2)
               for yhr, yhi in zip(Yhr, Yhi)]
    else:
        Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1),
                            dtype=torch.float32, device=dev)
               for yhr, yhi in zip(Yhr, Yhi)]
    for j in range(J):
        if hps[j]:
            Yh2[j] = torch.tensor([])
            Yh1[j] = np.zeros_like(Yh1[j])

    ifm = DTCWTInverse(J=J, o_before_c=o_before_c).to(dev)
    X = ifm((torch.tensor(Yl, dtype=torch.float32, requires_grad=True,
                          device=dev), Yh2))
    # Also test giving None instead of an empty tensor
    for j in range(J):
        if hps[j]:
            Yh2[j] = None
    X2 = ifm((torch.tensor(Yl, dtype=torch.float32, device=dev), Yh2))
    f1 = Transform2d_np()
    x = f1.inverse(Yl, Yh1)

    np.testing.assert_array_almost_equal(
        X.detach().cpu(), x, decimal=PRECISION_DECIMAL)
    np.testing.assert_array_almost_equal(
        X2.cpu(), x, decimal=PRECISION_DECIMAL)

    # Test gradients are ok
    X.backward(torch.ones_like(X))


# Test end to end with numpy inputs
@pytest.mark.parametrize("biort,qshift,size,J", [
    ('antonini','qshift_a', (128,128), 3),
    ('antonini','qshift_a', (126,126), 3),
    ('legall','qshift_a', (99,100), 4),
    ('near_sym_a','qshift_c', (104, 101), 2),
    ('near_sym_b','qshift_d', (126, 126), 3),
])
def test_end2end(biort, qshift, size, J):
    im = np.random.randn(5,6,*size).astype('float32')
    xfm = DTCWTForward(J=J).to(dev)
    Yl, Yh = xfm(torch.tensor(im, dtype=torch.float32, device=dev))
    ifm = DTCWTInverse(J=J).to(dev)
    y = ifm((Yl, Yh))

    # Compare with numpy results
    f_np = Transform2d_np(biort=biort,qshift=qshift)
    yl, yh = f_np.forward(im, nlevels=J)
    y2 = f_np.inverse(yl, yh)

    np.testing.assert_array_almost_equal(y.cpu(), y2, decimal=PRECISION_DECIMAL)
