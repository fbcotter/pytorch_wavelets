import pytest

import numpy as np
from Transform2d_np import Transform2d as Transform2d_np
from dtcwt_pytorch import DTCWTForward, DTCWTInverse
import datasets
import torch
import py3nvml
PRECISION_DECIMAL = 3


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
    barbara_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32),
                                dim=0)
    ch = barbara_t.shape[1]


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_simple():
    xfm = DTCWTForward(C=3, J=3)
    Yl, Yh = xfm(barbara_t)
    assert len(Yl.shape) == 4
    assert len(Yh) == 3
    assert Yh[0].shape[-1] == 2


def test_specific_wavelet():
    xfm = DTCWTForward(C=3, J=3, biort='antonini', qshift='qshift_06')
    Yl, Yh = xfm(barbara_t)
    assert len(Yl.shape) == 4
    assert len(Yh) == 3
    assert Yh[0].shape[-1] == 2


def test_odd_rows():
    xfm = DTCWTForward(C=3, J=3)
    Yl, Yh = xfm(barbara_t[:,:,:509])


def test_odd_cols():
    xfm = DTCWTForward(C=3, J=3)
    Yl, Yh = xfm(barbara_t[:,:,:,:509])


def test_odd_rows_and_cols():
    xfm = DTCWTForward(C=3, J=3)
    Yl, Yh = xfm(barbara_t[:,:,:509,:509])


#  def test_rot_symm_modified():
    #  # This test only checks there is no error running these functions,
    #  # not that they work
    #  xfm = Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
    #  Yl, Yh, Yscale = xfm.forward(barbara_t[:,:,:509,:509], nlevels=4,
                                 #  include_scale=True)


#  def test_0_levels():
    #  xfm = Transform2d()
    #  Yl, Yh = xfm.forward(barbara_t, nlevels=0)
    #  with tf.Session(config=config) as sess:
        #  sess.run(tf.global_variables_initializer())
        #  out = sess.run(Yl)[0]
    #  np.testing.assert_array_almost_equal(out, barbara, PRECISION_DECIMAL)
    #  assert len(Yh) == 0


#  def test_0_levels_w_scale():
    #  xfm = Transform2d()
    #  Yl, Yh, Yscale = xfm.forward(barbara_t, nlevels=0, include_scale=True)
    #  with tf.Session(config=config) as sess:
        #  sess.run(tf.global_variables_initializer())
        #  out = sess.run(Yl)[0]
    #  np.testing.assert_array_almost_equal(out, barbara, PRECISION_DECIMAL)
    #  assert len(Yh) == 0
    #  assert len(Yscale) == 0


@pytest.mark.parametrize("J", [1,2,3,4,5])
def test_fwd(J):
    X = 100*np.random.randn(3, 5, 100, 100)
    xfm = DTCWTForward(C=5, J=J)
    Yl, Yh = xfm(torch.tensor(X, dtype=torch.float32))
    f1 = Transform2d_np()
    yl, yh = f1.forward(X, nlevels=J)

    np.testing.assert_array_almost_equal(
        Yl, yl, decimal=PRECISION_DECIMAL)
    for i in range(len(yh)):
        np.testing.assert_array_almost_equal(
            Yh[i][...,0], yh[i].real, decimal=PRECISION_DECIMAL)
        np.testing.assert_array_almost_equal(
            Yh[i][...,1], yh[i].imag, decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("J", [1,2,3,4,5])
def test_fwd_nol1(J):
    X = 100*np.random.randn(3, 5, 100, 100)
    xfm = DTCWTForward(C=5, J=J, skip_hps=True)
    Yl, Yh = xfm(torch.tensor(X, dtype=torch.float32))
    f1 = Transform2d_np()
    yl, yh = f1.forward(X, nlevels=J)

    assert Yh[0] is None

    np.testing.assert_array_almost_equal(
        Yl, yl, decimal=PRECISION_DECIMAL)
    for i in range(1,len(yh)):
        np.testing.assert_array_almost_equal(
            Yh[i][...,0], yh[i].real, decimal=PRECISION_DECIMAL)
        np.testing.assert_array_almost_equal(
            Yh[i][...,1], yh[i].imag, decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("J", [1,2,3,4,5])
def test_inv(J):
    Yl = 100*np.random.randn(3, 5, 64, 64)
    Yhr = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yhi = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yh1 = [yhr + 1j*yhi for yhr, yhi in zip(Yhr, Yhi)]
    Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1), dtype=torch.float32)
           for yhr, yhi in zip(Yhr, Yhi)]

    ifm = DTCWTInverse(C=5, J=J)
    X = ifm(torch.tensor(Yl, dtype=torch.float32), Yh2)
    f1 = Transform2d_np()
    x = f1.inverse(Yl, Yh1)

    np.testing.assert_array_almost_equal(
        X, x, decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("J", [1,2,3,4,5])
def test_inv_nol1(J):
    Yl = 100*np.random.randn(3, 5, 64, 64)
    Yhr = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yhi = [np.random.randn(3, 5, 6, 2**j, 2**j) for j in range(4+J,4,-1)]
    Yh1 = [yhr + 1j*yhi for yhr, yhi in zip(Yhr, Yhi)]
    # Set the numpy l1 coeffs to 0
    Yh1[0] = np.zeros_like(Yh1[0])
    Yh2 = [torch.tensor(np.stack((yhr, yhi), axis=-1), dtype=torch.float32)
           for yhr, yhi in zip(Yhr, Yhi)]
    # Set the torch l1 coeffs to None
    Yh2[0] = None

    ifm = DTCWTInverse(C=5, J=J, skip_hps=True)
    X = ifm(torch.tensor(Yl, dtype=torch.float32), Yh2)
    f1 = Transform2d_np()
    x = f1.inverse(Yl, Yh1)

    np.testing.assert_array_almost_equal(
        X, x, decimal=PRECISION_DECIMAL)


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
    xfm = DTCWTForward(C=6, J=J)
    Yl, Yh = xfm(torch.tensor(im, dtype=torch.float32))
    ifm = DTCWTInverse(C=6, J=J)
    y = ifm(Yl, Yh)

    # Compare with numpy results
    f_np = Transform2d_np(biort=biort,qshift=qshift)
    yl, yh = f_np.forward(im, nlevels=J)
    y2 = f_np.inverse(yl, yh)

    np.testing.assert_array_almost_equal(y, y2, decimal=PRECISION_DECIMAL)
