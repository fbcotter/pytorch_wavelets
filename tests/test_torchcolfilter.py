import numpy as np
from dtcwt_pytorch.coeffs import biort, qshift
from dtcwt.numpy.lowlevel import colfilter as np_colfilter

import pytest
import datasets
from dtcwt_pytorch.backend.lowlevel import colfilter, prep_filt
import torch


def setup():
    global barbara, barbara_t
    global bshape, bshape_extrarow
    global ref_colfilter, ch
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_extrarow = bshape[:]
    bshape_extrarow[1] += 1
    barbara_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32),
                                dim=0)
    ch = barbara_t.shape[1]

    # Some useful functions
    ref_colfilter = lambda x, h: np.stack(
        [np_colfilter(s, h) for s in x], axis=0)


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_odd_size():
    h = [-1, 2, -1]
    y_op = colfilter(barbara_t, prep_filt(h, ch))
    assert list(y_op.shape)[1:] == bshape


def test_even_size():
    h = [-1, -1]
    y_op = colfilter(barbara_t, prep_filt(h, ch))
    assert list(y_op.shape)[1:] == bshape_extrarow


def test_qshift():
    h = qshift('qshift_a')[0]
    y_op = colfilter(barbara_t, prep_filt(h, ch))
    assert list(y_op.shape)[1:] == bshape_extrarow


def test_biort():
    h = biort('antonini')[0]
    y_op = colfilter(barbara_t, prep_filt(h, ch))
    assert list(y_op.shape)[1:] == bshape


def test_even_size_batch():
    zero_t = torch.zeros((1, *barbara.shape), dtype=torch.float32)
    y = colfilter(zero_t, prep_filt([-1,1], ch))
    assert list(y.shape)[1:] == bshape_extrarow
    assert not np.any(y[:] != 0.0)


def test_equal_small_in():
    h = qshift('qshift_b')[0]
    im = barbara[:,0:4,0:4]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, 3))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_biort1():
    h = biort('near_sym_b')[0]
    ref = ref_colfilter(barbara, h)
    y = colfilter(barbara_t, prep_filt(h, ch))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_biort2():
    h = biort('near_sym_b')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, ch))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift1():
    h = qshift('qshift_c')[0]
    ref = ref_colfilter(barbara, h)
    y = colfilter(barbara_t, prep_filt(h, ch))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift2():
    h = qshift('qshift_c')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, ch))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


@pytest.mark.skip
def test_gradients():
    h = biort('near_sym_b')[0]
    im_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32,
                                        requires_grad=True), dim=0)
    y_t = colfilter(im_t, prep_filt(h, 3))
    dy = np.random.randn(*tuple(y_t.shape)).astype('float32')
    dx = torch.autograd.grad(y_t, im_t, grad_outputs=torch.tensor(dy))

    np.testing.assert_array_almost_equal(dx2, dx[0].numpy(), decimal=4)
# vim:sw=4:sts=4:et
