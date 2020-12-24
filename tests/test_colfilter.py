import numpy as np
from pytorch_wavelets.dtcwt.coeffs import biort, qshift
from dtcwt.numpy.lowlevel import colfilter as np_colfilter

import pytest
import datasets
from pytorch_wavelets.dtcwt.lowlevel import colfilter, prep_filt
import torch
import py3nvml

HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')


def setup():
    global barbara, barbara_t
    global bshape, bshape_extrarow
    global ref_colfilter, ch
    py3nvml.grab_gpus(1, gpu_fraction=0.5, env_set_ok=True)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_extrarow = bshape[:]
    bshape_extrarow[1] += 1
    barbara_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32),
                                dim=0).to(dev)
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
    y_op = colfilter(barbara_t, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape


def test_even_size():
    h = [-1, -1]
    y_op = colfilter(barbara_t, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape_extrarow


def test_qshift():
    h = qshift('qshift_a')[0]
    y_op = colfilter(barbara_t, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape_extrarow


def test_biort():
    h = biort('antonini')[0]
    y_op = colfilter(barbara_t, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape


def test_even_size_batch():
    zero_t = torch.zeros((1, *barbara.shape), dtype=torch.float32).to(dev)
    y = colfilter(zero_t, prep_filt([-1,1], 1).to(dev))
    assert list(y.shape)[1:] == bshape_extrarow
    assert not np.any(y.cpu().numpy()[:] != 0.0)


def test_equal_small_in():
    h = qshift('qshift_b')[0]
    im = barbara[:,0:4,0:4]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_biort1():
    h = biort('near_sym_b')[0]
    ref = ref_colfilter(barbara, h)
    y = colfilter(barbara_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_biort2():
    h = biort('near_sym_b')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_qshift1():
    h = qshift('qshift_c')[0]
    ref = ref_colfilter(barbara, h)
    y = colfilter(barbara_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_qshift2():
    h = qshift('qshift_c')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.skip
def test_gradients():
    h = biort('near_sym_b')[0]
    im_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32,
                                        requires_grad=True), dim=0)
    y_t = colfilter(im_t, prep_filt(h, 1))
    dy = np.random.randn(*tuple(y_t.shape)).astype('float32')
    torch.autograd.grad(y_t, im_t, grad_outputs=torch.tensor(dy))
