import numpy as np
from pytorch_wavelets.dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.numpy.lowlevel import colfilter as np_colfilter
from pytorch_wavelets.dtcwt.lowlevel import rowfilter, prep_filt
import torch
import py3nvml
import pytest
import datasets

HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')


def setup():
    global barbara, barbara_t, tf
    global bshape, bshape_extracol
    global ref_rowfilter, ch
    py3nvml.grab_gpus(1, gpu_fraction=0.5, env_set_ok=True)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_extracol = bshape[:]
    bshape_extracol[2] += 1
    barbara_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32),
                                dim=0).to(dev)
    ch = barbara_t.shape[1]

    # Some useful functions
    ref_rowfilter = lambda x, h: np.stack(
        [np_colfilter(s.T, h).T for s in x], axis=0)


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_odd_size():
    h = [-1, 2, -1]
    y_op = rowfilter(barbara_t, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape


def test_even_size():
    h = [-1, -1]
    y_op = rowfilter(barbara_t, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape_extracol


def test_qshift():
    h = _qshift('qshift_a')[0]
    x = barbara_t
    y_op = rowfilter(x, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape_extracol


def test_biort():
    h = _biort('antonini')[0]
    y_op = rowfilter(barbara_t, prep_filt(h, 1).to(dev))
    assert list(y_op.shape)[1:] == bshape


def test_even_size_batch():
    zero_t = torch.zeros([1, *barbara.shape], dtype=torch.float32).to(dev)
    h = [-1, 1]
    y = rowfilter(zero_t, prep_filt(h, 1).to(dev))
    assert list(y.shape)[1:] == bshape_extracol
    assert not np.any(y.cpu().numpy()[:] != 0.0)


def test_equal_small_in():
    h = _qshift('qshift_b')[0]
    im = barbara[:,0:4,0:4]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_rowfilter(im, h)
    y = rowfilter(im_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_biort1():
    h = _biort('near_sym_b')[0]
    ref = ref_rowfilter(barbara, h)
    y = rowfilter(barbara_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_biort2():
    h = _biort('near_sym_b')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_rowfilter(im, h)
    y = rowfilter(im_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_qshift1():
    h = _qshift('qshift_c')[0]
    ref = ref_rowfilter(barbara, h)
    y = rowfilter(barbara_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


def test_equal_numpy_qshift2():
    h = _qshift('qshift_c')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_rowfilter(im, h)
    y = rowfilter(im_t, prep_filt(h, 1).to(dev))
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.skip
def test_gradients():
    h = _biort('near_sym_b')[0]
    im_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32,
                                        requires_grad=True), dim=0).to(dev)
    y_t = rowfilter(im_t, prep_filt(h, 1).to(dev))
    dy = np.random.randn(*tuple(y_t.shape)).astype('float32')
    torch.autograd.grad(y_t, im_t, grad_outputs=torch.tensor(dy))
