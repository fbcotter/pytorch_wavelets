from pytest import raises
import pytest

import numpy as np
from pytorch_wavelets.dtcwt.coeffs import qshift
from dtcwt.numpy.lowlevel import coldfilt as np_coldfilt
import datasets
from pytorch_wavelets.dtcwt.lowlevel import coldfilt, prep_filt
import torch
import py3nvml

HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')


def setup():
    global barbara, barbara_t
    global bshape, bshape_half
    global ref_coldfilt, ch
    py3nvml.grab_gpus(1, gpu_fraction=0.5, env_set_ok=True)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_half = bshape[:]
    bshape_half[1] //= 2
    barbara_t = torch.unsqueeze(
        torch.tensor(barbara, dtype=torch.float32, device=dev), dim=0)
    ch = barbara_t.shape[1]

    # Some useful functions
    ref_coldfilt = lambda x, ha, hb: np.stack(
        [np_coldfilt(s, ha, hb) for s in x], axis=0)


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


@pytest.mark.skip("Don't currently check for this in lowlevel code for speed")
def test_odd_filter():
    with raises(ValueError):
        ha = prep_filt((-1,2,-1), 1).to(dev)
        hb = prep_filt((-1,2,1), 1).to(dev)
        coldfilt(barbara_t, ha, hb)


@pytest.mark.skip("Don't currently check for this in lowlevel code for speed")
def test_different_size():
    with raises(ValueError):
        ha = prep_filt((-0.5,-1,2,0.5), 1).to(dev)
        hb = prep_filt((-1,2,1), 1).to(dev)
        coldfilt(barbara_t, ha, hb)


def test_bad_input_size():
    with raises(ValueError):
        ha = prep_filt((-1, 1), 1).to(dev)
        hb = prep_filt((1, -1), 1).to(dev)
        coldfilt(barbara_t[:,:,:511,:], ha, hb)


def test_good_input_size():
    ha = prep_filt((-1, 1), 1).to(dev)
    hb = prep_filt((1, -1), 1).to(dev)
    coldfilt(barbara_t[:,:,:,:511], ha, hb)


def test_good_input_size_non_orthogonal():
    ha = prep_filt((1, 1), 1).to(dev)
    hb = prep_filt((1, -1), 1).to(dev)
    coldfilt(barbara_t[:,:,:,:511], ha, hb)


def test_output_size():
    ha = prep_filt((-1, 1), 1).to(dev)
    hb = prep_filt((1, -1), 1).to(dev)
    y_op = coldfilt(barbara_t, ha, hb)
    assert list(y_op.shape)[1:] == bshape_half


@pytest.mark.parametrize('hp', [False, True])
def test_equal_small_in(hp):
    if hp:
        ha = qshift('qshift_a')[4]
        hb = qshift('qshift_a')[5]
    else:
        ha = qshift('qshift_a')[0]
        hb = qshift('qshift_a')[1]
    im = barbara[:,0:4,0:4]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_coldfilt(im, ha, hb)
    y = coldfilt(im_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                 highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift1(hp):
    if hp:
        ha = qshift('qshift_a')[4]
        hb = qshift('qshift_a')[5]
    else:
        ha = qshift('qshift_a')[0]
        hb = qshift('qshift_a')[1]
    ref = ref_coldfilt(barbara, ha, hb)
    y = coldfilt(barbara_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                 highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift2(hp):
    if hp:
        ha = qshift('qshift_a')[4]
        hb = qshift('qshift_a')[5]
    else:
        ha = qshift('qshift_a')[0]
        hb = qshift('qshift_a')[1]
    im = barbara[:, :508, :502]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_coldfilt(im, ha, hb)
    y = coldfilt(im_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                 highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.parametrize('hp', [False, True])
def test_equal_numpy_qshift3(hp):
    if hp:
        ha = qshift('qshift_a')[4]
        hb = qshift('qshift_a')[5]
    else:
        ha = qshift('qshift_a')[0]
        hb = qshift('qshift_a')[1]
    im = barbara[:, :508, :502]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0).to(dev)
    ref = ref_coldfilt(im, ha, hb)
    y = coldfilt(im_t, prep_filt(ha, 1).to(dev), prep_filt(hb, 1).to(dev),
                 highpass=hp)
    np.testing.assert_array_almost_equal(y[0].cpu(), ref, decimal=4)


@pytest.mark.skip
def test_gradients():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    im_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32,
                                        requires_grad=True), dim=0)
    y_t = coldfilt(im_t, prep_filt(ha, 1), prep_filt(hb, 1), np.sum(ha*hb) > 0)
    dy = np.random.randn(*tuple(y_t.shape)).astype('float32')
    torch.autograd.grad(y_t, im_t, grad_outputs=torch.tensor(dy))
