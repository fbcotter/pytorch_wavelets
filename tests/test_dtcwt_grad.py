import torch
import pytest
from torch.autograd import gradcheck
from pytorch_wavelets.dtcwt.transform2d import DTCWTForward, DTCWTInverse
import pytorch_wavelets.dtcwt.transform_funcs as tf
from pytorch_wavelets.dwt.lowlevel import mode_to_int
import py3nvml
from contextlib import contextmanager
ATOL = 1e-4

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


def setup():
    global mode, o_dim, ri_dim
    mode = mode_to_int('symmetric')
    o_dim = 2
    ri_dim = -1
    py3nvml.grab_gpus(1, gpu_fraction=0.5, env_set_ok=True)


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize("skip_hps", [False, True])
def test_fwd_j1(skip_hps):
    with set_double_precision():
        x = torch.randn(1,3,16,16, device=dev, requires_grad=True)
        xfm = DTCWTForward(J=2).to(dev)

    input = (x, xfm.h0o, xfm.h1o, skip_hps, o_dim, ri_dim, mode)
    gradcheck(tf.FWD_J1.apply, input, eps=1e-3, atol=ATOL)


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize("skip_hps", [False, True])
def test_fwd_j2(skip_hps):
    with set_double_precision():
        x = torch.randn(1,3,16,16, device=dev, requires_grad=True)
        xfm = DTCWTForward(J=2).to(dev)
    input = (x, xfm.h0a, xfm.h1a, xfm.h0b, xfm.h1b, skip_hps, o_dim, ri_dim, mode)
    gradcheck(tf.FWD_J2PLUS.apply, input, eps=1e-3, atol=ATOL)


@pytest.mark.skip("These tests take a very long time to compute")
def test_inv_j1():
    with set_double_precision():
        low = torch.randn(1,3,16,16, device=dev, requires_grad=True)
        high = torch.randn(1,3,6,8,8,2, device=dev, requires_grad=True)
        ifm = DTCWTInverse().to(dev)
    input = (low, high, ifm.g0o, ifm.g1o, o_dim, ri_dim, mode)
    gradcheck(tf.INV_J1.apply, input, eps=1e-3, atol=ATOL)


@pytest.mark.skip("These tests take a very long time to compute")
def test_inv_j2():
    with set_double_precision():
        low = torch.randn(1,3,16,16, device=dev, requires_grad=True)
        high = torch.randn(1,3,6,8,8,2, device=dev, requires_grad=True)
        ifm = DTCWTInverse().to(dev)
    input = (low, high, ifm.g0a, ifm.g1a, ifm.g0b, ifm.g1b, o_dim, ri_dim, mode)
    gradcheck(tf.INV_J2PLUS.apply, input, eps=1e-3, atol=ATOL)
