import torch
import pytest
from torch.autograd import gradcheck
from pytorch_wavelets.dwt.lowlevel import AFB2D, SFB2D
from pytorch_wavelets import DWTForward, DWTInverse
import py3nvml
from contextlib import contextmanager
ATOL = 1e-4
EPS = 1e-4

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
    py3nvml.grab_gpus(1, gpu_fraction=0.5, env_set_ok=True)


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize("mode", [0, 1, 6])
def test_fwd(mode):
    with set_double_precision():
        x = torch.randn(1,3,16,16, device=dev, requires_grad=True)
        xfm = DWTForward(J=2).to(dev)

    input = (x, xfm.h0_row, xfm.h1_row, xfm.h0_col, xfm.h1_col, mode)
    gradcheck(AFB2D.apply, input, eps=EPS, atol=ATOL)


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize("mode", [0, 1, 6])
def test_inv_j2(mode):
    with set_double_precision():
        low = torch.randn(1,3,16,16, device=dev, requires_grad=True)
        high = torch.randn(1,3,3,16,16, device=dev, requires_grad=True)
        ifm = DWTInverse().to(dev)
    input = (low, high, ifm.g0_row, ifm.g1_row, ifm.g0_col, ifm.g1_col, mode)
    gradcheck(SFB2D.apply, input, eps=EPS, atol=ATOL)
