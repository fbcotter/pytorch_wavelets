from torch.autograd import gradcheck
from pytorch_wavelets.scatternet import ScatLayer, ScatLayerj2
from pytorch_wavelets.scatternet.lowlevel import SmoothMagFn
import torch
import pytest
import py3nvml


HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')


def setup():
    py3nvml.grab_gpus(1, gpu_fraction=0.5, env_set_ok=True)


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b', 'near_sym_b_bp'])
def test_grad_scat(biort):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double, device=dev)
    scat = ScatLayer(biort=biort).to(dev)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b', 'near_sym_b_bp'])
def test_grad_scat_colour(biort):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double, device=dev)
    scat = ScatLayer(biort=biort, combine_colour=True).to(dev)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize('biort,qshift', [('near_sym_a', 'qshift_a'),
                                          ('near_sym_b', 'qshift_b'),
                                          ('near_sym_b_bp', 'qshift_b_bp')])
def test_grad_scatj2(biort, qshift):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double, device=dev)
    scat = ScatLayerj2(biort=biort, qshift=qshift).to(dev)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize('biort,qshift', [('near_sym_a', 'qshift_a'),
                                          ('near_sym_b', 'qshift_b'),
                                          ('near_sym_b_bp', 'qshift_b_bp')])
def test_grad_scatj2_colour(biort, qshift):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double, device=dev)
    scat = ScatLayerj2(biort=biort, qshift=qshift, combine_colour=True).to(dev)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize('sz', [32, 30, 31, 29, 28])
def test_grad_odd_size(sz):
    x = torch.randn(1, 3, sz, sz, requires_grad=True, dtype=torch.double, device=dev)
    scat = ScatLayer(biort='near_sym_a').to(dev)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize('sz', [32, 30, 31, 29, 28])
def test_grad_odd_size_j2(sz):
    x = torch.randn(1, 3, sz, sz, requires_grad=True, dtype=torch.double, device=dev)
    scat = ScatLayerj2(biort='near_sym_a', qshift='qshift_a').to(dev)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.skip("These tests take a very long time to compute")
@pytest.mark.parametrize('magbias', [0, 1e-1, 1e-2, 1e-3])
def test_grad_mag(magbias):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double, device=dev)
    y = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double, device=dev)
    gradcheck(SmoothMagFn.apply, (x, y, magbias))
