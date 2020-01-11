from torch.autograd import gradcheck
from pytorch_wavelets.scatternet import ScatLayer, ScatLayerj2
from pytorch_wavelets.scatternet.lowlevel import SmoothMagFn
import torch
import pytest


@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b', 'near_sym_b_bp'])
def test_grad_scat(biort):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double)
    scat = ScatLayer(biort=biort)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b', 'near_sym_b_bp'])
def test_grad_scat_colour(biort):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double)
    scat = ScatLayer(biort=biort, combine_colour=True)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.parametrize('biort,qshift', [('near_sym_a', 'qshift_a'),
                                          ('near_sym_b', 'qshift_b'),
                                          ('near_sym_b_bp', 'qshift_b_bp')])
def test_grad_scatj2(biort, qshift):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double)
    scat = ScatLayerj2(biort=biort, qshift=qshift)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.parametrize('biort,qshift', [('near_sym_a', 'qshift_a'),
                                          ('near_sym_b', 'qshift_b'),
                                          ('near_sym_b_bp', 'qshift_b_bp')])
def test_grad_scatj2_colour(biort, qshift):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double)
    scat = ScatLayerj2(biort=biort, qshift=qshift, combine_colour=True)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.parametrize('sz', [32, 30, 31, 29, 28])
def test_grad_odd_size(sz):
    x = torch.randn(1, 3, sz, sz, requires_grad=True, dtype=torch.double)
    scat = ScatLayer(biort='near_sym_a')
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.parametrize('sz', [32, 30, 31, 29, 28])
def test_grad_odd_size_j2(sz):
    x = torch.randn(1, 3, sz, sz, requires_grad=True, dtype=torch.double)
    scat = ScatLayerj2(biort='near_sym_a', qshift='qshift_a')
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.parametrize('magbias', [0, 1e-1, 1e-2, 1e-3])
def test_grad_mag(magbias):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double)
    y = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double)
    gradcheck(SmoothMagFn.apply, (x, y, magbias))
