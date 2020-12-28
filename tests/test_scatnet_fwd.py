from Transform2d_np import Transform2d
from pytorch_wavelets.scatternet import ScatLayer, ScatLayerj2
import numpy as np
import torch
import torch.nn.functional as F
import pytest


@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b', 'near_sym_b_bp'])
def test_equal(biort):
    b = 1e-2

    scat = ScatLayer(biort=biort, magbias=b)
    xfm = Transform2d(biort=biort)
    x = torch.randn(3, 4, 32, 32)
    z = scat(x)

    X = x.data.numpy()
    Yl, Yh = xfm.forward(X, nlevels=1)
    yl = torch.tensor(Yl)
    yl2 = F.avg_pool2d(yl, 2)

    M = np.sqrt(Yh[0].real**2 + Yh[0].imag**2 + b**2) - b
    M = M.transpose(0, 2, 1, 3, 4)
    m = torch.tensor(M)
    m2 = m.reshape(3, 24, 16, 16)
    z2 = torch.cat((yl2, m2), dim=1)
    np.testing.assert_array_almost_equal(z, z2, decimal=4)


@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b', 'near_sym_b_bp'])
def test_equal_colour(biort):
    b = 1e-2

    scat = ScatLayer(biort=biort, combine_colour=True, magbias=b)
    xfm = Transform2d(biort=biort)
    x = torch.randn(4, 3, 32, 32)
    z = scat(x)

    X = x.data.numpy()
    Yl, Yh = xfm.forward(X, nlevels=1)
    yl = torch.tensor(Yl)
    yl2 = F.avg_pool2d(yl, 2)

    M = np.sqrt(Yh[0][:,0].real**2 + Yh[0][:,0].imag**2 +
                Yh[0][:,1].real**2 + Yh[0][:,1].imag**2 +
                Yh[0][:,2].real**2 + Yh[0][:,2].imag**2 + b**2) - b
    m = torch.tensor(M)
    z2 = torch.cat((yl2, m), dim=1)
    np.testing.assert_array_almost_equal(z, z2, decimal=4)


@pytest.mark.parametrize('sz', [32, 30, 31, 29, 28])
def test_odd_size(sz):
    scat = ScatLayer(biort='near_sym_a')
    x = torch.randn(5, 5, sz, sz)
    z = scat(x)
    assert z.shape[-1] == (sz + 1)//2


@pytest.mark.parametrize('biort,qshift', [('near_sym_a', 'qshift_a'),
                                          ('near_sym_b', 'qshift_b'),
                                          ('near_sym_b_bp', 'qshift_b_bp')])
def test_equal_j2(biort, qshift):
    b = 1e-2

    scat = ScatLayerj2(biort=biort, qshift=qshift, magbias=b)
    xfm = Transform2d(biort=biort, qshift=qshift)
    x = torch.randn(3, 4, 32, 32)
    z = scat(x)

    X = x.data.numpy()
    yl, yh = xfm.forward(X, nlevels=2)
    # Make it a tensor to average pool
    yl = torch.tensor(yl)
    S0 = F.avg_pool2d(yl, 2).numpy()

    # First order scatter coeffs
    M1 = np.sqrt(yh[0].real**2 + yh[0].imag**2 + b**2) - b
    M1 = M1.transpose(0, 2, 1, 3, 4)
    M2 = np.sqrt(yh[1].real**2 + yh[1].imag**2 + b**2) - b
    S1_2 = M2.transpose(0, 2, 1, 3, 4)

    M1 = M1.reshape(3, 24, 16, 16)
    yl, yh = xfm.forward(M1, nlevels=1)
    # Make yl a tensor to average pool
    yl = torch.tensor(yl)
    S1_1 = F.avg_pool2d(yl, 2).numpy()
    S1_1 = S1_1.reshape(3, 6, 4, 8, 8)

    M2_1 = np.sqrt(yh[0].real**2 + yh[0].imag**2 + b**2) - b
    S2_1 = M2_1.transpose(0, 2, 1, 3, 4)
    S2_1 = S2_1.reshape(3, 36, 4, 8, 8)

    z2 = np.concatenate((S0[:, None], S1_1, S1_2, S2_1), axis=1)
    z2 = z2.reshape(3, (1+6+6+36)*4, 8, 8)
    np.testing.assert_array_almost_equal(z.numpy(), z2, decimal=4)


@pytest.mark.parametrize('biort,qshift', [('near_sym_a', 'qshift_a'),
                                          ('near_sym_b', 'qshift_b'),
                                          ('near_sym_b_bp', 'qshift_b_bp')])
def test_equal_j2_colour(biort, qshift):
    b = 1e-2

    scat = ScatLayerj2(biort=biort, qshift=qshift, magbias=b,
                       combine_colour=True)
    xfm = Transform2d(biort=biort, qshift=qshift)
    x = torch.randn(4, 3, 32, 32)
    z = scat(x)

    X = x.data.numpy()
    yl, Yh = xfm.forward(X, nlevels=2)
    # Make it a tensor to average pool
    yl = torch.tensor(yl)
    S0 = F.avg_pool2d(yl, 2).numpy()

    # First order scatter coeffs
    M1 = np.sqrt(Yh[0][:,0].real**2 + Yh[0][:,0].imag**2 +
                 Yh[0][:,1].real**2 + Yh[0][:,1].imag**2 +
                 Yh[0][:,2].real**2 + Yh[0][:,2].imag**2 + b**2) - b
    M2 = np.sqrt(Yh[1][:,0].real**2 + Yh[1][:,0].imag**2 +
                 Yh[1][:,1].real**2 + Yh[1][:,1].imag**2 +
                 Yh[1][:,2].real**2 + Yh[1][:,2].imag**2 + b**2) - b
    yl, yh = xfm.forward(M1, nlevels=1)
    # Make yl a tensor to average pool
    yl = torch.tensor(yl)
    S1_1 = F.avg_pool2d(yl, 2).numpy()
    M2_1 = np.sqrt(yh[0].real**2 + yh[0].imag**2 + b**2) - b
    S2_1 = M2_1.transpose(0, 2, 1, 3, 4)
    S2_1 = S2_1.reshape(4, 36, 8, 8)

    z2 = np.concatenate((S0, S1_1, M2, S2_1), axis=1)
    np.testing.assert_array_almost_equal(z.numpy(), z2, decimal=4)


@pytest.mark.parametrize('sz', [32, 30, 31, 29, 28])
def test_odd_size_j2(sz):
    scat = ScatLayerj2(biort='near_sym_a', qshift='qshift_a')
    x = torch.randn(5, 5, sz, sz)
    z = scat(x)
    assert z.shape[-1] == 8
