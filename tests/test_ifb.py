import pytest

import numpy as np
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch
import py3nvml
from scipy.io import loadmat


HAVE_GPU = torch.cuda.is_available()


def setup():
    global sf, lo1, hi1, y1, lo2, hi2, y2
    with np.load('sfb2d_a.npz') as f:
        lo1 = torch.tensor(f['lo1']).reshape(1,1,16,32)
        hi1 = torch.tensor(f['hi1']).reshape(1,1,16,32)
        y1 = torch.tensor(f['y1']).reshape(1,1,32,32)
        lo2 = torch.tensor(f['lo2']).reshape(1,1,32,16)
        hi2 = torch.tensor(f['hi2']).reshape(1,1,32,16)
        y2 = torch.tensor(f['y2']).reshape(1,1,32,32)
        # Don't need to reverse the order as we use conv_transpose
        sf = torch.tensor(f['sf']).reshape(2,1,10,1)


def test_sfb2d_a():
    z1 = lowlevel.sfb2d_a(lo1, hi1, sf, d=2)
    z2 = lowlevel.sfb2d_a(lo2, hi2, sf.transpose(3,2), d=3)
    np.testing.assert_array_almost_equal(z1, y1)
    np.testing.assert_array_almost_equal(z2, y2)


@pytest.mark.skipif(not HAVE_GPU, reason='Need a gpu to test cuda')
def test_afb2d_a_gpu():
    z1 = lowlevel.sfb2d_a(lo1.cuda(), hi1.cuda(), sf.cuda(), d=2)
    z2 = lowlevel.sfb2d_a(lo2.cuda(), hi2.cuda(), sf.transpose(3,2).cuda(), d=3)
    np.testing.assert_array_almost_equal(z1.cpu(), y1)
    np.testing.assert_array_almost_equal(z2.cpu(), y2)


def test_sfb2d_a_channels():
    lo = torch.cat((lo1, -1*lo1), dim=1)
    hi = torch.cat((hi1, -1*hi1), dim=1)
    z1 = lowlevel.sfb2d_a(lo, hi, sf, d=2)
    lo = torch.cat((lo2, -1*lo2), dim=1)
    hi = torch.cat((hi2, -1*hi2), dim=1)
    z2 = lowlevel.sfb2d_a(lo, hi, sf.transpose(3,2), d=3)

    np.testing.assert_array_almost_equal(z1[0,0], y1[0,0])
    np.testing.assert_array_almost_equal(z1[0,1], -y1[0,0])
    np.testing.assert_array_almost_equal(z2[0,0], y2[0,0])
    np.testing.assert_array_almost_equal(z2[0,1], -y2[0,0])

def test_afb2d():
    y = lowlevel.afb2d(x, af, af)
    np.testing.assert_array_almost_equal(y[0,0], ll)
    np.testing.assert_array_almost_equal(y[0,1], lh)
    np.testing.assert_array_almost_equal(y[0,2], hl)
    np.testing.assert_array_almost_equal(y[0,3], hh)


def test_afb2d_channels():
    z = torch.cat((x, -1*x), dim=1)
    y = lowlevel.afb2d(z, af, af)
    np.testing.assert_array_almost_equal(y[0,0], ll)
    np.testing.assert_array_almost_equal(y[0,1], lh)
    np.testing.assert_array_almost_equal(y[0,2], hl)
    np.testing.assert_array_almost_equal(y[0,3], hh)
    np.testing.assert_array_almost_equal(y[0,4], -ll)
    np.testing.assert_array_almost_equal(y[0,5], -lh)
    np.testing.assert_array_almost_equal(y[0,6], -hl)
    np.testing.assert_array_almost_equal(y[0,7], -hh)


def test_cpxdual2d():
    z = torch.tensor(S['x'], dtype=torch.float).reshape(1,1,256,256)
    J = 3
    lows, w = lowlevel.cplxdual2D(z, J, qshift='qshift_06')
    for m in range(2):
        for n in range(2):
            for j in range(J):
                for l in range(3):
                    np.testing.assert_array_almost_equal(
                        w[j][m][n][l][0,0], S['w'][0,j][0,m][0,n][0,l])
            np.testing.assert_array_almost_equal(
                lows[m][n][0,0], S['w'][0,J][0,m][0,n])


def test_cpxdual2d_gpu():
    z = torch.tensor(S['x'], dtype=torch.float).reshape(1,1,256,256).cuda()
    J = 3
    lows, w = lowlevel.cplxdual2D(z, J, qshift='qshift_06')
    for m in range(2):
        for n in range(2):
            for j in range(J):
                for l in range(3):
                    np.testing.assert_array_almost_equal(
                        w[j][m][n][l][0,0].cpu(), S['w'][0,j][0,m][0,n][0,l])
            np.testing.assert_array_almost_equal(
                lows[m][n][0,0].cpu(), S['w'][0,J][0,m][0,n])

