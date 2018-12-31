import pytest

import numpy as np
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch
import py3nvml
from scipy.io import loadmat


HAVE_GPU = torch.cuda.is_available()


def setup():
    global af, x1, x2, lo1, hi1, lo2, hi2
    with np.load('afb2d_a.npz') as f:
        x1 = torch.tensor(f['x1']).reshape(1,1,32,16)
        x2 = torch.tensor(f['x2']).reshape(1,1,16,32)
        lo1 = torch.tensor(f['lo1'])
        hi1 = torch.tensor(f['hi1'])
        lo2 = torch.tensor(f['lo2'])
        hi2 = torch.tensor(f['hi2'])
        # Reverse the order as pytorch does correlation not convolution
        af = np.copy(f['af'][:,::-1])
        af = torch.tensor(af).reshape(2,1,10,1)

    global x, ll, lh, hl, hh
    with np.load('afb2d.npz') as f:
        x = torch.tensor(f['x']).reshape(1,1,32,32)
        ll = torch.tensor(f['ll'])
        lh = torch.tensor(f['lh'])
        hl = torch.tensor(f['hl'])
        hh = torch.tensor(f['hh'])
    py3nvml.grab_gpus(1)
    global S
    S = loadmat('cplx.mat')


def test_afb1d_periodic():
    y1 = lowlevel.afb1d_periodic(x1, af, d=2)
    y2 = lowlevel.afb1d_periodic(x2, af.transpose(3,2), d=3)
    np.testing.assert_array_almost_equal(y1[0,0], lo1)
    np.testing.assert_array_almost_equal(y1[0,1], hi1)
    np.testing.assert_array_almost_equal(y2[0,0], lo2)
    np.testing.assert_array_almost_equal(y2[0,1], hi2)


@pytest.mark.skipif(not HAVE_GPU, reason='Need a gpu to test cuda')
def test_afb1d_periodic_gpu():
    y1 = lowlevel.afb1d_periodic(x1.cuda(), af.cuda(), d=2)
    y2 = lowlevel.afb1d_periodic(x2.cuda(), af.transpose(3,2).cuda(), d=3)
    np.testing.assert_array_almost_equal(y1[0,0].cpu(), lo1)
    np.testing.assert_array_almost_equal(y1[0,1].cpu(), hi1)
    np.testing.assert_array_almost_equal(y2[0,0].cpu(), lo2)
    np.testing.assert_array_almost_equal(y2[0,1].cpu(), hi2)

def test_afb1d_periodic_channels():
    z = torch.cat((x1, 2*x1), dim=1)
    y1 = lowlevel.afb1d_periodic(z, af, d=2)
    z = torch.cat((x2, 2*x2), dim=1)
    y2 = lowlevel.afb1d_periodic(z, af.transpose(3,2), d=3)

    np.testing.assert_array_almost_equal(y1[0,0], lo1)
    np.testing.assert_array_almost_equal(y1[0,1], hi1)
    np.testing.assert_array_almost_equal(y1[0,2], 2*lo1)
    np.testing.assert_array_almost_equal(y1[0,3], 2*hi1)
    np.testing.assert_array_almost_equal(y2[0,0], lo2)
    np.testing.assert_array_almost_equal(y2[0,1], hi2)
    np.testing.assert_array_almost_equal(y2[0,2], 2*lo2)
    np.testing.assert_array_almost_equal(y2[0,3], 2*hi2)


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
