import numpy as np
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch
import py3nvml


HAVE_GPU = torch.cuda.is_available()


def setup():
    py3nvml.grab_gpus(1)


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
