import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        C (int): Number of channels in input
        w (str or pywt.Wavelet): Which wavelet to use
        J (int): Number of levels of decomposition
        skip_hps (bool): True if the decomposition should not calculate the
            first scale's highpass coefficients (these are often noise). Will
            speed up computation significantly.

    Shape:
        - Input x: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output yl: :math:`(N, C_{in}, H_{in}', W_{in}')`
        - Output yh: :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')` where
            :math:`H_{in}', W_{in}'` are the shapes of a DTCWT pyramid.

    """
    def __init__(self, C, J=1, wave='db1'):
        super().__init__()
        self.wave = wave
        w = pywt.Wavelet(wave)
        ll = np.outer(w.dec_lo, w.dec_lo)
        lh = np.outer(w.dec_hi, w.dec_lo)
        hl = np.outer(w.dec_lo, w.dec_hi)
        hh = np.outer(w.dec_hi, w.dec_hi)
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                          hl[None,::-1,::-1], hh[None,::-1,::-1]],
                         axis=0)
        filts = np.concatenate([filts]*C, axis=0).astype('float32')
        self.weight = nn.Parameter(torch.tensor(filts))
        self.C = C
        s = 2*(len(w.dec_lo) // 2 - 1)
        self.pad = lambda x: F.pad(x, (s,s,s,s), mode='reflect')
        self.J = J

    def forward(self, x):
        yh = []
        yl = x
        for j in range(self.J):
            y = F.conv2d(self.pad(yl), self.weight, groups=self.C, stride=2)
            y = y.reshape((y.shape[0], self.C, 4, y.shape[-2], y.shape[-1]))
            yl = y[:,:,0]
            yh.append(y[:,:,1:])

        return yl, yh


class DWTInverse(nn.Module):
    """ 2d DTCWT Inverse

    Args:
        C (int): Number of channels in input
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters.
        J (int): Number of levels of decomposition
        skip_hps (bool): True if the inverse method should not look at the first
            scale's highpass coefficients (these are often noise). Will speed up
            computation significantly.

    Shape:
        - Input yl: :math:`(N, C_{in}, H_{in}', W_{in}')`
        - Input yh: :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')` where
            :math:`H_{in}', W_{in}'` are the shapes of a DTCWT pyramid.
        - Output y: :math:`(N, C_{in}, H_{in}, W_{in})`

    """
    def __init__(self, C, wave='db1'):
        super().__init__()
        self.wave = wave
        w = pywt.Wavelet(wave)
        ll = np.outer(w.dec_lo, w.dec_lo)
        lh = np.outer(w.dec_hi, w.dec_lo)
        hl = np.outer(w.dec_lo, w.dec_hi)
        hh = np.outer(w.dec_hi, w.dec_hi)
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                          hl[None,::-1,::-1], hh[None,::-1,::-1]],
                         axis=0)
        filts = np.concatenate([filts]*C, axis=0).astype('float32')
        self.weight = nn.Parameter(torch.tensor(filts))
        self.C = C
        self.s = 2*(len(w.dec_lo) // 2 - 1)

    def forward(self, coeffs):
        yl, yh = coeffs
        ll = yl
        s = self.s
        for h in yh[::-1]:
            y = torch.cat((ll[:,:,None], h), dim=2).reshape(
                ll.shape[0], 4*ll.shape[1], ll.shape[-2], ll.shape[-1])
            ll = F.conv_transpose2d(y, self.weight, groups=self.C, stride=2)
            ll = ll[:,:,s:ll.shape[-2]-s,s:ll.shape[-1]-s]
        return ll
