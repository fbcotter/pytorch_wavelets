import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        C (int): Number of channels in input
        J (int): Number of levels of decomposition
        w (str or pywt.Wavelet): Which wavelet to use

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
        self.weight = nn.Parameter(torch.tensor(filts), requires_grad=False)
        self.C = C
        self.sz = 2*(len(w.dec_lo) // 2 - 1)
        self.J = J

    def forward(self, x):
        yh = []
        yl = x
        sz = self.sz
        for j in range(self.J):
            # Pad odd length images
            if yl.shape[-2] % 2 == 1 and yl.shape[-1] % 2 == 1:
                yl = F.pad(yl, (sz, sz+1, sz, sz+1), mode='reflect')
            elif yl.shape[-2] % 2 == 1:
                yl = F.pad(yl, (sz, sz+1, sz, sz), mode='reflect')
            elif yl.shape[-1] % 2 == 1:
                yl = F.pad(yl, (sz, sz, sz, sz+1), mode='reflect')
            else:
                yl = F.pad(yl, (sz, sz, sz, sz), mode='reflect')

            y = F.conv2d(yl, self.weight, groups=self.C, stride=2)
            y = y.reshape((y.shape[0], self.C, 4, y.shape[-2], y.shape[-1]))
            yl = y[:,:,0]
            yh.append(y[:,:,1:])

        return [yl, yh]


class DWTInverse(nn.Module):
    """ 2d DWT Inverse

    Args:
        C (int): Number of channels in input
        wave (str): Which wavelet to use

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
        ll = np.outer(w.rec_lo, w.rec_lo)
        lh = np.outer(w.rec_hi, w.rec_lo)
        hl = np.outer(w.rec_lo, w.rec_hi)
        hh = np.outer(w.rec_hi, w.rec_hi)
        filts = np.stack([ll[None,], lh[None,],
                          hl[None,], hh[None,]],
                         axis=0)
        filts = np.concatenate([filts]*C, axis=0).astype('float32')
        self.weight = nn.Parameter(torch.tensor(filts), requires_grad=False)
        self.C = C
        self.s = 2*(len(w.dec_lo) // 2 - 1)

    def forward(self, coeffs):
        yl, yh = coeffs
        ll = yl
        s = self.s
        for h in yh[::-1]:
            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]

            y = torch.cat((ll[:,:,None], h), dim=2).reshape(
                ll.shape[0], 4*ll.shape[1], ll.shape[-2], ll.shape[-1])
            ll = F.conv_transpose2d(y, self.weight, groups=self.C, stride=2)
            ll = ll[:,:,s:ll.shape[-2]-s,s:ll.shape[-1]-s]
        return ll
