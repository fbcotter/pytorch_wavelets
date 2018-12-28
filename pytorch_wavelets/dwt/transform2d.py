import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import warnings


class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        C: deprecated, will be removed in future
        downsample (bool): true to downsample the output (as you would expect)
    """
    def __init__(self, C=None, J=1, wave='db1', downsample=True):
        super().__init__()
        if C is not None:
            warnings.warn('C parameter is deprecated. do not need to pass it '
                          'anymore.')
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, h1 = wave.dec_lo, wave.dec_hi
        else:
            h0, h1 = wave[0], wave[1]
        ll = np.outer(h0, h0)
        lh = np.outer(h1, h0)
        hl = np.outer(h0, h1)
        hh = np.outer(h1, h1)
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                          hl[None,::-1,::-1], hh[None,::-1,::-1]],
                         axis=0)
        # Make the data contiguous
        filts = np.copy(filts)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
        self.sz = 2*(len(h0) // 2 - 1)
        self.J = J
        if downsample:
            self.stride = 2
        else:
            self.stride = 1

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh)
                coefficients. yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        yl = x
        sz = self.sz

        # Concatenate the filters to match the input size
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)

        # Do a multilevel transform
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

            y = F.conv2d(yl, filters, groups=C, stride=self.stride)
            y = y.reshape((y.shape[0], C, 4, y.shape[-2], y.shape[-1]))
            yl = y[:,:,0].contiguous()
            yh.append(y[:,:,1:].contiguous())

        return yl, yh


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, C=None, wave='db1'):
        super().__init__()
        if C is not None:
            warnings.warn('C parameter is deprecated. do not need to pass it '
                          'anymore.')
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
        else:
            g0, g1 = wave[0], wave[1]
        ll = np.outer(g0, g0)
        lh = np.outer(g1, g0)
        hl = np.outer(g0, g1)
        hh = np.outer(g1, g1)
        filts = np.stack([ll[None,], lh[None,],
                          hl[None,], hh[None,]],
                         axis=0)
        filts = np.copy(filts)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
        self.s = 2*(len(g0) // 2 - 1)

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        """
        yl, yh = coeffs
        ll = yl
        s = self.s

        # Concatenate the filters to match the input size
        C = yl.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]

            y = torch.cat((ll[:,:,None], h), dim=2).reshape(
                ll.shape[0], 4*ll.shape[1], ll.shape[-2], ll.shape[-1])
            ll = F.conv_transpose2d(y, filters, groups=C, stride=2)
            ll = ll[:,:,s:ll.shape[-2]-s,s:ll.shape[-1]-s]
        return ll.contiguous()
