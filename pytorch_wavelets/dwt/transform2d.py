import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch


class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        separable (bool): whether to do the filtering separably or not (the
            naive implementation can be faster on a gpu).
    """
    def __init__(self, J=1, wave='db1', mode='zero', separable=True):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        if separable:
            filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
            self.h0_col = nn.Parameter(filts[0], requires_grad=False)
            self.h1_col = nn.Parameter(filts[1], requires_grad=False)
            self.h0_row = nn.Parameter(filts[2], requires_grad=False)
            self.h1_row = nn.Parameter(filts[3], requires_grad=False)
            self.h = (self.h0_col, self.h1_col, self.h0_row, self.h1_row)
        else:
            filts = lowlevel.prep_filt_afb2d_nonsep(h0_col, h1_col, h0_row, h1_row)
            self.h = nn.Parameter(filts, requires_grad=False)
        self.J = J
        self.mode = mode
        self.separable = separable

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
        ll = x

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            if self.separable:
                y = lowlevel.afb2d(ll, self.h, self.mode)
            else:
                y = lowlevel.afb2d_nonsep(ll, self.h, self.mode)

            # Separate the low and bandpasses
            s = y.shape
            y = y.reshape(s[0], -1, 4, s[-2], s[-1])
            ll = y[:,:,0].contiguous()
            yh.append(y[:,:,1:].contiguous())

        return ll, yh


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, wave='db1', mode='zero', separable=True):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        if separable:
            filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
            self.g0_col = nn.Parameter(filts[0], requires_grad=False)
            self.g1_col = nn.Parameter(filts[1], requires_grad=False)
            self.g0_row = nn.Parameter(filts[2], requires_grad=False)
            self.g1_row = nn.Parameter(filts[3], requires_grad=False)
            self.h = (self.g0_col, self.g1_col, self.g0_row, self.g1_row)
        else:
            filts = lowlevel.prep_filt_sfb2d_nonsep(g0_col, g1_col, g0_row, g1_row)
            self.h = nn.Parameter(filts, requires_grad=False)
        self.mode = mode
        self.separable = separable

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

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]

            # Do the synthesis filter banks
            if self.separable:
                lh, hl, hh = torch.unbind(h, dim=2)
                ll = lowlevel.sfb2d(ll, lh, hl, hh, self.h, mode=self.mode)
            else:
                c = torch.cat((ll[:,:,None], h), dim=2)
                ll = lowlevel.sfb2d_nonsep(c, self.h, mode=self.mode)
        return ll
