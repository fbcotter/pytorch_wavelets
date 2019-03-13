import torch
import torch.nn as nn
from numpy import ndarray
from math import sqrt
import numpy as np

from pytorch_wavelets.dtcwt.coeffs import qshift as _qshift, biort as _biort
from pytorch_wavelets.dtcwt.lowlevel import prep_filt
import pytorch_wavelets.dwt.lowlevel as lowlevel
from pytorch_wavelets.dtcwt import transform_funcs as tf


class Scatter(nn.Module):
    """ Performs a single scale scattering with DTCWTs """
    def __init__(self, biort='near_sym_a', separable=True):
        super().__init__()
        self.biort = biort
        self.separable = separable
        if isinstance(biort, str):
            h0, _, h1, _ = _biort(biort)
        else:
            h0, h1 = biort
        if h0.shape[0] < h1.shape[0]:
            h0 = np.pad(h0, ((1,1), (0,0)), 'constant')
        elif h1.shape[0] < h0.shape[0]:
            h1 = np.pad(h1, ((1,1), (0,0)), 'constant')
        # Prepare the filters
        if separable:
            filts = lowlevel.prep_filt_afb2d(h0, h1, h0, h1)
            self.h0_col = nn.Parameter(filts[0], requires_grad=False)
            self.h1_col = nn.Parameter(filts[1], requires_grad=False)
            self.h0_row = nn.Parameter(filts[2], requires_grad=False)
            self.h1_row = nn.Parameter(filts[3], requires_grad=False)
            self.h = (self.h0_col, self.h1_col, self.h0_row, self.h1_row)
        else:
            filts = lowlevel.prep_filt_afb2d_nonsep(h0, h1, h0, h1)
            self.h = nn.Parameter(filts, requires_grad=False)
        self.b = 1e-4

    def forward(self, x):
        # If the row/col count of X is not divisible by 2 then we need to
        # extend X
        batch, ch, r, c = x.shape
        if r % 2 != 0:
            x = torch.cat((x, x[:,:,-1:]), dim=2)
        if c % 2 != 0:
            x = torch.cat((x, x[:,:,:,-1:]), dim=3)
        batch, ch, r, c = x.shape

        if self.separable:
            y = lowlevel.afb2d(x, self.h, mode='symmetric',
                                      decimate=False)
        else:
            y = lowlevel.afb2d_nonsep(x, self.h, mode='symmetric', decimate=False)
        y = y.view(batch, ch, 4, r, c)
        lo = y[:,:,0]
        hi = y[:,:,1:]/sqrt(2)
        r1 = hi[..., ::2, ::2] - hi[..., 1::2,1::2]
        r2 = hi[..., ::2, ::2] + hi[..., 1::2,1::2]
        i1 = hi[..., ::2, 1::2] + hi[..., 1::2, ::2]
        i2 = hi[..., ::2, 1::2] - hi[..., 1::2, ::2]
        m1 = torch.sqrt(r1**2 + i1**2 + self.b) - sqrt(self.b)
        m2 = torch.sqrt(r2**2 + i2**2 + self.b) - sqrt(self.b)
        return lo, torch.cat((m1, m2), dim=2).view(batch, ch*6, r//2, c//2)


class DTCWTForward(nn.Module):
    """ Performs a 2d DTCWT Forward decomposition of an image

    Args:
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters. Can also
            give a two tuple for the low and highpass filters directly.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters. Can
            also give a 4-tuple for the low tree a, low tree b, high tree a and
            high tree b filters directly.
        J (int): Number of levels of decomposition
        skip_hps (bools): List of bools of length J which specify whether or
            not to calculate the bandpass outputs at the given scale.
            skip_hps[0] is for the first scale. Can be a single bool in which
            case that is applied to all scales.
        include_scale (bool): If true, return the bandpass outputs. Can also be
            a list of length J specifying which lowpasses to return. I.e. if
            [False, True, True], the forward call will return the second and
            third lowpass outputs, but discard the lowpass from the first level
            transform.
        downsample (bool): If true, subsample the output lowpass (to match the
            bandpass output sizes)
        o_dim (int): Which dimension to put the orientations in
        ri_dim (int): which dimension to put the real and imaginary parts
    """
    def __init__(self, biort='near_sym_a', qshift='qshift_a',
                 J=3, skip_hps=False, include_scale=False,
                 downsample=False, o_dim=2, ri_dim=-1):
        super().__init__()
        if o_dim == ri_dim:
            raise ValueError("Orientations and real/imaginary parts must be "
                             "in different dimensions.")

        self.biort = biort
        self.qshift = qshift
        self.J = J
        self.downsample = downsample
        self.o_dim = o_dim
        self.ri_dim = ri_dim
        if isinstance(biort, str):
            h0o, _, h1o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
        else:
            self.h0o = torch.nn.Parameter(prep_filt(biort[0], 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(biort[1], 1), False)
        if isinstance(qshift, str):
            h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
            self.h0a = torch.nn.Parameter(prep_filt(h0a, 1), False)
            self.h0b = torch.nn.Parameter(prep_filt(h0b, 1), False)
            self.h1a = torch.nn.Parameter(prep_filt(h1a, 1), False)
            self.h1b = torch.nn.Parameter(prep_filt(h1b, 1), False)
        else:
            self.h0a = torch.nn.Parameter(prep_filt(qshift[0], 1), False)
            self.h0b = torch.nn.Parameter(prep_filt(qshift[1], 1), False)
            self.h1a = torch.nn.Parameter(prep_filt(qshift[2], 1), False)
            self.h1b = torch.nn.Parameter(prep_filt(qshift[3], 1), False)

        # Get the function to do the DTCWT
        if isinstance(skip_hps, (list, tuple, ndarray)):
            self.skip_hps = skip_hps
        else:
            self.skip_hps = [skip_hps,] * self.J
        if isinstance(include_scale, (list, tuple, ndarray)):
            self.include_scale = include_scale
        else:
            self.include_scale = [include_scale,] * self.J
        if True in self.include_scale:
            self.dtcwt_func = getattr(tf, 'xfm{J}scale'.format(J=J))
        else:
            self.dtcwt_func = getattr(tf, 'xfm{J}'.format(J=J))

    def forward(self, x):
        """ Forward Dual Tree Complex Wavelet Transform

        Args:
            x (tensor): Input to transform. Should be of shape
                :math:`(N, C_{in}, H_{in}, W_{in})`.

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                If include_scale was true, yl will be a list of lowpass
                coefficients, otherwise will be just the final lowpass
                coefficient of shape :math:`(N, C_{in}, H_{in}', W_{in}')`. Yh
                will be a list of the complex bandpass coefficients of shape
                :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)`, or similar
                shape depending on o_dim and ri_dim

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` are the shapes of a
            DTCWT pyramid.
        """
        coeffs = self.dtcwt_func.apply(
            x, self.h0o, self.h1o, self.h0a, self.h0b, self.h1a,
            self.h1b, self.skip_hps, self.include_scale, self.o_dim,
            self.ri_dim, self.downsample)

        if True in self.include_scale:
            return coeffs[:self.J], coeffs[self.J:]
        else:
            return coeffs[0], coeffs[1:]


class DTCWTInverse(nn.Module):
    """ 2d DTCWT Inverse

    Args:
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters. Can also
            give a two tuple for the low and highpass filters directly.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters. Can
            also give a 4-tuple for the low tree a, low tree b, high tree a and
            high tree b filters directly.
        J (int): Number of levels of decomposition.
        o_dim (int):which dimension the orientations are in
        ri_dim (int): which dimension to put th real and imaginary parts in
    """

    def __init__(self, biort='near_sym_a', qshift='qshift_a', J=3,
                 o_dim=2, ri_dim=-1):
        super().__init__()
        self.biort = biort
        self.qshift = qshift
        self.o_dim = o_dim
        self.ri_dim = ri_dim
        self.J = J
        if isinstance(biort, str):
            _, g0o, _, g1o = _biort(biort)
            self.g0o = torch.nn.Parameter(prep_filt(g0o, 1), False)
            self.g1o = torch.nn.Parameter(prep_filt(g1o, 1), False)
        else:
            self.g0o = torch.nn.Parameter(prep_filt(biort[0], 1), False)
            self.g1o = torch.nn.Parameter(prep_filt(biort[1], 1), False)
        if isinstance(qshift, str):
            _, _, g0a, g0b, _, _, g1a, g1b = _qshift(qshift)
            self.g0a = torch.nn.Parameter(prep_filt(g0a, 1), False)
            self.g0b = torch.nn.Parameter(prep_filt(g0b, 1), False)
            self.g1a = torch.nn.Parameter(prep_filt(g1a, 1), False)
            self.g1b = torch.nn.Parameter(prep_filt(g1b, 1), False)
        else:
            self.g0a = torch.nn.Parameter(prep_filt(qshift[0], 1), False)
            self.g0b = torch.nn.Parameter(prep_filt(qshift[1], 1), False)
            self.g1a = torch.nn.Parameter(prep_filt(qshift[2], 1), False)
            self.g1b = torch.nn.Parameter(prep_filt(qshift[3], 1), False)

        # Create the function to do the DTCWT
        self.dtcwt_func = getattr(tf, 'ifm{J}'.format(J=J))

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
                yl is a tensor of shape :math:`(N, C_{in}, H_{in}', W_{in}')`
                and yh is a list of  the complex bandpass coefficients of shape
                :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)`, or similar
                depending on o_dim and ri_dim

        Returns:
            Reconstructed output

        Note:
            Can accept Nones or an empty tensor (torch.tensor([])) for the lowpass
            or bandpass inputs. In this cases, an array of zeros replaces that
            input.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` are the shapes of a
            DTCWT pyramid.

        Note:
            If include_scale was true for the forward pass, you should provide
            only the final lowpass output here, as normal for an inverse wavelet
            transform.

        Note:
            Won't work if the forward transform lowpass was downsampled.
        """
        yl, yh = coeffs
        for s in yh:
            if s is not None and s.shape != torch.Size([0]):
                assert s.shape[self.o_dim] == 6, "Inverse transform must " \
                    "have input with 6 orientations"
                assert len(s.shape) == 6, "Bandpass inputs must have " \
                    "6 dimensions"
                assert s.shape[self.ri_dim] == 2, "Inputs must be complex " \
                    "with real and imaginary parts in the ri dimension"
        assert len(yh) == self.J, "The input provided has more scales than J"

        return self.dtcwt_func.apply(
            yl, *yh, self.g0o, self.g1o, self.g0a, self.g0b, self.g1a, self.g1b,
            self.o_dim, self.ri_dim)
