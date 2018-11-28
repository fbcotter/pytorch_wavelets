import torch
import torch.nn as nn
from numpy import ndarray
import warnings

from pytorch_wavelets.dtcwt.coeffs import biort as _biort, qshift as _qshift
from pytorch_wavelets.dtcwt.lowlevel import prep_filt
from pytorch_wavelets.dtcwt import transform_funcs as tf


class DTCWTForward(nn.Module):
    """ Performs a 2d DTCWT Forward decomposition of an image

    Args:
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters.
        J (int): Number of levels of decomposition
        skip_hps list(bool): List of bools of length J which specify whether or
            not to calculate the bandpass outputs at the given scale.
            skip_hps[0] is for the first scale. Can be a single bool in which
            case that is applied to all scales.
        o_before_c (bool): whether or not to put the orientations before the
            channel dimension.
    Shape:
        - Input x: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output yl: :math:`(N, C_{in}, H_{in}', W_{in}')`
        - Output yh: :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)` where
            :math:`H_{in}', W_{in}'` are the shapes of a DTCWT pyramid.
            or :math:`list(N, 6, C_{in}, H_{in}'', W_{in}'', 2)` if o_before_c
            is true


    Attributes:
        h0o (tensor): Non learnable lowpass biorthogonal analysis filter
        h1o (tensor): Non learnable highpass biorthogonal analysis filter
        h0a (tensor): Non learnable lowpass qshift tree a analysis filter
        h1a (tensor): Non learnable highpass qshift tree a analysis filter
        h0b (tensor): Non learnable lowpass qshift tree b analysis filter
        h1b (tensor): Non learnable highpass qshift tree b analysis filter
    """
    def __init__(self, C=None, biort='near_sym_a', qshift='qshift_a',
                 J=3, skip_hps=False, o_before_c=False, include_scale=False):
        super().__init__()
        if C is not None:
            warnings.warn('C parameter is deprecated. do not need to pass it')

        self.biort = biort
        self.qshift = qshift
        self.o_before_c = o_before_c
        self.J = J
        h0o, _, h1o, _ = _biort(biort)
        self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
        h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
        self.h0a = torch.nn.Parameter(prep_filt(h0a, 1), False)
        self.h0b = torch.nn.Parameter(prep_filt(h0b, 1), False)
        self.h1a = torch.nn.Parameter(prep_filt(h1a, 1), False)
        self.h1b = torch.nn.Parameter(prep_filt(h1b, 1), False)

        # Create the function to do the DTCWT
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

    def forward(self, input):
        coeffs = self.dtcwt_func.apply(
            input, self.h0o, self.h1o, self.h0a, self.h0b, self.h1a,
            self.h1b, self.skip_hps, self.o_before_c, self.include_scale)

        if True in self.include_scale:
            return coeffs[:self.J], coeffs[self.J:]
        else:
            # Return in the format: (yl, yh)
            return coeffs[0], coeffs[1:]


class DTCWTInverse(nn.Module):
    """ 2d DTCWT Inverse

    Args:
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters.
        J (int): Number of levels of decomposition
        o_before_c (bool): whether or not to put the orientations before the
            channel dimension.

    Shape:
        - Input yl: :math:`(N, C_{in}, H_{in}', W_{in}')`
        - Input yh: :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)` where
            :math:`H_{in}', W_{in}'` are the shapes of a DTCWT pyramid.
            or :math:`list(N, 6, C_{in}, H_{in}'', W_{in}'', 2)` if o_before_c
            is true
        - Output y: :math:`(N, C_{in}, H_{in}, W_{in})`

    Note:
        Can accept Nones or an empty tensor (torch.tensor([])) for the lowpass
        or bandpass inputs. In this cases, an array of zeros replaces that
        input.

    Attributes:
        g0o (tensor): Non learnable lowpass biorthogonal synthesis filter
        g1o (tensor): Non learnable highpass biorthogonal synthesis filter
        g0a (tensor): Non learnable lowpass qshift tree a synthesis filter
        g1a (tensor): Non learnable highpass qshift tree a synthesis filter
        g0b (tensor): Non learnable lowpass qshift tree b synthesis filter
        g1b (tensor): Non learnable highpass qshift tree b synthesis filter
    """

    def __init__(self, C=None, biort='near_sym_a', qshift='qshift_a', J=3,
                 o_before_c=False):
        super().__init__()
        if C is not None:
            warnings.warn('C parameter is deprecated. do not need to pass it')
        self.biort = biort
        self.qshift = qshift
        self.o_before_c = o_before_c
        self.J = J
        _, g0o, _, g1o = _biort(biort)
        self.g0o = torch.nn.Parameter(prep_filt(g0o, 1), False)
        self.g1o = torch.nn.Parameter(prep_filt(g1o, 1), False)
        _, _, g0a, g0b, _, _, g1a, g1b = _qshift(qshift)
        self.g0a = torch.nn.Parameter(prep_filt(g0a, 1), False)
        self.g0b = torch.nn.Parameter(prep_filt(g0b, 1), False)
        self.g1a = torch.nn.Parameter(prep_filt(g1a, 1), False)
        self.g1b = torch.nn.Parameter(prep_filt(g1b, 1), False)

        # Create the function to do the DTCWT
        self.dtcwt_func = getattr(tf, 'ifm{J}'.format(J=J))

    def forward(self, x):
        yl, yh = x
        for s in yh:
            if s is not None and s.shape != torch.Size([0]):
                if self.o_before_c:
                    assert s.shape[1] == 6, "Inverse transform must have " \
                        "input with 6 orientations"
                    assert len(s.shape) == 6, "Bandpass inputs must have " \
                        "shape (n, 6, c, h, w, 2)"
                else:
                    assert s.shape[2] == 6, "Inverse transform must have " \
                        "input with 6 orientations"
                    assert len(s.shape) == 6, "Bandpass inputs must have " \
                        "shape (n, c, 6, h, w, 2)"
                assert s.shape[-1] == 2, "Inputs must be complex with real " \
                    "and imaginary parts in the last dimension"

        return self.dtcwt_func.apply(yl, *yh, self.g0o, self.g1o, self.g0a,
                                     self.g0b, self.g1a, self.g1b,
                                     self.o_before_c)
