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
        include_scale (bool): If true, return the bandpass outputs. Can also be
            a list of length J specifying which lowpasses to return. I.e. if
            [False, True, True], the forward call will return the second and
            third lowpass outputs, but discard the lowpass from the first level
            transform.
        downsample (bool): If true, subsample the output lowpass (to match the
            bandpass output sizes)
        C: deprecated, will be removed in future
    """
    def __init__(self, C=None, biort='near_sym_a', qshift='qshift_a',
                 J=3, skip_hps=False, o_before_c=False, include_scale=False,
                 downsample=False):
        super().__init__()
        if C is not None:
            warnings.warn('C parameter is deprecated. do not need to pass it '
                          'anymore.')

        self.biort = biort
        self.qshift = qshift
        self.o_before_c = o_before_c
        self.J = J
        self.downsample = downsample
        h0o, _, h1o, _ = _biort(biort)
        self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
        h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
        self.h0a = torch.nn.Parameter(prep_filt(h0a, 1), False)
        self.h0b = torch.nn.Parameter(prep_filt(h0b, 1), False)
        self.h1a = torch.nn.Parameter(prep_filt(h1a, 1), False)
        self.h1b = torch.nn.Parameter(prep_filt(h1b, 1), False)

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
                :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)`, or
                :math:`list(N, 6, C_{in}, H_{in}'', W_{in}'', 2)` if o_before_c
                was true.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` are the shapes of a
            DTCWT pyramid.
        """
        coeffs = self.dtcwt_func.apply(
            x, self.h0o, self.h1o, self.h0a, self.h0b, self.h1a,
            self.h1b, self.skip_hps, self.o_before_c, self.include_scale)

        if True in self.include_scale:
            if self.downsample:
                lps = tuple([c[:,:, ::2, ::2] for c in coeffs[:self.J]])
                return lps, coeffs[self.J:]
            else:
                return coeffs[:self.J], coeffs[self.J:]
        else:
            # Return in the format: (yl, yh)
            if self.downsample:
                return coeffs[0][:,:,::2, ::2], coeffs[1:]
            else:
                return coeffs[0], coeffs[1:]


class DTCWTInverse(nn.Module):
    """ 2d DTCWT Inverse

    Args:
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters.
        J (int): Number of levels of decomposition.
        o_before_c (bool): whether or not to put the orientations before the
            channel dimension.
        C: deprecated, will be removed in future
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

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
                yl is a tensor of shape :math:`(N, C_{in}, H_{in}', W_{in}')`
                and yh is a list of  the complex bandpass coefficients of shape
                :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)`, or
                :math:`list(N, 6, C_{in}, H_{in}'', W_{in}'', 2)` if o_before_c
                was true.

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

        return self.dtcwt_func.apply(
            yl, *yh, self.g0o, self.g1o, self.g0a, self.g0b, self.g1a, self.g1b,
            self.o_before_c)
