from __future__ import absolute_import

import numpy as np

from dtcwt.numpy import Transform2d as Transform2d_np
from dtcwt.numpy import Pyramid


def appropriate_complex_type_for(X):
    """Return an appropriate complex data type depending on the type of X. If X
    is already complex, return that, if it is floating point return a complex
    type of the appropriate size and if it is integer, choose an complex
    floating point type depending on the result of :py:func:`numpy.asfarray`.

    """
    X = asfarray(X)
    if np.issubsctype(X.dtype, np.complex64) or \
            np.issubsctype(X.dtype, np.complex128):
        return X.dtype
    elif np.issubsctype(X.dtype, np.float32):
        return np.complex64
    elif np.issubsctype(X.dtype, np.float64):
        return np.complex128

    # God knows, err on the side of caution
    return np.complex128


def asfarray(X):
    """Similar to :py:func:`numpy.asfarray` except that this function tries to
    preserve the original datatype of X if it is already a floating point type
    and will pass floating point arrays through directly without copying.

    """
    X = np.asanyarray(X)
    return np.asfarray(X, dtype=X.dtype)


class Transform2d(object):
    """
    An implementation of the 2D DT-CWT via numpy.

    Parameters
    ----------
    biort: str or np.array
        The biorthogonal wavelet family to use. If a string, will use this to
        call pytorch_wavelets.dtcwt.coeffs.biort. If an array, will use these as the values.
    qshift: str or np.array
        The quarter shift wavelet family to use. If a string, will use this to
        call pytorch_wavelets.dtcwt.coeffs.biort. If an array, will use these as the values.

    .. note::

        *biort* and *qshift* are the wavelets which parameterise the transform.
        If *biort* or *qshift* are strings, they are used as an argument to the
        :py:func:`dtcwt.coeffs.biort` or :py:func:`dtcwt.coeffs.qshift`
        functions.  Otherwise, they are interpreted as tuples of vectors giving
        filter coefficients. In the *biort* case, this should be (h0o, g0o, h1o,
        g1o). In the *qshift* case, this should be (h0a, h0b, g0a, g0b, h1a,
        h1b, g1a, g1b).

    .. note::

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
    """
    def __init__(self, biort='near_sym_a', qshift='qshift_a'):
        self.xfm = Transform2d_np(biort, qshift)

    def forward(self, X, nlevels=3, include_scale=False):
        """ Perform a forward transform on an image with multiple channels.

        Will perform the DTCWT independently on each channel. Data format for
        the input must have the height and width as the last 2 dimensions.

        Parameters
        ----------
        X: np.array
            Input image which you wish to transform. Can be 2, 3, or 4
            dimensions, but height and width must be the last 2.
        nlevels: int
            Number of levels of the dtcwt transform to calculate.
        include_scale: bool
            Whether or not to return the lowpass results at each scale of the
            transform, or only at the highest scale (as is custom for
            multiresolution analysis)

        Returns
        -------
            Yl: ndarray
                Lowpass output
            Yh: list(ndarray)
                Highpass outputs. Will be complex and have one more dimension
                than the input representing the 6 orientations of the wavelets.
                This extra dimension will be the third last dimension. The first
                entry in the list is the first scale.
            Yscale: list(ndarray)
                Only returns if include_scale was true. A list of lowpass
                outputs at each scale.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
        """
        # Reshape the inputs to all be 3d inputs of shape (batch, h, w)
        X = asfarray(X)
        s = X.shape
        if len(s) == 2:
            X = np.reshape(X, (1, *s))
        elif len(s) == 4:
            X = np.reshape(X, (s[0]*s[1], s[2], s[3]))

        # Do the dtcwt now with a 3 dimensional input
        p = self.xfm.forward(X[0], nlevels, include_scale)
        Yl = np.zeros((X.shape[0], *p.lowpass.shape), dtype=X.dtype)
        Yh = [np.zeros((X.shape[0], 6, *p.highpasses[i].shape[0:2]),
                       dtype=appropriate_complex_type_for(X)) for i in
              range(nlevels)]
        if include_scale:
            Yscale = [np.zeros((X.shape[0], *p.scales[i].shape), dtype=X.dtype)
                      for i in range(nlevels)]
        Yl[0] = p.lowpass
        for i in range(nlevels):
            Yh[i][0] = p.highpasses[i].transpose((2,0,1))
            if include_scale:
                Yscale[i][0] = p.scales[i]

        for n in range(1, X.shape[0]):
            p = self.xfm.forward(X[n], nlevels, include_scale)
            Yl[n] = p.lowpass
            for i in range(nlevels):
                Yh[i][n] = p.highpasses[i].transpose((2,0,1))
                if include_scale:
                    Yscale[i][n] = p.scales[i]

        # Reshape output to match input
        if len(s) == 2:
            Yl = Yl[0]
            Yh = [Yh[i][0] for i in range(nlevels)]
            if include_scale:
                Yscale = [Yscale[i][0] for i in range(nlevels)]
        elif len(s) == 4:
            Yl = np.reshape(Yl, (s[0], s[1], *Yl.shape[-2:]))
            Yh = [np.reshape(Yh[i], (s[0], s[1], *Yh[i].shape[-3:])) for i in
                  range(nlevels)]
            if include_scale:
                Yscale = [np.reshape(Yscale[i],
                                     (s[0], s[1], *Yscale[i].shape[-2:]))
                          for i in range(nlevels)]

        if include_scale:
            return Yl, Yh, Yscale
        else:
            return Yl, Yh

    def inverse(self, Yl, Yh, gain_mask=None):
        """
        Perform an inverse transform on an image with multiple channels.

        Parameters
        ----------
        Yl: ndarray
            The lowpass coefficients. Can be 2, 3, or 4 dimensions
        Yh: list(ndarray)
            The complex high pass coefficients. Must be compatible with the
            lowpass coefficients. Should have one more dimension. E.g if Yl
            was of shape [batch, ch, h, w], then the Yh's should be each of
            shape [batch, ch, 6, h', w'] (with h' and w' being dependent on the
            scale).
        gain_mask: None or ndarray
            Can use this to set subbands to have non-unit gain. Should be
            anarray of size [nlevels, 6] and can be complex or real. Useful for
            masking out subbands.

        Returns
        -------
        X: ndarray
            An array , X, compatible with the reconstruction.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
        """
        J = len(Yh)
        s = Yl.shape

        # Reshape the inputs to all be 3d inputs of shape (batch, h, w)
        if len(s) == 2:
            Yl = np.reshape(Yl, (1, *s))
            Yh = [np.reshape(Yh[i], (1, *Yh[i].shape)) for i in range(J)]
        elif len(s) == 4:
            Yl = np.reshape(Yl, (s[0]*s[1], *s[-2:]))
            Yh = [np.reshape(Yh[i], (s[0]*s[1], *Yh[i].shape[-3:]))
                  for i in range(J)]

        # Do the inverse dtcwt now with a 3 dimensional input
        X = self.xfm.inverse(
            Pyramid(Yl[0], [np.transpose(Yh[i][0], (1,2,0)) for i in range(J)]),
            gain_mask=gain_mask)

        x = np.zeros((Yl.shape[0], *X.shape), dtype=X.dtype)
        x[0] = X
        for n in range(1, Yl.shape[0]):
            X = self.xfm.inverse(
                Pyramid(Yl[n], [np.transpose(Yh[i][n], (1,2,0))
                                for i in range(J)]))
            x[n] = X

        # Reshape output to match input
        if len(s) == 2:
            X = x[0]
        elif len(s) == 4:
            X = np.reshape(x, (s[0], s[1], *x.shape[-2:]))
        else:
            X = x

        return X
