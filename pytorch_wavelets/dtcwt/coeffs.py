"""Functions to load standard wavelet coefficients.

"""
from __future__ import absolute_import

from numpy import load
from pkg_resources import resource_stream
try:
    import pywt
    _HAVE_PYWT = True
except ImportError:
    _HAVE_PYWT = False

COEFF_CACHE = {}


def _load_from_file(basename, varnames):

    try:
        mat = COEFF_CACHE[basename]
    except KeyError:
        with resource_stream('pytorch_wavelets.dtcwt.data', basename + '.npz') as f:
            mat = dict(load(f))
        COEFF_CACHE[basename] = mat

    try:
        return tuple(mat[k] for k in varnames)
    except KeyError:
        raise ValueError(
            'Wavelet does not define ({0}) coefficients'.format(
                ', '.join(varnames)))


def biort(name):
    """ Deprecated. Use :py::func:`pytorch_wavelets.dtcwt.coeffs.level1`
    Instead
    """
    return level1(name, compact=True)


def level1(name, compact=False):
    """Load level 1 wavelet by name.

    :param name: a string specifying the wavelet family name
    :returns: a tuple of vectors giving filter coefficients

    =============  ============================================
    Name           Wavelet
    =============  ============================================
    antonini       Antonini 9,7 tap filters.
    farras         Farras 8,8 tap filters
    legall         LeGall 5,3 tap filters.
    near_sym_a     Near-Symmetric 5,7 tap filters.
    near_sym_b     Near-Symmetric 13,19 tap filters.
    near_sym_b_bp  Near-Symmetric 13,19 tap filters + BP filter
    =============  ============================================

    Return a tuple whose elements are a vector specifying the h0o, g0o, h1o and
    g1o coefficients.

    See :ref:`rot-symm-wavelets` for an explanation of the ``near_sym_b_bp``
    wavelet filters.

    :raises IOError: if name does not correspond to a set of wavelets known to
        the library.
    :raises ValueError: if name doesn't specify
        :py:func:`pytorch_wavelets.dtcwt.coeffs.qshift` wavelet.

    """
    if compact:
        if name == 'near_sym_b_bp':
            return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o', 'h2o', 'g2o'))
        else:
            return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o'))
    else:
        return _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b',
                                      'g1a', 'g1b'))


def qshift(name):
    """Load level >=2 wavelet by name,

    :param name: a string specifying the wavelet family name
    :returns: a tuple of vectors giving filter coefficients

    ============ ============================================
    Name         Wavelet
    ============ ============================================
    qshift_06    Quarter Sample Shift Orthogonal (Q-Shift) 10,10 tap filters,
                 (only 6,6 non-zero taps).
    qshift_a     Q-shift 10,10 tap filters,
                 (with 10,10 non-zero taps, unlike qshift_06).
    qshift_b     Q-Shift 14,14 tap filters.
    qshift_c     Q-Shift 16,16 tap filters.
    qshift_d     Q-Shift 18,18 tap filters.
    qshift_b_bp  Q-Shift 18,18 tap filters + BP
    ============ ============================================

    Return a tuple whose elements are a vector specifying the h0a, h0b, g0a,
    g0b, h1a, h1b, g1a and g1b coefficients.

    See :ref:`rot-symm-wavelets` for an explanation of the ``qshift_b_bp``
    wavelet filters.

    :raises IOError: if name does not correspond to a set of wavelets known to
        the library.
    :raises ValueError: if name doesn't specify a
        :py:func:`pytorch_wavelets.dtcwt.coeffs.biort` wavelet.

    """
    if name == 'qshift_b_bp':
        return _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b',
                                      'g1a', 'g1b', 'h2a', 'h2b', 'g2a','g2b'))
    else:
        return _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b',
                                      'g1a', 'g1b'))


def pywt_coeffs(name):
    """ Wraps pywt Wavelet function. """
    if not _HAVE_PYWT:
        raise ImportError("Could not find PyWavelets module")
    return pywt.Wavelet(name)

# vim:sw=4:sts=4:et
