""" Useful utilities for testing the 2-D DTCWT with synthetic images"""

from __future__ import absolute_import

import functools
import numpy as np


def unpack(pyramid, backend='numpy'):
    """ Unpacks a pyramid give back the constituent parts.

    :param pyramid: The Pyramid of DTCWT transforms you wish to unpack
    :param str backend: A string from 'numpy', 'opencl', or 'tf' indicating
        which attributes you want to unpack from the pyramid.

    :returns: returns a generator which can be unpacked into the Yl, Yh and
        Yscale components of the pyramid. The generator will only return 2
        values if the pyramid was created with the include_scale parameter set
        to false.

    .. note::

        You can still unpack a tf or opencl pyramid as if it were created by a
        numpy. In this case it will return a numpy array, rather than the
        backend specific array type.
    """
    backend = backend.lower()
    if backend == 'numpy':
        yield pyramid.lowpass
        yield pyramid.highpasses
        if pyramid.scales is not None:
            yield pyramid.scales
    elif backend == 'opencl':
        yield pyramid.cl_lowpass
        yield pyramid.cl_highpasses
        if pyramid.cl_scales is not None:
            yield pyramid.cl_scales
    elif backend == 'tf':
        yield pyramid.lowpass_op
        yield pyramid.highpasses_ops
        if pyramid.scales_ops is not None:
            yield pyramid.scales_ops


def drawedge(theta,r,w,N):
    """Generate an image of size N * N pels, of an edge going from 0 to 1 in
    height at theta degrees to the horizontal (top of image = 1 if angle = 0).
    r is a two-element vector, it is a coordinate in ij coords through which the
    step should pass.
    The shape of the intensity step is half a raised cosine w pels wide (w>=1).

    T. E . Gale's enhancement to drawedge() for MATLAB, transliterated
    to Python by S. C. Forshaw, Nov. 2013. """

    # convert theta from degrees to radians
    thetar = np.array(theta * np.pi / 180)

    # Calculate image centre from given width
    imCentre = (np.array([N,N]).T - 1) / 2 + 1

    # Calculate values to subtract from the plane
    r = np.array([np.cos(thetar), np.sin(thetar)])*(-1) * (r - imCentre)

    # check width of raised cosine section
    w = np.maximum(1,w)

    ramp = np.arange(0,N) - (N+1)/2
    hgrad = np.sin(thetar)*(-1) * np.ones([N,1])
    vgrad = np.cos(thetar)*(-1) * np.ones([1,N])
    plane = ((hgrad * ramp) - r[0]) + ((ramp * vgrad).T - r[1])
    x = 0.5 + 0.5 * np.sin(np.minimum(np.maximum(
        plane*(np.pi/w), np.pi/(-2)), np.pi/2))

    return x


def drawcirc(r,w,du,dv,N):

    """Generate an image of size N*N pels, containing a circle
    radius r pels and centred at du,dv relative
    to the centre of the image.  The edge of the circle is a cosine shaped
    edge of width w (from 10 to 90% points).

    Python implementation by S. C. Forshaw, November 2013."""

    # check value of w to avoid dividing by zero
    w = np.maximum(w,1)

    # x plane
    x = np.ones([N,1]) * ((np.arange(0,N,1, dtype='float') -
                          (N+1) / 2 - dv) / r)

    # y vector
    y = (((np.arange(0,N,1, dtype='float') - (N+1) / 2 - du) / r) *
         np.ones([1,N])).T

    # Final circle image plane
    p = 0.5 + 0.5 * np.sin(np.minimum(np.maximum((
            np.exp(np.array([-0.5]) * (x**2 + y**2)).T - np.exp((-0.5))) * (r * 3 / w),  # noqa
        np.pi/(-2)), np.pi/2))
    return p


def asfarray(X):
    """Similar to :py:func:`numpy.asfarray` except that this function tries to
    preserve the original datatype of X if it is already a floating point type
    and will pass floating point arrays through directly without copying.

    """
    X = np.asanyarray(X)
    return np.asfarray(X, dtype=X.dtype)


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


def as_column_vector(v):
    """Return *v* as a column vector with shape (N,1).

    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


def symm_pad_1d(l, m):
    """ Creates indices for symmetric padding. Works for 1-D.

    Inptus:
        l (int): size of input
        m (int): size of filter
    """
    xe = reflect(np.arange(-m, l+m, dtype='int32'), -0.5, l-0.5)
    return xe


# note that this decorator ignores **kwargs
# From https://wiki.python.org/moin/PythonDecoratorLibrary#Alternate_memoize_as_nested_functions  # noqa
def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer


def stacked_2d_matrix_vector_prod(mats, vecs):
    """
    Interpret *mats* and *vecs* as arrays of 2D matrices and vectors. I.e.
    *mats* has shape PxQxNxM and *vecs* has shape PxQxM. The result
    is a PxQxN array equivalent to:

    .. code::

        result[i,j,:] = mats[i,j,:,:].dot(vecs[i,j,:])

    for all valid row and column indices *i* and *j*.
    """
    return np.einsum('...ij,...j->...i', mats, vecs)


def stacked_2d_vector_matrix_prod(vecs, mats):
    """
    Interpret *mats* and *vecs* as arrays of 2D matrices and vectors. I.e.
    *mats* has shape PxQxNxM and *vecs* has shape PxQxN. The result
    is a PxQxM array equivalent to:

    .. code::

        result[i,j,:] = mats[i,j,:,:].T.dot(vecs[i,j,:])

    for all valid row and column indices *i* and *j*.
    """
    vecshape = np.array(vecs.shape + (1,))
    vecshape[-1:-3:-1] = vecshape[-2:]
    outshape = mats.shape[:-2] + (mats.shape[-1],)
    return stacked_2d_matrix_matrix_prod(vecs.reshape(vecshape), mats).reshape(outshape)  # noqa


def stacked_2d_matrix_matrix_prod(mats1, mats2):
    """
    Interpret *mats1* and *mats2* as arrays of 2D matrices. I.e.
    *mats1* has shape PxQxNxM and *mats2* has shape PxQxMxR. The result
    is a PxQxNxR array equivalent to:

    .. code::

        result[i,j,:,:] = mats1[i,j,:,:].dot(mats2[i,j,:,:])

    for all valid row and column indices *i* and *j*.
    """
    return np.einsum('...ij,...jk->...ik', mats1, mats2)
