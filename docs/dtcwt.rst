DTCWT in Pytorch Wavelets
=========================

Pytorch wavelets is a port of `dtcwt_slim`__, which was my first attempt at
doing the DTCWT quickly on a GPU. It has since been cleaned up to run for
pytorch and do the quickest forward and inverse transforms I can make, as well
as being able to pass gradients through the inputs.

For those unfamiliar with the DTCWT, it is a shift invariant wavelet transform
that comes with limited redundancy. Compared to the undecimated wavelet
transform, which has :math:`2^J` redundancy, the DTCWT only has :math:`2^d`
redundancy (where d is the number of input dimensions - i.e. 4:1 redundancy for
image transforms). Instead of producing 3 output subbands like the DWT, it
produces 6, which roughly represent 15, 45, 75, 105, 135 and 165 degree
wavelets. On top of this, the 6 subbands have real and imaginary outputs which
are in quadrature with each other (similar to windowed sine and cosine
functions, or the gabor wavelet).

It is possible to calculate similar transforms (such as the morlet or gabor)
using fourier transforms, but the DTCWT is faster as it uses separable
convolutions.

.. image:: dtcwt_bands.png

__ https://github.com/fbcotter/dtcwt_slim

Notes
-----
Because of the above mentioned properties of the DTCWT, the output is slightly
different to the DWT. As mentioned, it is 4:1 redundant in 2D, so we expect
4 times as many coefficients as from the decimated wavelet transform. These
extra coefficients come from:

- 6 subband outputs instead of 3 with ial and imaginary coefficients instead 
  of just real. For an :math:`N \times N` image, the first level bandpass has
  :math:`3N^2` instead of :math:`3N^2/4` coefficients.
- The lowpass is always at double the resolution of what you'd expect it to be
  for the level in the wavelet tree. I.e. for a 1 level transform, the lowapss
  output is still :math:`N\times N`. For a two level transform, it is :math:`N/2
  \times N/2` and so on.  

Example
-------

.. code:: python

    import torch
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
    X = torch.randn(10,5,64,64)
    Yl, Yh = xfm(X) 
    print(Yl.shape)
    >>> torch.Size([10, 5, 16, 16])
    print(Yh[0].shape) 
    >>> torch.Size([10, 5, 6, 32, 32, 2])
    print(Yh[1].shape)
    >>> torch.Size([10, 5, 6, 16, 16, 2])
    print(Yh[2].shape)
    >>> torch.Size([10, 5, 6, 8, 8, 2])
    ifm = DTCWTInverse(J=3, biort='near_sym_b', qshift='qshift_b')
    Y = ifm((Yl, Yh))

Like with the DWT, Yh returned is a tuple. There are 2 extra dimensions - the
first comes between the channel dimension of the input and the row dimension.
This is the 6 orientations of the DTCWT. The second is the final dimension, which is the
real an imaginary parts (complex numbers are not native to pytorch). I.e. to
access the real part of the 45 degree wavelet for the first subband, you would
use :code:`Yh[0][:,:,1,:,:,0]`, and the imaginary part of the 165 degree wavelet
would be :code:`Yh[0][:,:,5,:,:,1]`. 

The above images were created by doing a forward transform with an input of
zeros (creates a pyramid with the correct size bands), and then setting the
centre spatial value to 1 for each of the orientations at the third scale. I.e.:

.. code:: python

    import numpy as np
    import torch
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    xfm = DTCWTForward(J=3)
    ifm = DTCWTInverse(J=3)
    x = torch.zeros(1,1,64,64)
    # Create 12 outputs, one for the real and imaginary point spread functions
    # for each of the 6 orientations
    out = np.zeros((12,64,64)
    yl, yh = xfm(x)
    for b in range(6):
      for ri in range(2):
        yh[2][0,0,b,4,4,ri] = 1
        out[b*2 + ri] = ifm((yl, yh))
        yh[2][0,0,b,4,4,ri] = 0
    # Can now plot the output

