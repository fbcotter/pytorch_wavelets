DWT in Pytorch Wavelets
=======================

While pytorch_wavelets was initially built as a repo to do the dual tree wavelet
transform efficiently in pytorch, I have also built a thin wrapper over
PyWavelets, allowing the calculation of the 2D-DWT in pytorch on a GPU on
a batch of images. 

Older versions did the DWT non separably. As of v1.0.0 we now
have code to do it separably. The old non-separable code is still there and is
surprisingly sometimes faster. You can test the two out to see which is better
for you by changing the `separable` flag in the DWT/IDWT constructor.

The DWT/IDWT now supports most of the padding schemes that PyWavelets uses. In
particular:

- symmetric padding
- reflection padding
- zero padding
- periodization 

You can see the source `here <_modules/pytorch_wavelets/dwt/transform2d.html#DWTForward>`_. 
It is pretty minimal and should be clear what is going on.

In particular, the DWT and IWT classes initialize the filter banks as pytorch
tensors (taking care to flip them as pytorch uses cross-correlation not
convolution). It then performs non-separable 2D convolution on the input, using
strided convolution to calculate the LL, LH, HL, and HH subbands. It also takes
care of padding to match the PyWavelets implementation.

Differences to PyWavelets
-------------------------

Inputs
~~~~~~
The pytorch_wavelets DWT expects the standard pytorch image format of NCHW
- i.e., a batch of N images, with C channels, height H and width W. For a single
RGB image, you would need to make it a torch tensor of size :code:`(1, 3, H,
W)`, or for a batch of 100 grayscale images, you would need to make it a tensor
of size :code:`(100, 1, H, W)`.

Returned Coefficients
~~~~~~~~~~~~~~~~~~~~~
We deviate slightly from PyWavelets with the format of the returned
coefficients.  In particular, we return a tuple of :code:`(yl, yh)` where yl is
the LL band, and `yh` is a list. The first list entry `yh[0]` are the scale
1 bandpass coefficients (finest resolution), and the last list entry `yh[-1]`
are the coarsest bandpass coefficients. Note that this is the reverse of the
PyWavelets format (but fits with the dtcwt standard output). Each of the bands
is a single stacked tensor of the LH (horiz), HL (vertic), and HH (diag)
coefficients for each scale (as opposed to PyWavelets style of returning as
a tuple) with the stack along the third dimension. As the input had
4 dimensions, this output has 5 dimensions, with shape :code:`(N, C, 3, H, W)`. 
This is easily transformed into the PyWavelets style by unstacking the
list elements in `yh`.

Example
-------

.. code:: python

    import torch
    from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
    xfm = DWTForward(J=3, mode='zero', wave='db3')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='db3')
    X = torch.randn(10,5,64,64)
    Yl, Yh = xfm(X) 
    print(Yl.shape)
    >>> torch.Size([10, 5, 12, 12])
    print(Yh[0].shape) 
    >>> torch.Size([10, 5, 3, 34, 34])
    print(Yh[1].shape)
    >>> torch.Size([10, 5, 3, 19, 19])
    print(Yh[2].shape)
    >>> torch.Size([10, 5, 3, 12, 12])
    Y = ifm((Yl, Yh))
    import numpy as np
    np.testing.assert_array_almost_equal(Y.cpu().numpy(), X.cpu().numpy())

Other Notes
-----------
GPU Calculations
~~~~~~~~~~~~~~~~
As you would expect, you can move the transforms to the GPU by calling
:code:`xfm.cuda()` or :code:`ifm.cuda()`, where `xfm`, `ifm` are instances of
:class:`pytorch_wavelets.DWTForward` and :class:`pytorch_wavelets.DWTInverse`.

