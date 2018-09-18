2D Wavelet Transforms in Pytorch
================================

|Build Status|

.. |Build Status| image:: https://travis-ci.org/fbcotter/pytorch_wavelets.png?branch=master
    :target: https://travis-ci.org/fbcotter/pytorch_wavelets

This package provides support for computing the 2D discrete wavelet and 
the 2d dual-tree complex wavelet transforms, their inverses, and passing 
gradients through both using pytorch.

The implementation is designed to be used with batches of multichannel images.
We use the standard pytorch implementation of having 'NCHW' data format.

This repo originally was only for the use of the DTCWT, but I have added some DWT support. This is still in development,
and has the following known issues:

- Uses reflection padding instead of symmetric padding for the DWT
- Doesn't compute the DWT separably, instead uses the full `N x N` kernel.

Installation
````````````
The easiest way to install ``pytorch_wavelets`` is to clone the repo and pip install
it. Later versions will be released on PyPi but the docs need to updated first::

    $ git clone https://github.com/fbcotter/pytorch_wavelets
    $ cd pytorch_wavelets
    $ python setup.py install # (or pip install .)

(Although the `develop` command may be more useful if you intend to perform any
significant modification to the library.) A test suite is provided so that you
may verify the code works on your system::

    $ pip install -r tests/requirements.txt
    $ pytest tests/

Example Use
```````````
For the DWT - note that the highpass output has an extra dimension, in which we stack the (lh, hl, hh) coefficients:

.. code:: python

    import torch
    from pytorch_wavelets import DWTForward, DWTInverse
    xfm = DWTForward(C=5, J=3, wave='db3')
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
    ifm = DWTInverse(C=5, J=3, wave='db3')
    Y = ifm((Yl, Yh))

For the DTCWT:

.. code:: python

    import torch
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    xfm = DTCWTForward(C=5, J=3, biort='near_sym_b', qshift='qshift_b')
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
    ifm = DTCWTInverse(C=5, J=3, biort='near_sym_b', qshift='qshift_b')
    Y = ifm((Yl, Yh))

Some initial notes:

- You need to specify the number of channels and the number of scales for both
  the forward and inverse transform. Make sure they are the same! The same goes
  for the filter types used.
- Yh returned is a tuple. There are 2 extra dimensions - the first comes between
  the channel dimension of the input and the row dimension. This is the
  6 orientations of the DTCWT. The second is the final dimension, which is the
  real an imaginary parts (complex numbers are not native to pytorch)

Running on the GPU
~~~~~~~~~~~~~~~~~~
This should come as no surprise to pytorch users. The DTCWT transforms support
cuda calling:

.. code:: python

    import torch
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    xfm = DTCWTForward(C=5, J=3, biort='near_sym_b', qshift='qshift_b').cuda()
    X = torch.randn(10,5,64,64).cuda()
    Yl, Yh = xfm(X) 
    ifm = DTCWTInverse(C=5, J=3, biort='near_sym_b', qshift='qshift_b').cuda()
    Y = ifm(Yl, Yh)

The automated tests cannot test the gpu functionality, but do check cpu running.
To test whether the repo is working on your gpu, you can download the repo,
ensure you have pytorch with cuda enabled (the tests will check to see if
:code:`torch.cuda.is_available()` returns true), and run:

.. code:: 

    pip install -r tests/requirements.txt
    pytest tests/

From the base of the repo.

Backpropagation
~~~~~~~~~~~~~~~
It is possible to pass gradients through the forward and backward transforms.
All you need to do is ensure that the input to each has the required_grad
attribute set to true.

Provenance
``````````
Based on the Dual-Tree Complex Wavelet Transform Pack for MATLAB by Nick
Kingsbury, Cambridge University. The original README can be found in
ORIGINAL_README.txt.  This file outlines the conditions of use of the original
MATLAB toolbox.

.. vim:sw=4:sts=4:et
