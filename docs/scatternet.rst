DTCWT ScatterNet in Pytorch Wavelets
====================================

We have moved the DTCWT scatternet over from its original home in
`scatnet_learn`__. It is still there (as well as an improved, learnable
scatternet described in our `paper`__).

The original ScatterNet paper describes the properties of ScatterNet:
`Invariant Scattering Convolution Networks`__.

__ https://github.com/fbcotter/scatnet_learn
__ https://arxiv.org/abs/1903.03137
__ https://arxiv.org/abs/1203.1513 

We will release a paper soon describing the implementation of our DTCWT
ScatterNet but the main differences between it and the original, Morlet-based
ScatterNet are summarised here:

+------------------+------------------------+--------------+--------------------+--------------------+
| Package          | Backend                | Orientations | Boundary Extension | Backprop supported |
+------------------+------------------------+--------------+--------------------+--------------------+
| `KyMatIO`__      | FFT-based              | Flexible     | Periodic           | Yes*               |
+------------------+------------------------+--------------+--------------------+--------------------+
| Pytorch Wavelets | Separable Filter Banks | 6            | Flexible           | Yes                |
+------------------+------------------------+--------------+--------------------+--------------------+

\* Supported but with very large memory usage

__ https://github.com/kymatio/kymatio

For a input of size :code:`(128, 3, 256, 256)`, the execution times measured on
a machine with a GTX1080 and 14 Intel Xeon E5-2660 CPU cores were (averaged over
5 runs):

+------------------+-------------+-------------+-------------+-------------+
| Package          | CPU Fwd (s) | CPU Bwd (s) | GPU Fwd (s) | GPU Bwd (s) |
+------------------+-------------+-------------+-------------+-------------+
| KyMatIO          | 95          | 130         | 1.44        | 2.51        |
+------------------+-------------+-------------+-------------+-------------+
| Pytorch Wavelets | 2.8         | 3.2         | 0.10        | 0.16        |
+------------------+-------------+-------------+-------------+-------------+

