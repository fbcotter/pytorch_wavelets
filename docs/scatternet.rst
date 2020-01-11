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

The cost of using these quicker spatially implemented wavelets is in the approximate shift invariance that they provide.
The original ScatterNet paper introduced 3 main desirable properties of the ScatterNet:

1. Invariant to additive noise
2. Invariant to small shifts
3. Invariant to small deformations

We test these 3 properties and compare the DTCWT implementation to the Morlet based one. The experiment code can be
found on the github for this repo under `tests/Measure of Stability.ipynb`. We take 1000 samples (:math:`x`) from Tiny Imagenet, 
apply these transformations (:math:`y = F(x)`) and measure the average distance between the scattered outputs 
:math:`\frac{1}{N}||Sx - Sy||^2` and compare it to the distance of the inputs :math:`\frac{1}{N}||x-y||^2`. The results were:

+---------------------------+-------------------+----------------------------+----------------------------+
| Test                      | :math:`||x-y||^2` | Morlet :math:`||Sx-Sy||^2` | DTCWT :math:`||Sx-Sy||^2`  |
+---------------------------+-------------------+----------------------------+----------------------------+
| Additive Noise            | 0.49              | 0.003                      | 0.03                       |
+---------------------------+-------------------+----------------------------+----------------------------+
| Shifts of the Input       | 0.74              | 0.004                      | 0.009                      |
+---------------------------+-------------------+----------------------------+----------------------------+
| Deformations of the Input | 0.68              | 0.004                      | 0.010                      |
+---------------------------+-------------------+----------------------------+----------------------------+

These numbers show that the Morlet implementation is better at achieving the desired properties, but the DTCWT
scatternet performs comparatively well. Our experiments have shown that the degradation in performance on these
3 properties has little effect when ScatterNets are used as part of a deeper system (e.g. before a CNN there was no
noticeable change in classification accuracy), but when used before an SVM, the cost was slightly more noticeable
(1-2% drop in classification accuracy).
