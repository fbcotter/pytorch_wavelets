Notes on Speed
==============

Under tests/, the `profile_xfms`
script tests the speed of several layers of the DTCWT for working on a moderately sized input 
:math:`X \in \mathbb{R}^{10 \times 10 \times 128 \times 128}`.  As a reference, an 11 by 11 
convolution takes 2.53ms for a tensor of this size. 

A single layer DTCWT using the 'near_sym_a' filters (lengths 5 and 7) has 6 convolutional calls. I timed them at 238us
each for a total of 1.43ms. Unfortunately, there is also a bit of overhead in calculating the DTCWT, and not all non
convolutional operations are free. In addition to the 6 convolutions, there were:

- 6 move ops @ 119us = 714us
- 10 pointwise add ops @ 122us = 465us
- 12 copy ops @ 35us = 381us
- 6 different add ops @ 38us = 232us
- 6 subtraction ops @ 37us = 220us
- 3 constant division ops @ 57us = 173us
- 6 more move ops @ 28us = 171us

Making the overheads 2.3ms, and 3.7ms total time.

For a two layer DTCWT, there are now 12 convolutional ops. The second layer kernels are slightly larger (10 taps each)
so although they act over 1/4 the sample size, they take up an extra 1.1ms (2.5ms total for the 12 convs). The overhead
for non convolution operations is 4.4ms, making 6.9ms. Roughly 3 times a long as an 11 by 11 convolution.

There is an option to not calculate the highpass coefficients for the first scale, as these often have limited useful
information (see the `skip_hps` option). For a two scale transform, this takes the convolution run time down to 1.13ms
and the overhead down to 2.49ms, totaling 3.6ms, or roughly the same time as the 1 layer transform.

A single layer inverse transform takes: 1.43ms (conv) + 2.7ms (overhead) totaling 4.1ms, slightly longer than the 3.7ms
for the forward transform.

A two layer inverse transform takes: 2.24 (conv) + 5.9 (overhead) totaling 8.1ms, again slightly longer than the 6.9ms
for the forward transform.

A single layer end to end transform takes 2.86ms (conv) + 5.8ms (overhead) = 8.6ms :math:`\approx` 3.7 (forward) + 4.1 (inverse).

Similarly, a two layer end to end transform takes 4.4ms (conv) + 10.4ms (overhead) = 14.8ms :math:`\approx` 6.9 (forward) + 8.1 
(inverse).

If we use the `near_sym_b` filters for layer 1 (13 and 19 taps), the overhead doesn't increase, but the time taken to do
each convolution unsurprisingly triples to 600us each (up from 200us for `near_sym_a`). 

