from __future__ import absolute_import

import torch


class ComplexTensor(object):
    """ A wrapper to handle complex tensor as a pair of real and imaginary
    numbers.
    """
    def __init__(self, val):
        # Work out what the form of val is. Is it a pair of real and imaginary?
        # Or a single complex/real number
        if isinstance(val, ComplexTensor):
            return val
        else:
            assert len(val) == 2
            self._real = val[0]
            self._imag = val[1]
        #  super().__init__(op=self.complex.op, value_index=0, dtype=tf.complex64)

        self._norm = None
        self._norm2 = None
        self._phase = None

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    @property
    def norm(self):
        if self._norm is None:
            self._norm = torch.sqrt(self.norm2)
        return self._norm

    @property
    def norm2(self):
        """ Returns the norm squared
        """
        if self._norm2 is None:
            self._norm2 = self.real**2 + self.imag**2
        return self._norm2

    @property
    def shape(self):
        return self._real.shape

    @property
    def dtype(self):
        return 'torch.Complex64'

    def dim(self):
        return self._real.dim()

    #  @property
    #  def phase(self):
        #  if self._phase is None:
            #  self._phase = tf.angle(self.complex)
        #  return self._phase

    def apply_func(self, f):
        """ Applies the functions independently on real and imaginary components
        then returns a complex tensor instance.
        """
        return ComplexTensor([f(self.real), f(self.imag)])

    def __add__(self, other):
        if type(other) == torch.Tensor:
            return ComplexTensor((self.real+other, self.imag))
        elif hasattr(other, 'real'):
            return ComplexTensor((self.real+other.real, self.imag+other.imag))
        else:
            raise ValueError("Don't know how to add other to x")

    def __radd__(self, other):
        if type(other) == torch.Tensor:
            return ComplexTensor((self.real+other, self.imag))
        elif hasattr(other, 'real'):
            return ComplexTensor((self.real+other.real, self.imag+other.imag))
        else:
            raise ValueError("Don't know how to add other to x")

    def __sub__(self, other):
        if type(other) == torch.Tensor:
            return ComplexTensor((self.real-other, self.imag))
        elif hasattr(other, 'real'):
            return ComplexTensor((self.real-other.real, self.imag-other.imag))
        else:
            raise ValueError("Don't know how to subtract other from x")

    def __rsub__(self, other):
        if type(other) == torch.Tensor:
            return ComplexTensor((other-self.real, -self.imag))
        elif hasattr(other, 'real'):
            return ComplexTensor((other.real-self.real, other.imag-self.imag))
        else:
            raise ValueError("Don't know how to subtract x from other")

    def __mul__(self, other):
        if type(other) == torch.Tensor:
            return ComplexTensor([self.real*other, self.imag*other])
        elif hasattr(other, 'real'):
            return ComplexTensor([
                self.real*other.real - self.imag*other.imag,
                self.real*other.imag + self.imag*other.real])
        else:
            raise ValueError("Don't know how to multiply other with x")

    def __rmul__(self, other):
        # Complex multiplication is commutative
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ComplexTensor):
            raise NotImplementedError
        else:
            return ComplexTensor([self.real/other, self.imag/other])

    def __repr__(self):
        return "<ComplexTensor shape={}>".format(self.shape)
