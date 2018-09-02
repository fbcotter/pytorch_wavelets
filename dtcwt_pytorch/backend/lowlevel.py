from __future__ import absolute_import

try:
    import cupy
    _HAVE_CUPY = True
    @cupy.util.memoize(for_each_device=True)
    def load_kernel(kernel_name, code, **kwargs):
        code = Template(code).substitute(**kwargs)
        kernel_code = cupy.cuda.compile_with_cache(code)
        return kernel_code.get_function(kernel_name)
except ImportError:
    _HAVE_CUPY = False

import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from dtcwt_pytorch.utils import reflect
from string import Template
from collections import namedtuple
import pkg_resources
Stream = namedtuple('Stream', ['ptr'])

CUDA_NUM_THREADS = 1024
cuda_source = pkg_resources.resource_string(__name__, 'filters.cu')
cuda_source = cuda_source.decode('utf-8')


def as_column_vector(v):
    """Return *v* as a column vector with shape (N,1).

    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v


def _as_row_vector(v):
    """Return *v* as a row vector with shape (1, N).
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v
    else:
        return v.T


def _as_row_tensor(h):
    if isinstance(h, torch.Tensor):
        h = torch.reshape(h, [1, -1])
    else:
        h = as_column_vector(h).T
        h = torch.tensor(h, dtype=torch.float32)
    return h


def _as_col_vector(v):
    """Return *v* as a column vector with shape (N,1).
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v


def _as_col_tensor(h):
    if isinstance(h, torch.Tensor):
        h = torch.reshape(h, [-1, 1])
    else:
        h = as_column_vector(h)
        h = torch.tensor(h, dtype=torch.float32)
    return h


def prep_filt(h, c, transpose=False):
    """ Prepares an array to be of the correct format for pytorch.
    Can also specify whether to make it a row filter (set tranpose=True)"""
    h = _as_col_vector(h)[::-1]
    h = np.reshape(h, [1, 1, *h.shape])
    h = np.repeat(h, repeats=c, axis=0)
    if transpose:
        h = h.transpose((0,1,3,2))
    h = np.copy(h)
    return torch.tensor(h, dtype=torch.float32)


def colfilter(X, h):
    ch, r = X.shape[1:3]
    m = h.shape[2] // 2
    xe = reflect(np.arange(-m, r+m, dtype='int32'), -0.5, r-0.5)
    return F.conv2d(X[:,:,xe], h, groups=ch)


def rowfilter(X, h):
    ch, _, c = X.shape[1:]
    m = h.shape[2] // 2
    xe = reflect(np.arange(-m, c+m, dtype='int32'), -0.5, c-0.5)
    h = h.transpose(2,3).contiguous()
    return F.conv2d(X[:,:,:,xe], h, groups=ch)


class RowFilter(Function):
    def __init__(self, weight, klow=None, khigh=None):
        if not _HAVE_CUPY:
            raise ValueError("Need cupy installed to use this function")
        super(RowFilter, self).__init__()
        self.weight = weight
        if klow is None:
            klow = -np.floor((weight.shape[0] - 1) / 2)
            khigh = np.ceil((weight.shape[0] - 1) / 2)
        assert abs(klow) == khigh, "can only do odd filters for the moment"
        self.klow = klow
        self.khigh = khigh
        assert abs(klow) == khigh
        self.f = load_kernel('rowfilter', cuda_source)
        self.fbwd = load_kernel('rowfilter_bwd', cuda_source)

    #  @staticmethod
    def forward(ctx, input):
        assert input.dim() == 4 and input.is_cuda and ctx.weight.is_cuda
        n, ch, h, w = input.shape
        ctx.in_shape = (n, ch, h, w)
        pad_end = 0
        output = torch.zeros((n, ch, h, w + pad_end),
                             dtype=torch.float32,
                             requires_grad=input.requires_grad).cuda()

        with torch.cuda.device_of(input):
            ctx.f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[output.data_ptr(), input.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w+pad_end), np.int32(w),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(1)],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return output

    #  @staticmethod
    def backward(ctx, grad_out):
        in_shape = ctx.in_shape
        n, ch, h, w = grad_out.shape
        grad_input = torch.zeros(in_shape, dtype=torch.float32).cuda()

        with torch.cuda.device_of(grad_out):
            ctx.f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[grad_input.data_ptr(), grad_out.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w), np.int32(w),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(1)],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input


class ColFilter(Function):
    def __init__(self, weight, klow=None, khigh=None):
        if not _HAVE_CUPY:
            raise ValueError("Need cupy installed to use this function")
        super(ColFilter, self).__init__()
        self.weight = weight
        if klow is None:
            klow = -np.floor((weight.shape[0] - 1) / 2)
            khigh = np.ceil((weight.shape[0] - 1) / 2)
        assert abs(klow) == khigh, "can only do odd filters for the moment"
        self.klow = klow
        self.khigh = khigh
        self.f = load_kernel('colfilter', cuda_source)
        self.fbwd = load_kernel('colfilter_bwd', cuda_source)

    #  @staticmethod
    def forward(ctx, input):
        assert input.dim() == 4 and input.is_cuda and ctx.weight.is_cuda
        n, ch, h, w = input.shape
        ctx.in_shape = (n, ch, h, w)
        pad_end = 0
        output = torch.zeros((n, ch, h + pad_end, w),
                             dtype=torch.float32,
                             requires_grad=input.requires_grad).cuda()

        with torch.cuda.device_of(input):
            ctx.f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[output.data_ptr(), input.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w), np.int32(h+pad_end),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(1)],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return output

    #  @staticmethod
    def backward(ctx, grad_out):
        in_shape = ctx.in_shape
        n, ch, h, w = grad_out.shape
        grad_input = torch.zeros(in_shape, dtype=torch.float32).cuda()

        with torch.cuda.device_of(grad_out):
            ctx.f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(128,1,1),
                  args=[grad_input.data_ptr(), grad_out.data_ptr(),
                        ctx.weight.data_ptr(), np.int32(n), np.int32(ch),
                        np.int32(h), np.int32(w), np.int32(h),
                        np.int32(ctx.klow), np.int32(ctx.khigh), np.int32(1)],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input


def coldfilt(X, ha, hb, highpass=False):
    batch, ch, r, c = X.shape
    r2 = r // 2
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4\n' +
                         'X was {}'.format(X.shape))
    m = ha.shape[2]
    xe = reflect(np.arange(-m, r+m, dtype='int32'), -0.5, r-0.5)
    t1 = xe[2:r + 2 * m - 2:2]
    t2 = xe[3:r + 2 * m - 1:2]
    if highpass:
        hb, ha = ha, hb
        t1, t2 = t2, t1
    Y1 = F.conv2d(X[:,:,t1], ha, stride=(2,1), groups=ch)
    Y2 = F.conv2d(X[:,:,t2], hb, stride=(2,1), groups=ch)

    # Stack a_rows and b_rows (both of shape [Batch, ch, r/4, c]) along the
    # third dimension to make a tensor of shape [Batch, ch, r/4, 2, c].
    Y = torch.stack((Y1, Y2), dim=3)

    # Reshape result to be shape [Batch, ch, r/2, c]. This reshaping
    # interleaves the columns
    Y = Y.view(batch, ch, r2, c)

    return Y


def rowdfilt(X, ha, hb, highpass=False):
    batch, ch, r, c = X.shape
    c2 = c // 2
    if c % 4 != 0:
        raise ValueError('No. of cols in X must be a multiple of 4\n' +
                         'X was {}'.format(X.shape))
    m = ha.shape[2]
    xe = reflect(np.arange(-m, c+m, dtype='int32'), -0.5, c-0.5)
    t1 = xe[2:c + 2 * m - 2:2]
    t2 = xe[3:c + 2 * m - 1:2]
    if highpass:
        hb, ha = ha, hb
        t1, t2 = t2, t1

    ha = ha.transpose(2,3).contiguous()
    hb = hb.transpose(2,3).contiguous()
    Y1 = F.conv2d(X[:,:,:,t1], ha, stride=(1,2), groups=ch)
    Y2 = F.conv2d(X[:,:,:,t2], hb, stride=(1,2), groups=ch)

    # Stack a_rows and b_rows (both of shape [Batch, ch, r, c/4]) along the
    # fourth dimension to make a tensor of shape [Batch, ch, r, c/4, 2].
    Y = torch.stack((Y1, Y2), dim=4)

    # Reshape result to be shape [Batch, ch, r, c/2]. This reshaping
    # interleaves the columns
    Y = Y.view(batch, ch, r, c2)

    return Y


def colifilt(X, ha, hb, highpass=False):
        m = ha.shape[2]
        m2 = m // 2
        hao = ha[:,:,1::2]
        hae = ha[:,:,::2]
        hbo = hb[:,:,1::2]
        hbe = hb[:,:,::2]
        batch, ch, r, c = X.shape
        if r % 2 != 0:
            raise ValueError('No. of rows in X must be a multiple of 2.\n' +
                             'X was {}'.format(X.shape))
        xe = reflect(np.arange(-m2, r+m2, dtype=np.int), -0.5, r-0.5)
        if m2 % 2 == 0:
            h1 = hae
            h2 = hbe
            h3 = hao
            h4 = hbo
            t1 = xe[:-2:2]
            t2 = xe[1:-2:2]
            t3 = xe[2::2]
            t4 = xe[3::2]
        else:
            h1 = hao
            h2 = hbo
            h3 = hae
            h4 = hbe
            t1 = xe[1:-1:2]
            t2 = xe[2:-1:2]
            t3 = t1
            t4 = t2
        if highpass:
            t1, t2 = t2, t1
            t3, t4 = t4, t3
        h1 = h1.contiguous()
        h2 = h2.contiguous()
        h3 = h3.contiguous()
        h4 = h4.contiguous()

        Y1 = F.conv2d(X[:,:,t1], h1, groups=ch)
        Y2 = F.conv2d(X[:,:,t2], h2, groups=ch)
        Y3 = F.conv2d(X[:,:,t3], h3, groups=ch)
        Y4 = F.conv2d(X[:,:,t4], h4, groups=ch)
        # Stack 4 tensors of shape [batch, ch, r2, c] into one tensor
        # [batch, ch, r2, 4, c]
        Y = torch.stack((Y1, Y2, Y3, Y4), dim=3)

        # Reshape to be [batch, ch,r * 2, c]. This interleaves the rows
        Y = Y.view(batch, ch, r*2, c)
        return Y


def rowifilt(X, ha, hb, highpass=False):
        m = ha.shape[2]
        m2 = m // 2
        hao = ha[:,:,1::2]
        hae = ha[:,:,::2]
        hbo = hb[:,:,1::2]
        hbe = hb[:,:,::2]
        batch, ch, r, c = X.shape
        if c % 2 != 0:
            raise ValueError('No. of cols in X must be a multiple of 2.\n' +
                             'X was {}'.format(X.shape))
        xe = reflect(np.arange(-m2, c+m2, dtype=np.int), -0.5, c-0.5)
        if m2 % 2 == 0:
            h1 = hae
            h2 = hbe
            h3 = hao
            h4 = hbo
            t1 = xe[:-2:2]
            t2 = xe[1:-2:2]
            t3 = xe[2::2]
            t4 = xe[3::2]
        else:
            h1 = hao
            h2 = hbo
            h3 = hae
            h4 = hbe
            t1 = xe[1:-1:2]
            t2 = xe[2:-1:2]
            t3 = t1
            t4 = t2
        if highpass:
            t1, t2 = t2, t1
            t3, t4 = t4, t3

        h1 = h1.transpose(2,3).contiguous()
        h2 = h2.transpose(2,3).contiguous()
        h3 = h3.transpose(2,3).contiguous()
        h4 = h4.transpose(2,3).contiguous()
        Y1 = F.conv2d(X[:,:,:,t1], h1, groups=ch)
        Y2 = F.conv2d(X[:,:,:,t2], h2, groups=ch)
        Y3 = F.conv2d(X[:,:,:,t3], h3, groups=ch)
        Y4 = F.conv2d(X[:,:,:,t4], h4, groups=ch)
        # Stack 4 tensors of shape [batch, ch, r, c2] into one tensor
        # [batch, ch, r, c2, 4]
        Y = torch.stack((Y1, Y2, Y3, Y4), dim=4)

        # Reshape to be [batch, ch, r, c*2]. This interleaves the rows
        Y = Y.view(batch, ch, r, c*2)
        return Y


def q2c(y):
    """
    Convert from quads in y to complex numbers in z.
    """

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d
    # Combine (a,b) and (d,c) to form two complex subimages.
    y = y/np.sqrt(2)
    a, b = y[:,:, 0::2, 0::2], y[:,:, 0::2, 1::2]
    c, d = y[:,:, 1::2, 0::2], y[:,:, 1::2, 1::2]

    return torch.stack((a-d, b+c), dim=-1), torch.stack((a+d, b-c), dim=-1)


def c2q(w):
    """
    Scale by gain and convert from complex w(:,:,1:2) to real quad-numbers
    in z.

    Arrange pixels from the real and imag parts of the 2 highpasses
    into 4 separate subimages .
     A----B     Re   Im of w(:,:,1)
     |    |
     |    |
     C----D     Re   Im of w(:,:,2)

    """

    # Input has shape [batch, ch, 2, r, c, 2]
    ch, _, r, c, _ = w.shape[1:]
    w = w/np.sqrt(2)
    # shape will be:
    #   x1   x2
    #   x3   x4
    x1 = w[:,:,0,:,:,0] + w[:,:,1,:,:,0]
    x2 = w[:,:,0,:,:,1] + w[:,:,1,:,:,1]
    x3 = w[:,:,0,:,:,1] - w[:,:,1,:,:,1]
    x4 = -w[:,:,0,:,:,0] + w[:,:,1,:,:,0]

    # Stack 2 inputs of shape [batch, ch, r, c] to [batch, ch, r, 2, c]
    x_rows1 = torch.stack((x1, x3), dim=-2)
    # Reshaping interleaves the results
    x_rows1 = x_rows1.view(-1, ch, 2*r, c)
    # Do the same for the even columns
    x_rows2 = torch.stack((x2, x4), dim=-2)
    x_rows2 = x_rows2.view(-1, ch, 2*r, c)

    # Stack the two [batch, ch, 2*r, c] tensors to [batch, ch, 2*r, c, 2]
    x_cols = torch.stack((x_rows1, x_rows2), dim=-1)
    y = x_cols.view(-1, ch, 2*r, 2*c)

    return y
