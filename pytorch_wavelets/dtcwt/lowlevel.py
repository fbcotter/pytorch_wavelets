from __future__ import absolute_import

import torch
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets.utils import symm_pad_1d as symm_pad


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
        h = torch.tensor(h, dtype=torch.get_default_dtype())
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
        h = torch.tensor(h, dtype=torch.get_default_dtype())
    return h


def prep_filt(h, c, transpose=False):
    """ Prepares an array to be of the correct format for pytorch.
    Can also specify whether to make it a row filter (set tranpose=True)"""
    h = _as_col_vector(h)[::-1]
    h = h[None, None, :]
    h = np.repeat(h, repeats=c, axis=0)
    if transpose:
        h = h.transpose((0,1,3,2))
    h = np.copy(h)
    return torch.tensor(h, dtype=torch.get_default_dtype())


def colfilter(X, h, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1,1, device=X.device)
    b, ch, row, col = X.shape
    m = h.shape[2] // 2
    if mode == 'symmetric':
        xe = symm_pad(row, m)
        X = F.conv2d(X[:,:,xe], h.repeat(ch,1,1,1), groups=ch)
    else:
        X = F.conv2d(X, h.repeat(ch, 1, 1, 1), groups=ch, padding=(m, 0))
    return X


def rowfilter(X, h, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1,1, device=X.device)
    b, ch, row, col = X.shape
    m = h.shape[2] // 2
    h = h.transpose(2,3).contiguous()
    if mode == 'symmetric':
        xe = symm_pad(col, m)
        X = F.conv2d(X[:,:,:,xe], h.repeat(ch,1,1,1), groups=ch)
    else:
        X = F.conv2d(X, h.repeat(ch,1,1,1), groups=ch, padding=(0, m))
    return X


def coldfilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1,1, device=X.device)
    batch, ch, r, c = X.shape
    r2 = r // 2
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4\n' +
                         'X was {}'.format(X.shape))

    if mode == 'symmetric':
        m = ha.shape[2]
        xe = symm_pad(r, m)
        X = torch.cat((X[:,:,xe[2::2]], X[:,:,xe[3::2]]), dim=1)
        h = torch.cat((ha.repeat(ch, 1, 1, 1), hb.repeat(ch, 1, 1, 1)), dim=0)
        X = F.conv2d(X, h, stride=(2,1), groups=ch*2)
    else:
        raise NotImplementedError()

    # Reshape result to be shape [Batch, ch, r/2, c]. This reshaping
    # interleaves the columns
    if highpass:
        X = torch.stack((X[:, ch:], X[:, :ch]), dim=-2).view(batch, ch, r2, c)
    else:
        X = torch.stack((X[:, :ch], X[:, ch:]), dim=-2).view(batch, ch, r2, c)

    return X


def rowdfilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1,1, device=X.device)
    batch, ch, r, c = X.shape
    c2 = c // 2
    if c % 4 != 0:
        raise ValueError('No. of cols in X must be a multiple of 4\n' +
                         'X was {}'.format(X.shape))

    if mode == 'symmetric':
        m = ha.shape[2]
        xe = symm_pad(c, m)
        X = torch.cat((X[:,:,:,xe[2::2]], X[:,:,:,xe[3::2]]), dim=1)
        h = torch.cat((ha.reshape(1,1,1,m).repeat(ch, 1, 1, 1),
                       hb.reshape(1,1,1,m).repeat(ch, 1, 1, 1)), dim=0)
        X = F.conv2d(X, h, stride=(1,2), groups=ch*2)
    else:
        raise NotImplementedError()

    # Reshape result to be shape [Batch, ch, r/2, c]. This reshaping
    # interleaves the columns
    if highpass:
        Y = torch.stack((X[:, ch:], X[:, :ch]), dim=-1).view(batch, ch, r, c2)
    else:
        Y = torch.stack((X[:, :ch], X[:, ch:]), dim=-1).view(batch, ch, r, c2)

    return Y


def colifilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1,1, device=X.device)
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
    xe = symm_pad(r, m2)

    if m2 % 2 == 0:
        h1 = hae
        h2 = hbe
        h3 = hao
        h4 = hbo
        if highpass:
            X = torch.cat((X[:,:,xe[1:-2:2]], X[:,:,xe[:-2:2]], X[:,:,xe[3::2]], X[:,:,xe[2::2]]), dim=1)
        else:
            X = torch.cat((X[:,:,xe[:-2:2]], X[:,:,xe[1:-2:2]], X[:,:,xe[2::2]], X[:,:,xe[3::2]]), dim=1)
    else:
        h1 = hao
        h2 = hbo
        h3 = hae
        h4 = hbe
        if highpass:
            X = torch.cat((X[:,:,xe[2:-1:2]], X[:,:,xe[1:-1:2]], X[:,:,xe[2:-1:2]], X[:,:,xe[1:-1:2]]), dim=1)
        else:
            X = torch.cat((X[:,:,xe[1:-1:2]], X[:,:,xe[2:-1:2]], X[:,:,xe[1:-1:2]], X[:,:,xe[2:-1:2]]), dim=1)
    h = torch.cat((h1.repeat(ch, 1, 1, 1), h2.repeat(ch, 1, 1, 1),
                   h3.repeat(ch, 1, 1, 1), h4.repeat(ch, 1, 1, 1)), dim=0)

    X = F.conv2d(X, h, groups=4*ch)
    # Stack 4 tensors of shape [batch, ch, r2, c] into one tensor
    # [batch, ch, r2, 4, c]
    X = torch.stack([X[:,:ch], X[:,ch:2*ch], X[:,2*ch:3*ch], X[:,3*ch:]], dim=3).view(batch, ch, r*2, c)

    return X


def rowifilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1,1, device=X.device)
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
    xe = symm_pad(c, m2)

    if m2 % 2 == 0:
        h1 = hae
        h2 = hbe
        h3 = hao
        h4 = hbo
        if highpass:
            X = torch.cat((X[:,:,:,xe[1:-2:2]], X[:,:,:,xe[:-2:2]], X[:,:,:,xe[3::2]], X[:,:,:,xe[2::2]]), dim=1)
        else:
            X = torch.cat((X[:,:,:,xe[:-2:2]], X[:,:,:,xe[1:-2:2]], X[:,:,:,xe[2::2]], X[:,:,:,xe[3::2]]), dim=1)
    else:
        h1 = hao
        h2 = hbo
        h3 = hae
        h4 = hbe
        if highpass:
            X = torch.cat((X[:,:,:,xe[2:-1:2]], X[:,:,:,xe[1:-1:2]], X[:,:,:,xe[2:-1:2]], X[:,:,:,xe[1:-1:2]]), dim=1)
        else:
            X = torch.cat((X[:,:,:,xe[1:-1:2]], X[:,:,:,xe[2:-1:2]], X[:,:,:,xe[1:-1:2]], X[:,:,:,xe[2:-1:2]]), dim=1)
    h = torch.cat((h1.repeat(ch, 1, 1, 1), h2.repeat(ch, 1, 1, 1),
                   h3.repeat(ch, 1, 1, 1), h4.repeat(ch, 1, 1, 1)),
                  dim=0).reshape(4*ch, 1, 1, m2)

    X = F.conv2d(X, h, groups=4*ch)
    # Stack 4 tensors of shape [batch, ch, r2, c] into one tensor
    # [batch, ch, r2, 4, c]
    X = torch.stack([X[:,:ch], X[:,ch:2*ch], X[:,2*ch:3*ch], X[:,3*ch:]], dim=4).view(batch, ch, r, c*2)
    return X


#  def q2c(y, dim=-1):
def q2c(y, dim=-1):
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

    #  return torch.stack((a-d, b+c), dim=dim), torch.stack((a+d, b-c), dim=dim)
    return ((a-d, b+c), (a+d, b-c))


def c2q(w1, w2):
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
    w1r, w1i = w1
    w2r, w2i = w2

    x1 = w1r + w2r
    x2 = w1i + w2i
    x3 = w1i - w2i
    x4 = -w1r + w2r

    # Get the shape of the tensor excluding the real/imagniary part
    b, ch, r, c = w1r.shape

    # Create new empty tensor and fill it
    y = w1r.new_zeros((b, ch, r*2, c*2), requires_grad=w1r.requires_grad)
    y[:, :, ::2,::2] = x1
    y[:, :, ::2, 1::2] = x2
    y[:, :, 1::2, ::2] = x3
    y[:, :, 1::2, 1::2] = x4
    y /= np.sqrt(2)

    return y
