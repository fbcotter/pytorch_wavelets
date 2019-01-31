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
    h = np.reshape(h, [1, 1, *h.shape])
    h = np.repeat(h, repeats=c, axis=0)
    if transpose:
        h = h.transpose((0,1,3,2))
    h = np.copy(h)
    return torch.tensor(h, dtype=torch.get_default_dtype())


def colfilter(X, h):
    if X is None or X.shape == torch.Size([0]):
        return torch.zeros(1,1,1,1, device=X.device)
    ch, r = X.shape[1:3]
    m = h.shape[2] // 2
    xe = symm_pad(r, m)
    return F.conv2d(X[:,:,xe], h.repeat(ch,1,1,1), groups=ch)


def rowfilter(X, h):
    if X is None or X.shape == torch.Size([0]):
        return torch.zeros(1,1,1,1, device=X.device)
    ch, _, c = X.shape[1:]
    m = h.shape[2] // 2
    xe = symm_pad(c, m)
    h = h.transpose(2,3).contiguous()
    return F.conv2d(X[:,:,:,xe], h.repeat(ch,1,1,1), groups=ch)


def coldfilt(X, ha, hb, highpass=False):
    if X is None or X.shape == torch.Size([0]):
        return torch.zeros(1,1,1,1, device=X.device)
    batch, ch, r, c = X.shape
    r2 = r // 2
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4\n' +
                         'X was {}'.format(X.shape))
    m = ha.shape[2]
    xe = symm_pad(r, m)
    X1 = X[:,:,xe[2::2]]
    X2 = X[:,:,xe[3::2]]
    h = torch.cat((ha.repeat(ch, 1, 1, 1), hb.repeat(ch, 1, 1, 1)), dim=0)
    Y = F.conv2d(torch.cat((X1, X2), dim=1), h, stride=(2,1), groups=ch*2)

    # Reshape result to be shape [Batch, ch, r/2, c]. This reshaping
    # interleaves the columns
    if highpass:
        Y = torch.stack((Y[:,ch:], Y[:,:ch]), dim=-2).view(batch, ch, r2, c)
    else:
        Y = torch.stack((Y[:,:ch], Y[:,ch:]), dim=-2).view(batch, ch, r2, c)

    return Y


def rowdfilt(X, ha, hb, highpass=False):
    if X is None or X.shape == torch.Size([0]):
        return torch.zeros(1,1,1,1, device=X.device)
    batch, ch, r, c = X.shape
    c2 = c // 2
    if c % 4 != 0:
        raise ValueError('No. of cols in X must be a multiple of 4\n' +
                         'X was {}'.format(X.shape))
    m = ha.shape[2]
    xe = symm_pad(c, m)
    X1 = X[:,:,:,xe[2::2]]
    X2 = X[:,:,:,xe[3::2]]
    h = torch.cat((ha.reshape(1,1,1,m).repeat(ch, 1, 1, 1),
                   hb.reshape(1,1,1,m).repeat(ch, 1, 1, 1)), dim=0)
    Y = F.conv2d(torch.cat((X1, X2), dim=1), h, stride=(1,2), groups=ch*2)

    # Reshape result to be shape [Batch, ch, r/2, c]. This reshaping
    # interleaves the columns
    if highpass:
        Y = torch.stack((Y[:,ch:], Y[:,:ch]), dim=-1).view(batch, ch, r, c2)
    else:
        Y = torch.stack((Y[:,:ch], Y[:,ch:]), dim=-1).view(batch, ch, r, c2)

    return Y


def colifilt(X, ha, hb, highpass=False):
    if X is None or X.shape == torch.Size([0]):
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
        X1 = X[:,:,xe[:-2:2]]
        X2 = X[:,:,xe[1:-2:2]]
        X3 = X[:,:,xe[2::2]]
        X4 = X[:,:,xe[3::2]]
    else:
        h1 = hao
        h2 = hbo
        h3 = hae
        h4 = hbe
        X1 = X[:,:,xe[1:-1:2]]
        X2 = X[:,:,xe[2:-1:2]]
        X3 = X[:,:,xe[1:-1:2]]
        X4 = X[:,:,xe[2:-1:2]]
    if highpass:
        X2, X1 = X1, X2
        X4, X3 = X3, X4
    h = torch.cat((h1.repeat(ch, 1, 1, 1), h2.repeat(ch, 1, 1, 1),
                   h3.repeat(ch, 1, 1, 1), h4.repeat(ch, 1, 1, 1)), dim=0)
    X = torch.cat((X1, X2, X3, X4), dim=1)

    Y = F.conv2d(X, h, groups=4*ch)
    # Stack 4 tensors of shape [batch, ch, r2, c] into one tensor
    # [batch, ch, r2, 4, c]
    Y = torch.stack([Y[:,:ch], Y[:,ch:2*ch], Y[:,2*ch:3*ch], Y[:,3*ch:]],
                    dim=3).view(batch, ch, r*2, c)

    return Y


def rowifilt(X, ha, hb, highpass=False):
    if X is None or X.shape == torch.Size([0]):
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
        X1 = X[:,:,:,xe[:-2:2]]
        X2 = X[:,:,:,xe[1:-2:2]]
        X3 = X[:,:,:,xe[2::2]]
        X4 = X[:,:,:,xe[3::2]]
    else:
        h1 = hao
        h2 = hbo
        h3 = hae
        h4 = hbe
        X1 = X[:,:,:,xe[1:-1:2]]
        X2 = X[:,:,:,xe[2:-1:2]]
        X3 = X[:,:,:,xe[1:-1:2]]
        X4 = X[:,:,:,xe[2:-1:2]]
    if highpass:
        X2, X1 = X1, X2
        X4, X3 = X3, X4
    h = torch.cat((h1.repeat(ch, 1, 1, 1), h2.repeat(ch, 1, 1, 1),
                   h3.repeat(ch, 1, 1, 1), h4.repeat(ch, 1, 1, 1)),
                  dim=0).reshape(4*ch, 1, 1, m2)
    X = torch.cat((X1, X2, X3, X4), dim=1)

    Y = F.conv2d(X, h, groups=4*ch)
    # Stack 4 tensors of shape [batch, ch, r2, c] into one tensor
    # [batch, ch, r2, 4, c]
    Y = torch.stack([Y[:,:ch], Y[:,ch:2*ch], Y[:,2*ch:3*ch], Y[:,3*ch:]],
                    dim=4).view(batch, ch, r, c*2)
    return Y


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

    return torch.stack((a-d, b+c), dim=dim), torch.stack((a+d, b-c), dim=dim)


def c2q(w1, w2, dim=-1):
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
    # shape will be:
    #   x1   x2
    #   x3   x4
    if w1.shape[dim] != 2:
        raise ValueError("Wrong dimension specified for c2q. Need to specify "
                         "the dimension with real and imaginary components")

    r = torch.tensor(0, device=w1.device)
    i = torch.tensor(1, device=w1.device)
    x1 = torch.index_select(w1, dim, r) + torch.index_select(w2, dim, r)
    x2 = torch.index_select(w1, dim, i) + torch.index_select(w2, dim, i)
    x3 = torch.index_select(w1, dim, i) - torch.index_select(w2, dim, i)
    x4 = -torch.index_select(w1, dim, r) + torch.index_select(w2, dim, r)
    x1 = x1.squeeze(dim)
    x2 = x2.squeeze(dim)
    x3 = x3.squeeze(dim)
    x4 = x4.squeeze(dim)

    # Get the shape of the tensor excluding the real/imagniary part
    s = list(w1.shape)
    del s[dim]
    b, ch, r, c = s

    # Stack 2 inputs of shape [batch, ch, r, c] to [batch, ch, r, 2, c]
    x_rows1 = torch.stack((x1, x3), dim=-2)
    # Reshaping interleaves the results
    x_rows1 = x_rows1.view(-1, ch, 2*r, c)
    # Do the same for the even columns
    x_rows2 = torch.stack((x2, x4), dim=-2)
    x_rows2 = x_rows2.view(-1, ch, 2*r, c)

    # Stack the two [batch, ch, 2*r, c] tensors to [batch, ch, 2*r, c, 2]
    x_cols = torch.stack((x_rows1, x_rows2), dim=-1)
    y = x_cols.view(-1, ch, 2*r, 2*c)/np.sqrt(2)

    return y
