import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from dtcwt_pytorch.utils import reflect
from string import Template
from collections import namedtuple
import pkg_resources
Stream = namedtuple('Stream', ['ptr'])


def symm_pad(l, m):
    """ Creates indices for symmetric padding. Works for 1-D.

    Inptus:
        l (int): size of input
        m (int): size of filter
    """
    if (m // 2) % 2 == 1:
        xe = reflect(np.arange(-m+1, l+m, dtype='int32'), -0.5, l-0.5)
    else:
        xe = reflect(np.arange(-m, l+m-1, dtype='int32'), -0.5, l-0.5)
    return xe

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


def coldfilt(X, h, lev_below=1):
    ch, r = X.shape[1:3]
    m = h.shape[2] // 2
    xe = symm_pad(r, m*lev_below)
    return F.conv2d(X[:,:,xe], h, groups=ch, stride=(2,1))


def rowdfilt(X, h, lev_below=1):
    ch, _, c = X.shape[1:]
    m = h.shape[2] // 2
    xe = symm_pad(c, m*lev_below)
    h = h.transpose(2,3).contiguous()
    return F.conv2d(X[:,:,:,xe], h, groups=ch, stride=(1,2))


def colifilt(X, h, lev_below=1):
    ch, r = X.shape[1:3]
    m = h.shape[2] // 2
    xe = symm_pad(r, m*lev_below)
    return F.conv_transpose2d(X[:,:,xe], h, groups=ch, stride=(2,1))


def rowifilt(X, h, lev_below=1):
    ch, _, c = X.shape[1:]
    m = h.shape[2] // 2
    xe = symm_pad(c, m*lev_below)
    return F.conv_transpose2d(X[:,:,:,xe], h, groups=ch, stride=(1,2))
