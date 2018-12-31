import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np
from pytorch_wavelets.dtcwt.coeffs import qshift as _qshift, level1 as _level1
from pytorch_wavelets.utils import reflect
import pywt


def roll(x, n, dim):
    if n < 0:
        n = x.shape[dim] + n

    if dim == 0:
        return torch.cat((x[-n:], x[:-n]), dim=0)
    elif dim == 1:
        return torch.cat((x[:,-n:], x[:,:-n]), dim=1)
    elif dim == 2 or dim == -2:
        return torch.cat((x[:,:,-n:], x[:,:,:-n]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n]), dim=3)


def mypad(x, pad, mode='constant', value=0):
    if mode == 'symmetric':
        # Check if we are doing vertical or horizontal padding
        if pad[0] != 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,:,xe]
        else:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,xe]
    else:
        return F.pad(x, pad, mode, value)


def roll2(x: torch.Tensor, shift: int, dim: int = -1,
         fill_pad: Optional[int] = None):

    def myrange(lower, upper):
        return torch.arange(lower, upper, dtype=torch.long, device=x.device)

    if 0 == shift:
        return x
    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, myrange(0, shift))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        rest = x.index_select(dim, myrange(shift, x.size(dim)))
        return torch.cat([rest, gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, myrange(shift, x.size(dim)))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        rest = x.index_select(dim, myrange(0, shift))
        return torch.cat([gap, rest], dim=dim)


def afb1d_periodic(x, h0, h1, dim=-1):
    """ 1D analysis filter bank (along one dimension only) with periodic
    extension

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        af (tensor) - analysis low and highpass filters. Should have shape
        (2, 1, h, 1) or (2, 1, 1, w).
        d (int) - dimension of filtering. d=2 is for a vertical filter (called
        column filtering but filters across the rows). d=3 is for a horizontal
        filter, (called row filtering but filters across the columns).

    Returns:
        lo, hi: lowpass and highpass subbands

    """
    C = x.shape[1]
    N = x.shape[dim]
    L = h0.shape[dim] // 2
    if dim == 2 or dim == -2:
        pad = (h0.shape[dim]-1, 0)
        stride = (2, 1)
        x = roll(x, -L, dim=2)
    else:
        pad = (0, h0.shape[dim]-1)
        stride = (1, 2)
        x = roll(x, -L, dim=3)

    # Calculate the high and lowpass
    h = torch.cat([h0, h1] * C, dim=0)
    lohi = F.conv2d(x, h, padding=pad, stride=stride, groups=C)
    if dim == 2 or dim == -2:
        lohi[:,:,:L] = lohi[:,:,:L] + lohi[:,:,N//2:N//2+L]
        lohi = lohi[:,:,:N//2]
    else:
        lohi[:,:,:,:L] = lohi[:,:,:,:L] + lohi[:,:,:,N//2:N//2+L]
        lohi = lohi[:,:,:,:N//2]

    return lohi


def sfb1d_periodic(lo, hi, g0, g1, dim=-1):
    """ 1D synthesis filter bank (along one dimension only) with periodic
    extension

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        af (tensor) - analysis low and highpass filters. Should have shape
        (2, 1, h, 1) or (2, 1, 1, w).
        d (int) - dimension of filtering. d=2 is for a vertical filter (called
        column filtering but filters across the rows). d=3 is for a horizontal
        filter, (called row filtering but filters across the columns).

    Returns:
        lo, hi: lowpass and highpass subbands

    """
    C = lo.shape[1]
    N = 2*lo.shape[dim]
    L = g0.shape[dim]
    if dim == 2 or dim == -2:
        s = (2, 1)
    else:
        s = (1, 2)

    y = F.conv_transpose2d(lo, torch.cat([g0]*C,dim=0), stride=s, groups=C) + \
        F.conv_transpose2d(hi, torch.cat([g1]*C,dim=0), stride=s, groups=C)
    #  sf = torch.cat([sf,]*C, dim=0)
    #  # Would need to interleave the lo and hi to get this to work
    #  y = F.conv_transpose2d(torch.cat((lo, hi), dim=1), sf, stride=stride,
                           #  groups=C)
    if dim == 2 or dim == -2:
        y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
        y = y[:,:,:N]
    else:
        y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
        y = y[:,:,:,:N]
    y = roll(y, 1-L//2, dim=dim)

    return y


def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        af (tensor) - analysis low and highpass filters. Should have shape
        (2, 1, h, 1) or (2, 1, 1, w).
        d (int) - dimension of filtering. d=2 is for a vertical filter (called
        column filtering but filters across the rows). d=3 is for a horizontal
        filter, (called row filtering but filters across the columns).

    Returns:
        lo, hi: lowpass and highpass subbands
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)

    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(x.shape[dim], h0.shape[dim], mode=mode)
    p = 2 * (outsize - 1) - x.shape[dim] + h0.shape[dim]
    if mode == 'zero':
        # Sadly, pytorch only allows for same padding before and after, if we
        # need to do more padding after for odd length signals, have to prepad
        if p % 2 == 1:
            pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
            x = F.pad(x, pad)
        pad = (p//2, 0) if d == 2 else (0, p//2)
        # Calculate the high and lowpass
        lohi = F.conv2d(
            x, torch.cat([h0, h1]*C, dim=0), padding=pad, stride=s, groups=C)
    elif mode == 'symmetric' or mode == 'reflect':
        pad = (0, 0, p//2, (p+1)//2) if d == 2 else (p//2, (p+1)//2, 0, 0)
        x = mypad(x, pad=pad, mode=mode)
        lohi = F.conv2d(
            x, torch.cat([h0, h1]*C, dim=0), stride=s, groups=C)

    return lohi


def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    L = g0.shape[dim]
    if mode == 'zero':
        mode == 'constant'

    if dim == 2 or dim == -2:
        s = (2, 1)
        pad = (L-2, 0)
    else:
        s = (1, 2)
        pad = (0, L-2)

    y = F.conv_transpose2d(lo, torch.cat([g0]*C,dim=0), stride=s, padding=pad, groups=C) + \
        F.conv_transpose2d(hi, torch.cat([g1]*C,dim=0), stride=s, padding=pad, groups=C)

    return y


def afb2d(x, h0_col, h1_col, h0_row, h1_row, mode='zero', split=True):
    """ 2D seprable analysis filter bank """

    if mode == 'periodic':
        lohi = afb1d_periodic(x, h0_col, h1_col, mode=mode, dim=2)
        ll_hl_lh_hh = afb1d_periodic(lohi, h0_row, h1_row, mode=mode, dim=3)
    else:
        lohi = afb1d(x, h0_col, h1_col, mode=mode, dim=2)
        ll_hl_lh_hh = afb1d(lohi, h0_row, h1_row, mode=mode, dim=3)

    if split:
        return (ll_hl_lh_hh[:, ::4], (ll_hl_lh_hh[:, 2::4],
                ll_hl_lh_hh[:, 1::4], ll_hl_lh_hh[:, 3::4]))
    else:
        return ll_hl_lh_hh


def sfb2d(lo, highs, g0_col, g1_col, g0_row, g1_row, mode='zero'):
    """ 2D separable synthesis filter bank """
    if mode == 'periodic':
        lo = sfb1d_periodic(lo, highs[0], g0_row, g1_row, dim=3)
        hi = sfb1d_periodic(highs[1], highs[2], g0_row, g1_row, dim=3)
        y = sfb1d_periodic(lo, hi, g0_col, g1_col, dim=2)
    else:
        lo = sfb1d(lo, highs[0], g0_row, g1_row, mode=mode, dim=3)
        hi = sfb1d(highs[1], highs[2], g0_row, g1_row, mode=mode, dim=3)
        y = sfb1d(lo, hi, g0_col, g1_col, mode=mode, dim=2)

    return y


def cplxdual2D(x, J, level1='farras', qshift='qshift_a'):
    """ Do a complex dtcwt

    Returns:
        lows: lowpass outputs from each of the 4 trees. Is a 2x2 list of lists
        w: bandpass outputs from each of the 4 trees. Is a list of lists, with
        shape [J][2][2][3]. Initially the 3 outputs are the lh, hl and hh from
        each of the 4 trees. After doing sums and differences though, they
        become the real and imaginary parts for the 6 orientations. In
        particular:
            first index - indexes over scales
            second index - 0 = real, 1 = imaginary
            third and fourth indices:
            1,2 = 15 degrees
            2,3 = 45 degrees
            1,1 = 75 degrees
            2,1 = 105 degrees
            1,3 = 135 degrees
            2,2 = 165 degrees
    """
    x = x/2
    dev = x.device
    # Get the filters
    h0a1, h1a1, h0b1, h1b1 = _level1(level1)
    h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
    Faf_a = np.expand_dims(np.stack((h0a1[::-1], h1a1[::-1]), axis=0), axis=1)
    Faf_b = np.expand_dims(np.stack((h0b1[::-1], h1b1[::-1]), axis=0), axis=1)
    af_a = np.expand_dims(np.stack((h0a[::-1], h1a[::-1]), axis=0), axis=1)
    af_b = np.expand_dims(np.stack((h0b[::-1], h1b[::-1]), axis=0), axis=1)
    Faf_a = torch.tensor(np.copy(Faf_a), dtype=torch.float).to(dev)
    Faf_b = torch.tensor(np.copy(Faf_b), dtype=torch.float).to(dev)
    af_a = torch.tensor(np.copy(af_a), dtype=torch.float).to(dev)
    af_b = torch.tensor(np.copy(af_b), dtype=torch.float).to(dev)

    Faf = [Faf_a, Faf_b]
    af = [af_a, af_b]

    # Do 4 fully decimated dwts
    w = [[[None for _ in range(2)] for _ in range(2)] for j in range(J)]
    lows = [[None for _ in range(2)] for _ in range(2)]
    for m in range(2):
        for n in range(2):
            # Do the first level transform with the first level filters
            coeffs = afb2d(x, Faf[m], Faf[n])
            ll, lh, hl, hh = coeffs[:,::4], coeffs[:,1::4], coeffs[:,2::4], coeffs[:,3::4]
            w[0][m][n] = [lh, hl, hh]

            # Do the second+ level transform with the second level filters
            for j in range(1,J):
                coeffs = afb2d(ll, af[m], af[n])
                ll, lh, hl, hh = coeffs[:,::4], coeffs[:,1::4], coeffs[:,2::4], coeffs[:,3::4]
                w[j][m][n] = [lh, hl, hh]
            lows[m][n] = ll

    # Convert the quads into real and imaginary parts
    for j in range(J):
        for l in range(3):
            w[j][0][0][l], w[j][1][1][l] = pm(w[j][0][0][l], w[j][1][1][l])
            w[j][0][1][l], w[j][1][0][l] = pm(w[j][0][1][l], w[j][1][0][l])

    return lows, w


def pm(a, b):
    u = (a + b)/np.sqrt(2)
    v = (a - b)/np.sqrt(2)
    return u, v
