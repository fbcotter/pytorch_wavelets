import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np
from pytorch_wavelets.dtcwt.coeffs import qshift as _qshift


def roll(x: torch.Tensor, shift: int, dim: int = -1,
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


def afb2d_a(x, af, d=-1):
    """ 2D analysis filter bank (along one dimension only)

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
    N = x.shape[d]
    L = af.shape[d] // 2
    if d == 2 or d == -2:
        pad = (af.shape[d]-1, 0)
        stride = (2, 1)
        x = roll(x, shift=-L, dim=2)
    else:
        pad = (0, af.shape[d]-1)
        stride = (1, 2)
        x = roll(x, shift=-L, dim=3)

    # Calculate the high and lowpass
    lohi = F.conv2d(
        x, torch.cat([af,]*C, dim=0), padding=pad, stride=stride, groups=C)
    if d == 2 or d == -2:
        lohi[:,:,:L] = lohi[:,:,:L] + lohi[:,:,N//2:N//2+L]
        lohi = lohi[:,:,:N//2]
    else:
        lohi[:,:,:,:L] = lohi[:,:,:,:L] + lohi[:,:,:,N//2:N//2+L]
        lohi = lohi[:,:,:,:N//2]

    return lohi


def afb2d(x, af_col, af_row):
    """ 2D seprable analysis filter bank

    Uses periodic extension """
    # filter along columns
    lohi = afb2d_a(x, af_col, 2)
    ll_lh_hl_hh = afb2d_a(lohi, af_row.transpose(3,2), 3)
    return ll_lh_hl_hh


def get_level1(name):
    """ Load the level 1 filters for tree a and tree b """
    if name == 'farras':
        h0a = np.array([[0,
                        -0.08838834764832,
                        0.08838834764832,
                        0.69587998903400,
                        0.69587998903400,
                        0.08838834764832,
                        -0.08838834764832,
                        0.01122679215254,
                        0.01122679215254,
                        0]]).T
        h1a = np.array([[0,
                        -0.01122679215254,
                        0.01122679215254,
                        0.08838834764832,
                        0.08838834764832,
                        -0.69587998903400,
                        0.69587998903400,
                        -0.08838834764832,
                        -0.08838834764832,
                        0]]).T
        h0b = np.array([[0.01122679215254,
                        0.01122679215254,
                        -0.08838834764832,
                        0.08838834764832,
                        0.69587998903400,
                        0.69587998903400,
                        0.08838834764832,
                        -0.08838834764832,
                        0,
                        0]]).T
        h1b = np.array([[0,
                        0,
                        -0.08838834764832,
                        -0.08838834764832,
                        0.69587998903400,
                        -0.69587998903400,
                        0.08838834764832,
                        0.08838834764832,
                        0.01122679215254,
                        -0.01122679215254]]).T

    return h0a, h1a, h0b, h1b


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
    h0a1, h1a1, h0b1, h1b1 = get_level1(level1)
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
