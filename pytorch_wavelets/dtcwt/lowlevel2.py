""" This module was part of an attempt to speed up the DTCWT. The code was
ultimately slower than the original implementation, but it is a nice
reference point for doing a DTCWT directly as 4 separate DWTs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets.dwt.lowlevel import roll, mypad
import pywt
from pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse
from pytorch_wavelets.dwt.lowlevel import afb2d, sfb2d_nonsep as sfb2d
from pytorch_wavelets.dwt.lowlevel import prep_filt_afb2d, prep_filt_sfb2d_nonsep as prep_filt_sfb2d
from pytorch_wavelets.dtcwt.coeffs import level1 as _level1, qshift as _qshift, biort as _biort


class DTCWTForward2(nn.Module):
    """ DTCWT based on 4 DWTs. Still works, but the above implementation is
    faster """
    def __init__(self, biort='farras', qshift='qshift_a', J=3,
                 mode='symmetric'):
        super().__init__()
        self.biort = biort
        self.qshift = qshift
        self.J = J

        if isinstance(biort, str):
            biort = _level1(biort)
        assert len(biort) == 8
        h0a1, h0b1, _, _, h1a1, h1b1, _, _ = biort
        DWTaa1 = DWTForward(J=1, wave=(h0a1, h1a1, h0a1, h1a1), mode=mode)
        DWTab1 = DWTForward(J=1, wave=(h0a1, h1a1, h0b1, h1b1), mode=mode)
        DWTba1 = DWTForward(J=1, wave=(h0b1, h1b1, h0a1, h1a1), mode=mode)
        DWTbb1 = DWTForward(J=1, wave=(h0b1, h1b1, h0b1, h1b1), mode=mode)
        self.level1 = nn.ModuleList([DWTaa1, DWTab1, DWTba1, DWTbb1])

        if J > 1:
            if isinstance(qshift, str):
                qshift = _qshift(qshift)
            assert len(qshift) == 8
            h0a, h0b, _, _, h1a, h1b, _, _ = qshift
            DWTaa = DWTForward(J-1, (h0a, h1a, h0a, h1a), mode=mode)
            DWTab = DWTForward(J-1, (h0a, h1a, h0b, h1b), mode=mode)
            DWTba = DWTForward(J-1, (h0b, h1b, h0a, h1a), mode=mode)
            DWTbb = DWTForward(J-1, (h0b, h1b, h0b, h1b), mode=mode)
            self.level2 = nn.ModuleList([DWTaa, DWTab, DWTba, DWTbb])

    def forward(self, x):
        x = x/2
        J = self.J
        w = [[[None for _ in range(2)] for _ in range(2)] for j in range(J)]
        lows = [[None for _ in range(2)] for _ in range(2)]
        for m in range(2):
            for n in range(2):
                # Do the first level transform
                ll, (w[0][m][n],) = self.level1[m*2+n](x)
                #  w[0][m][n] = [bands[:,:,2], bands[:,:,1], bands[:,:,3]]

                # Do the second+ level transform with the second level filters
                if J > 1:
                    ll, bands = self.level2[m*2+n](ll)
                    for j in range(1,J):
                        w[j][m][n] = bands[j-1]
                lows[m][n] = ll

        # Convert the quads into real and imaginary parts
        yh = [None,] * J
        for j in range(J):
            deg75r, deg105i = pm(w[j][0][0][:,:,1], w[j][1][1][:,:,1])
            deg105r, deg75i = pm(w[j][0][1][:,:,1], w[j][1][0][:,:,1])
            deg15r, deg165i = pm(w[j][0][0][:,:,0], w[j][1][1][:,:,0])
            deg165r, deg15i = pm(w[j][0][1][:,:,0], w[j][1][0][:,:,0])
            deg135r, deg45i = pm(w[j][0][0][:,:,2], w[j][1][1][:,:,2])
            deg45r, deg135i = pm(w[j][0][1][:,:,2], w[j][1][0][:,:,2])
            w[j] = None
            yhr = torch.stack((deg15r, deg45r, deg75r,
                               deg105r, deg135r, deg165r), dim=1)
            yhi = torch.stack((deg15i, deg45i, deg75i,
                               deg105i, deg135i, deg165i), dim=1)
            yh[j] = torch.stack((yhr, yhi), dim=-1)

        return lows, yh


class DTCWTInverse2(nn.Module):
    def __init__(self, biort='farras', qshift='qshift_a',
                 mode='symmetric'):
        super().__init__()
        self.biort = biort
        self.qshift = qshift

        if isinstance(biort, str):
            biort = _level1(biort)
        assert len(biort) == 8
        _, _, g0a1, g0b1, _, _, g1a1, g1b1 = biort
        IWTaa1 = DWTInverse(wave=(g0a1, g1a1, g0a1, g1a1), mode=mode)
        IWTab1 = DWTInverse(wave=(g0a1, g1a1, g0b1, g1b1), mode=mode)
        IWTba1 = DWTInverse(wave=(g0b1, g1b1, g0a1, g1a1), mode=mode)
        IWTbb1 = DWTInverse(wave=(g0b1, g1b1, g0b1, g1b1), mode=mode)
        self.level1 = nn.ModuleList([IWTaa1, IWTab1, IWTba1, IWTbb1])

        if isinstance(qshift, str):
            qshift = _qshift(qshift)
        assert len(qshift) == 8
        _, _, g0a, g0b, _, _, g1a, g1b = qshift
        IWTaa = DWTInverse(wave=(g0a, g1a, g0a, g1a), mode=mode)
        IWTab = DWTInverse(wave=(g0a, g1a, g0b, g1b), mode=mode)
        IWTba = DWTInverse(wave=(g0b, g1b, g0a, g1a), mode=mode)
        IWTbb = DWTInverse(wave=(g0b, g1b, g0b, g1b), mode=mode)
        self.level2 = nn.ModuleList([IWTaa, IWTab, IWTba, IWTbb])

    def forward(self, x):
        # Convert the highs back to subbands
        yl, yh = x
        J = len(yh)
        #  w = [[[[None for i in range(3)] for j in range(2)]
              #  for k in range(2)] for l in range(J)]
        w = [[[[None for band in range(3)] for j in range(J)]
              for m in range(2)] for n in range(2)]
        for j in range(J):
            w[0][0][j][0], w[1][1][j][0] = pm(
                yh[j][:,2,:,:,:,0], yh[j][:,3,:,:,:,1])
            w[0][1][j][0], w[1][0][j][0] = pm(
                yh[j][:,3,:,:,:,0], yh[j][:,2,:,:,:,1])
            w[0][0][j][1], w[1][1][j][1] = pm(
                yh[j][:,0,:,:,:,0], yh[j][:,5,:,:,:,1])
            w[0][1][j][1], w[1][0][j][1] = pm(
                yh[j][:,5,:,:,:,0], yh[j][:,0,:,:,:,1])
            w[0][0][j][2], w[1][1][j][2] = pm(
                yh[j][:,1,:,:,:,0], yh[j][:,4,:,:,:,1])
            w[0][1][j][2], w[1][0][j][2] = pm(
                yh[j][:,4,:,:,:,0], yh[j][:,1,:,:,:,1])
            w[0][0][j] = torch.stack(w[0][0][j], dim=2)
            w[0][1][j] = torch.stack(w[0][1][j], dim=2)
            w[1][0][j] = torch.stack(w[1][0][j], dim=2)
            w[1][1][j] = torch.stack(w[1][1][j], dim=2)

        y = None
        for m in range(2):
            for n in range(2):
                lo = yl[m][n]
                if J > 1:
                    lo = self.level2[m*2+n]((lo, w[m][n][1:]))
                lo = self.level1[m*2+n]((lo, (w[m][n][0],)))

                # Add to the output
                if y is None:
                    y = lo
                else:
                    y = y + lo

        # Normalize
        y = y/2
        return y


def prep_filt_quad_afb2d_nonsep(
        h0a_col, h1a_col, h0a_row, h1a_row,
        h0b_col, h1b_col, h0b_row, h1b_row,
        h0c_col, h1c_col, h0c_row, h1c_row,
        h0d_col, h1d_col, h0d_row, h1d_row, device=None):
    """
    Prepares the filters to be of the right form for the afb2d_nonsep function.
    In particular, makes 2d point spread functions, and mirror images them in
    preparation to do torch.conv2d.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        filts: (4, 1, h, w) tensor ready to get the four subbands
    """
    lla = np.outer(h0a_col, h0a_row)
    lha = np.outer(h1a_col, h0a_row)
    hla = np.outer(h0a_col, h1a_row)
    hha = np.outer(h1a_col, h1a_row)
    llb = np.outer(h0b_col, h0b_row)
    lhb = np.outer(h1b_col, h0b_row)
    hlb = np.outer(h0b_col, h1b_row)
    hhb = np.outer(h1b_col, h1b_row)
    llc = np.outer(h0c_col, h0c_row)
    lhc = np.outer(h1c_col, h0c_row)
    hlc = np.outer(h0c_col, h1c_row)
    hhc = np.outer(h1c_col, h1c_row)
    lld = np.outer(h0d_col, h0d_row)
    lhd = np.outer(h1d_col, h0d_row)
    hld = np.outer(h0d_col, h1d_row)
    hhd = np.outer(h1d_col, h1d_row)
    filts = np.stack([lla[None,::-1,::-1], llb[None,::-1,::-1],
                      llc[None,::-1,::-1], lld[None,::-1,::-1],
                      lha[None,::-1,::-1], lhb[None,::-1,::-1],
                      lhc[None,::-1,::-1], lhd[None,::-1,::-1],
                      hla[None,::-1,::-1], hlb[None,::-1,::-1],
                      hlc[None,::-1,::-1], hld[None,::-1,::-1],
                      hha[None,::-1,::-1], hhb[None,::-1,::-1],
                      hhc[None,::-1,::-1], hhd[None,::-1,::-1]],
                     axis=0)
    filts = torch.tensor(filts, dtype=torch.get_default_dtype(), device=device)
    return filts


def prep_filt_quad_afb2d(h0a, h1a, h0b, h1b, device=None):
    """
    Prepares the filters to be of the right form for the quad_afb2d function.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        filts: (4, 1, h, w) tensor ready to get the four subbands
    """
    h0a_col = np.array(h0a).ravel()[::-1][None, :, None]
    h1a_col = np.array(h1a).ravel()[::-1][None, :, None]
    h0b_col = np.array(h0a).ravel()[::-1][None, :, None]
    h1b_col = np.array(h1a).ravel()[::-1][None, :, None]
    h0c_col = np.array(h0b).ravel()[::-1][None, :, None]
    h1c_col = np.array(h1b).ravel()[::-1][None, :, None]
    h0d_col = np.array(h0b).ravel()[::-1][None, :, None]
    h1d_col = np.array(h1b).ravel()[::-1][None, :, None]
    h0a_row = np.array(h0a).ravel()[::-1][None, None, :]
    h1a_row = np.array(h1a).ravel()[::-1][None, None, :]
    h0b_row = np.array(h0b).ravel()[::-1][None, None, :]
    h1b_row = np.array(h1b).ravel()[::-1][None, None, :]
    h0c_row = np.array(h0a).ravel()[::-1][None, None, :]
    h1c_row = np.array(h1a).ravel()[::-1][None, None, :]
    h0d_row = np.array(h0b).ravel()[::-1][None, None, :]
    h1d_row = np.array(h1b).ravel()[::-1][None, None, :]
    cols = np.stack((h0a_col, h1a_col,
                     h0b_col, h1b_col,
                     h0c_col, h1c_col,
                     h0d_col, h1d_col), axis=0)
    rows = np.stack((h0a_row, h1a_row,
                     h0a_row, h1a_row,
                     h0b_row, h1b_row,
                     h0b_row, h1b_row,
                     h0c_row, h1c_row,
                     h0c_row, h1c_row,
                     h0d_row, h1d_row,
                     h0d_row, h1d_row), axis=0)
    cols = torch.tensor(np.copy(cols), dtype=torch.get_default_dtype(),
                        device=device)
    rows = torch.tensor(np.copy(rows), dtype=torch.get_default_dtype(),
                        device=device)
    return cols, rows


def quad_afb2d(x, cols, rows, mode='zero', split=True, stride=2):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    x = x/2
    C = x.shape[1]
    cols = torch.cat([cols]*C, dim=0)
    rows = torch.cat([rows]*C, dim=0)

    if mode == 'per' or mode == 'periodization':
        # Do column filtering
        L = cols.shape[2]
        L2 = L // 2
        if x.shape[2] % 2 == 1:
            x = torch.cat((x, x[:,:,-1:]), dim=2)
        N2 = x.shape[2] // 2
        x = roll(x, -L2, dim=2)
        pad = (L-1, 0)
        lohi = F.conv2d(x, cols, padding=pad, stride=(stride,1), groups=C)
        lohi[:,:,:L2] = lohi[:,:,:L2] + lohi[:,:,N2:N2+L2]
        lohi = lohi[:,:,:N2]

        # Do row filtering
        L = rows.shape[3]
        L2 = L // 2
        if lohi.shape[3] % 2 == 1:
            lohi = torch.cat((lohi, lohi[:,:,:,-1:]), dim=3)
        N2 = x.shape[3] // 2
        lohi = roll(lohi, -L2, dim=3)
        pad = (0, L-1)
        w = F.conv2d(lohi, rows, padding=pad, stride=(1,stride), groups=8*C)
        w[:,:,:,:L2] = w[:,:,:,:L2] + w[:,:,:,N2:N2+L2]
        w = w[:,:,:,:N2]
    elif mode == 'zero':
        # Do column filtering
        N = x.shape[2]
        L = cols.shape[2]
        outsize = pywt.dwt_coeff_len(N, L, mode='zero')
        p = 2 * (outsize - 1) - N + L

        # Sadly, pytorch only allows for same padding before and after, if
        # we need to do more padding after for odd length signals, have to
        # prepad
        if p % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
        pad = (p//2, 0)
        # Calculate the high and lowpass
        lohi = F.conv2d(x, cols, padding=pad, stride=(stride,1), groups=C)

        # Do row filtering
        N = lohi.shape[3]
        L = rows.shape[3]
        outsize = pywt.dwt_coeff_len(N, L, mode='zero')
        p = 2 * (outsize - 1) - N + L
        if p % 2 == 1:
            lohi = F.pad(lohi, (0, 1, 0, 0))
        pad = (0, p//2)
        w = F.conv2d(lohi, rows, padding=pad, stride=(1,stride), groups=8*C)
    elif mode == 'symmetric' or mode == 'reflect':
        # Do column filtering
        N = x.shape[2]
        L = cols.shape[2]
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        x = mypad(x, pad=(0, 0, p//2, (p+1)//2), mode=mode)
        lohi = F.conv2d(x, cols, stride=(stride,1), groups=C)

        # Do row filtering
        N = lohi.shape[3]
        L = rows.shape[3]
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        lohi = mypad(lohi, pad=(p//2, (p+1)//2, 0, 0), mode=mode)
        w = F.conv2d(lohi, rows, stride=(1,stride), groups=8*C)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    y = w.view((w.shape[0], C, 4, 4, w.shape[-2], w.shape[-1]))
    yl = y[:,:,:,0]
    yh = y[:,:,:,1:]
    deg75r, deg105i = pm(yh[:,:,0,0], yh[:,:,3,0])
    deg105r, deg75i = pm(yh[:,:,1,0], yh[:,:,2,0])
    deg15r, deg165i = pm(yh[:,:,0,1], yh[:,:,3,1])
    deg165r, deg15i = pm(yh[:,:,1,1], yh[:,:,2,1])
    deg135r, deg45i = pm(yh[:,:,0,2], yh[:,:,3,2])
    deg45r, deg135i = pm(yh[:,:,1,2], yh[:,:,2,2])
    yhr = torch.stack((deg15r, deg45r, deg75r, deg105r, deg135r, deg165r), dim=1)
    yhi = torch.stack((deg15i, deg45i, deg75i, deg105i, deg135i, deg165i), dim=1)
    yh = torch.stack((yhr, yhi), dim=-1)

    yl_rowa = torch.stack((yl[:,:,1], yl[:,:,0]), dim=-1)
    yl_rowb = torch.stack((yl[:,:,3], yl[:,:,2]), dim=-1)
    yl_rowa = yl_rowa.view(yl.shape[0], C, yl.shape[-2], yl.shape[-1]*2)
    yl_rowb = yl_rowb.view(yl.shape[0], C, yl.shape[-2], yl.shape[-1]*2)
    z = torch.stack((yl_rowb, yl_rowa), dim=-2)
    yl = z.view(yl.shape[0], C, yl.shape[-2]*2, yl.shape[-1]*2)

    return yl.contiguous(), yh


def quad_afb2d_nonsep(x, filts, mode='zero'):
    """ Does a 1 level 2d wavelet decomposition of an input. Doesn't do separate
    row and column filtering.

    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list or torch.Tensor): If a list is given, should be the low and
            highpass filter banks. If a tensor is given, it should be of the
            form created by
            :py:func:`pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d_nonsep`
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    C = x.shape[1]
    Ny = x.shape[2]
    Nx = x.shape[3]

    # Check the filter inputs
    f = torch.cat([filts]*C, dim=0)
    Ly = f.shape[2]
    Lx = f.shape[3]

    if mode == 'periodization' or mode == 'per':
        if x.shape[2] % 2 == 1:
            x = torch.cat((x, x[:,:,-1:]), dim=2)
            Ny += 1
        if x.shape[3] % 2 == 1:
            x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            Nx += 1
        pad = (Ly-1, Lx-1)
        stride = (2, 2)
        x = roll(roll(x, -Ly//2, dim=2), -Lx//2, dim=3)
        y = F.conv2d(x, f, padding=pad, stride=stride, groups=C)
        y[:,:,:Ly//2] += y[:,:,Ny//2:Ny//2+Ly//2]
        y[:,:,:,:Lx//2] += y[:,:,:,Nx//2:Nx//2+Lx//2]
        y = y[:,:,:Ny//2, :Nx//2]
    elif mode == 'zero' or mode == 'symmetric' or mode == 'reflect':
        # Calculate the pad size
        out1 = pywt.dwt_coeff_len(Ny, Ly, mode=mode)
        out2 = pywt.dwt_coeff_len(Nx, Lx, mode=mode)
        p1 = 2 * (out1 - 1) - Ny + Ly
        p2 = 2 * (out2 - 1) - Nx + Lx
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p1 % 2 == 1 and p2 % 2 == 1:
                x = F.pad(x, (0, 1, 0, 1))
            elif p1 % 2 == 1:
                x = F.pad(x, (0, 0, 0, 1))
            elif p2 % 2 == 1:
                x = F.pad(x, (0, 1, 0, 0))
            # Calculate the high and lowpass
            y = F.conv2d(
                x, f, padding=(p1//2, p2//2), stride=2, groups=C)
        elif mode == 'symmetric' or mode == 'reflect':
            pad = (p2//2, (p2+1)//2, p1//2, (p1+1)//2)
            x = mypad(x, pad=pad, mode=mode)
            y = F.conv2d(x, f, stride=2, groups=C)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    y = y.reshape((y.shape[0], C, 4, y.shape[-2], y.shape[-1]))
    yl = y[:,:,0].contiguous()
    yh = y[:,:,1:].contiguous()
    return yl, yh


def cplxdual2D(x, J, level1='farras', qshift='qshift_a', mode='periodization',
               mag=False):
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
            0,1 = 15 degrees
            1,2 = 45 degrees
            0,0 = 75 degrees
            1,0 = 105 degrees
            0,2 = 135 degrees
            1,1 = 165 degrees
    """
    x = x/2
    # Get the filters
    h0a1, h0b1, _, _, h1a1, h1b1, _, _ = _level1(level1)
    h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)

    Faf = ((prep_filt_afb2d(h0a1, h1a1, h0a1, h1a1, device=x.device),
            prep_filt_afb2d(h0a1, h1a1, h0b1, h1b1, device=x.device)),
           (prep_filt_afb2d(h0b1, h1b1, h0a1, h1a1, device=x.device),
            prep_filt_afb2d(h0b1, h1b1, h0b1, h1b1, device=x.device)))
    af = ((prep_filt_afb2d(h0a, h1a, h0a, h1a, device=x.device),
           prep_filt_afb2d(h0a, h1a, h0b, h1b, device=x.device)),
          (prep_filt_afb2d(h0b, h1b, h0a, h1a, device=x.device),
           prep_filt_afb2d(h0b, h1b, h0b, h1b, device=x.device)))

    # Do 4 fully decimated dwts
    w = [[[None for _ in range(2)] for _ in range(2)] for j in range(J)]
    lows = [[None for _ in range(2)] for _ in range(2)]
    for m in range(2):
        for n in range(2):
            # Do the first level transform with the first level filters
            #  ll, bands = afb2d(x, (Faf[m][0], Faf[m][1], Faf[n][0], Faf[n][1]), mode=mode)
            bands = afb2d(x, Faf[m][n], mode=mode)
            # Separate the low and bandpasses
            s = bands.shape
            bands = bands.reshape(s[0], -1, 4, s[-2], s[-1])
            ll = bands[:,:,0].contiguous()
            w[0][m][n] = [bands[:,:,2], bands[:,:,1], bands[:,:,3]]

            # Do the second+ level transform with the second level filters
            for j in range(1,J):
                #  ll, bands = afb2d(ll, (af[m][0], af[m][1], af[n][0], af[n][1]), mode=mode)
                bands = afb2d(ll, af[m][n], mode=mode)
                # Separate the low and bandpasses
                s = bands.shape
                bands = bands.reshape(s[0], -1, 4, s[-2], s[-1])
                ll = bands[:,:,0].contiguous()
                w[j][m][n] = [bands[:,:,2], bands[:,:,1], bands[:,:,3]]
            lows[m][n] = ll

    # Convert the quads into real and imaginary parts
    yh = [None,] * J
    for j in range(J):
        deg75r, deg105i = pm(w[j][0][0][0], w[j][1][1][0])
        deg105r, deg75i = pm(w[j][0][1][0], w[j][1][0][0])
        deg15r, deg165i = pm(w[j][0][0][1], w[j][1][1][1])
        deg165r, deg15i = pm(w[j][0][1][1], w[j][1][0][1])
        deg135r, deg45i = pm(w[j][0][0][2], w[j][1][1][2])
        deg45r, deg135i = pm(w[j][0][1][2], w[j][1][0][2])
        yhr = torch.stack((deg15r, deg45r, deg75r, deg105r, deg135r, deg165r), dim=1)
        yhi = torch.stack((deg15i, deg45i, deg75i, deg105i, deg135i, deg165i), dim=1)
        if mag:
            yh[j] = torch.sqrt(yhr**2 + yhi**2 + 0.01) - np.sqrt(0.01)
        else:
            yh[j] = torch.stack((yhr, yhi), dim=-1)

    return lows, yh


def icplxdual2D(yl, yh, level1='farras', qshift='qshift_a', mode='periodization'):
    # Get the filters
    _, _, g0a1, g0b1, _, _, g1a1, g1b1 = _level1(level1)
    _, _, g0a, g0b, _, _, g1a, g1b = _qshift(qshift)

    dev = yl[0][0].device
    Faf = ((prep_filt_sfb2d(g0a1, g1a1, g0a1, g1a1, device=dev),
            prep_filt_sfb2d(g0a1, g1a1, g0b1, g1b1, device=dev)),
           (prep_filt_sfb2d(g0b1, g1b1, g0a1, g1a1, device=dev),
            prep_filt_sfb2d(g0b1, g1b1, g0b1, g1b1, device=dev)))
    af = ((prep_filt_sfb2d(g0a, g1a, g0a, g1a, device=dev),
           prep_filt_sfb2d(g0a, g1a, g0b, g1b, device=dev)),
          (prep_filt_sfb2d(g0b, g1b, g0a, g1a, device=dev),
           prep_filt_sfb2d(g0b, g1b, g0b, g1b, device=dev)))

    # Convert the highs back to subbands
    J = len(yh)
    w = [[[[None for i in range(3)] for j in range(2)] for k in range(2)] for l in range(J)]
    for j in range(J):
        w[j][0][0][0], w[j][1][1][0] = pm(yh[j][:,2,:,:,:,0],
                                          yh[j][:,3,:,:,:,1])
        w[j][0][1][0], w[j][1][0][0] = pm(yh[j][:,3,:,:,:,0],
                                          yh[j][:,2,:,:,:,1])
        w[j][0][0][1], w[j][1][1][1] = pm(yh[j][:,0,:,:,:,0],
                                          yh[j][:,5,:,:,:,1])
        w[j][0][1][1], w[j][1][0][1] = pm(yh[j][:,5,:,:,:,0],
                                          yh[j][:,0,:,:,:,1])
        w[j][0][0][2], w[j][1][1][2] = pm(yh[j][:,1,:,:,:,0],
                                          yh[j][:,4,:,:,:,1])
        w[j][0][1][2], w[j][1][0][2] = pm(yh[j][:,4,:,:,:,0],
                                          yh[j][:,1,:,:,:,1])
        w[j][0][0] = torch.stack(w[j][0][0], dim=2)
        w[j][0][1] = torch.stack(w[j][0][1], dim=2)
        w[j][1][0] = torch.stack(w[j][1][0], dim=2)
        w[j][1][1] = torch.stack(w[j][1][1], dim=2)

    y = None
    for m in range(2):
        for n in range(2):
            lo = yl[m][n]
            for j in range(J-1, 0, -1):
                lo = sfb2d(lo, w[j][m][n], af[m][n], mode=mode)
            lo = sfb2d(lo, w[0][m][n], Faf[m][n], mode=mode)

            # Add to the output
            if y is None:
                y = lo
            else:
                y = y + lo

    # Normalize
    y = y/2
    return y


def pm(a, b):
    u = (a + b)/np.sqrt(2)
    v = (a - b)/np.sqrt(2)
    return u, v
