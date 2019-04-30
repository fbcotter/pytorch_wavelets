import torch
from torch import tensor
from torch.autograd import Function
from pytorch_wavelets.dtcwt.lowlevel import colfilter, rowfilter
from pytorch_wavelets.dtcwt.lowlevel import coldfilt, rowdfilt
from pytorch_wavelets.dtcwt.lowlevel import colifilt, rowifilt, q2c, c2q


def get_dimensions(o_dim, ri_dim):
    # Calculate which dimension to put the real and imaginary parts and the
    # orientations. Also work out where the rows and columns in the original
    # image were
    o_dim = (o_dim % 6)
    ri_dim = (ri_dim % 6)

    if o_dim < ri_dim:
        ri_dim -= 1

    if o_dim >= 3 and ri_dim >= 3:
        h_dim = 2
    elif o_dim >= 4 or ri_dim >= 4:
        h_dim = 3
    else:
        h_dim = 4

    if o_dim >= 4 and ri_dim >= 4:
        w_dim = 3
    elif o_dim >= 4 or ri_dim >= 4:
        w_dim = 4
    else:
        w_dim = 5

    return o_dim, ri_dim, h_dim, w_dim


def highs_to_orientations(lh, hl, hh, o_dim):
    deg15, deg165 = q2c(lh, ri_dim)
    deg45, deg135 = q2c(hh, ri_dim)
    deg75, deg105 = q2c(hl, ri_dim)
    highs = torch.stack(
        [deg15, deg45, deg75, deg105, deg135, deg165], dim=o_dim)
    return highs


def orientations_to_highs(highs, o_dim, ri_dim):
    dev = highs.device
    horiz = torch.index_select(highs, o_dim, tensor([0, 5], device=dev))
    diag = torch.index_select(highs, o_dim, tensor([1, 4], device=dev))
    vertic = torch.index_select(highs, o_dim, tensor([2, 3], device=dev))
    deg15, deg165 = torch.unbind(horiz, dim=o_dim)
    deg45, deg135 = torch.unbind(diag, dim=o_dim)
    deg75, deg105 = torch.unbind(vertic, dim=o_dim)

    lh = c2q(deg15, deg165, ri_dim)
    hl = c2q(deg75, deg105, ri_dim)
    hh = c2q(deg45, deg135, ri_dim)

    return lh, hl, hh


def fwd_J1(x, h0, h1, skip_hps, o_dim, ri_dim):
    # Level 1 forward (biorthogonal analysis filters)
    lo = rowfilter(x, h0)
    if not skip_hps:
        hi = rowfilter(x, h1)
        ll = colfilter(lo, h0)
        lh = colfilter(lo, h1)
        hl = colfilter(hi, h0)
        hh = colfilter(hi, h1)
        del lo, hi
        highs = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = colfilter(lo, h0)
        highs = x.new_empty([0])
    return ll, highs


def fwd_J2plus(x, h0a, h0b, h1a, h1b, skip_hps, o_dim, ri_dim):
    lo = rowdfilt(x, h0b, h0a)
    if not skip_hps:
        hi = rowdfilt(x, h1b, h1a, highpass=True)
        ll = coldfilt(lo, h0b, h0a)
        lh = coldfilt(lo, h1b, h1a, highpass=True)
        hl = coldfilt(hi, h0b, h0a)
        hh = coldfilt(hi, h1b, h1a, highpass=True)
        del lo, hi
        highs = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = coldfilt(lo, h0b, h0a)
        highs = x.new_empty([0])

    return ll, highs


def inv_J1(ll, highs, g0, g1, o_dim, ri_dim, h_dim, w_dim):

    if highs is None or highs.shape == torch.Size([0]):
        y = rowfilter(colfilter(ll, g0), g0)
    else:
        # Get the double sampled bandpass coefficients
        lh, hl, hh = orientations_to_highs(highs, o_dim, ri_dim)

        if ll is None or ll.shape == torch.Size([0]):
            # Interpolate
            hi = colfilter(hh, g1) + colfilter(hl, g0)
            lo = colfilter(lh, g1)
            del lh, hh, hl
        else:
            # Possibly cut back some rows to make the ll match the highs
            r, c = ll.shape[2:]
            r1, c1 = highs.shape[h_dim], highs.shape[w_dim]
            if r != r1 * 2:
                ll = ll[:,:,1:-1]
            if c != c1 * 2:
                ll = ll[:,:,:,1:-1]
            # Interpolate
            hi = colfilter(hh, g1) + colfilter(hl, g0)
            lo = colfilter(lh, g1) + colfilter(ll, g0)
            del lh, hl, hh

        y = rowfilter(hi, g1) + rowfilter(lo, g0)

    return y


def inv_J2plus(ll, highs, g0a, g0b, g1a, g1b, o_dim, ri_dim, h_dim, w_dim):
    if highs is None or highs.shape == torch.Size([0]):
        y = rowifilt(colifilt(ll, g0b, g0a), g0b, g0a)
    else:
        # Get the double sampled bandpass coefficients
        lh, hl, hh = orientations_to_highs(highs, o_dim, ri_dim)

        if ll is None or ll.shape == torch.Size([0]):
            # Interpolate
            hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
            lo = colifilt(lh, g1b, g1a, True)
            del lh, hh, hl
        else:
            # Possibly cut back some rows to make the ll match the highs
            r, c = ll.shape[2:]
            r1, c1 = highs.shape[h_dim], highs.shape[w_dim]
            if r != r1 * 2:
                ll = ll[:,:,1:-1]
            if c != c1 * 2:
                ll = ll[:,:,:,1:-1]
            # Interpolate
            hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
            lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
            del lh, hl, hh

        y = rowifilt(hi, g1a, g1b, True) + rowifilt(lo, g0b, g0a)
    return y


class xfm(Function):

    @staticmethod
    def forward(ctx, input, J, h0o, h1o, h0a, h1a, h0b, h1b, skip_hps,
                include_scale, o_dim, ri_dim):
        ctx.save_for_backward(h0o, h1o, h0a, h1a, h0b, h1b)
        ctx.o_dim, ctx.ri_dim, ctx.h_dim, ctx.w_dim = get_dimensions(
            o_dim, ri_dim)
        ctx.in_shape = input.shape
        ctx.include_scale = include_scale
        ctx.extra_rows = 0
        ctx.extra_cols = 0
        batch, ch, r, c = input.shape

        # If the row/col count of X is not divisible by 2 then we need to
        # extend X
        if r % 2 != 0:
            input = torch.cat((input, input[:,:,-1:]), dim=2)
            ctx.extra_rows = 1
        if c % 2 != 0:
            input = torch.cat((input, input[:,:,:,-1:]), dim=3)
            ctx.extra_cols = 1


        if ctx.include_scale[0]:
            Ys1 = LoLo
        else:
            Ys1 = torch.tensor([], device=input.device)

        Yl = LoLo
        return Yl, Yh1

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yh1):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        in_shape = ctx.in_shape
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if True in ctx.needs_input_grad:
            ll = grad_LoLo

            # Level 1 backward (time reversed biorthogonal analysis filters)
            if not ctx.skip_hps[0]:
                dev = grad_Yh1.device
                deg15, deg165 = torch.unbind(torch.index_select(
                    grad_Yh1, ctx.o_dim, torch.tensor([0, 5], device=dev)), dim=ctx.o_dim)
                deg45, deg135 = torch.unbind(torch.index_select(
                    grad_Yh1, ctx.o_dim, torch.tensor([1, 4], device=dev)), dim=ctx.o_dim)
                deg75, deg105 = torch.unbind(torch.index_select(
                    grad_Yh1, ctx.o_dim, torch.tensor([2, 3], device=dev)), dim=ctx.o_dim)
                lh = c2q(deg15, deg165, ctx.ri_dim)
                hl = c2q(deg75, deg105, ctx.ri_dim)
                hh = c2q(deg45, deg135, ctx.ri_dim)
                Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
                Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
                grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)
            else:
                grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)
            if ctx.extra_rows:
                grad_input = grad_input[:,:,:-1]
            if ctx.extra_cols:
                grad_input = grad_input[:,:,:,:-1]


        return (grad_input,) + (None,) * 10

