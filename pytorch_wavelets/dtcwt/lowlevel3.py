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


def highs_to_orientations(lh, hl, hh, o_dim, ri_dim):
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


def fwd_j1(x, h0, h1, skip_hps, o_dim, ri_dim):
    """ Level 1 forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    # Level 1 forward (biorthogonal analysis filters)
    if not skip_hps:
        lo = rowfilter(x, h0)
        hi = rowfilter(x, h1)
        ll = colfilter(lo, h0)
        lh = colfilter(lo, h1)
        hl = colfilter(hi, h0)
        hh = colfilter(hi, h1)
        del lo, hi
        highs = highs_to_orientations(lh, hl, hh, o_dim, ri_dim)
    else:
        ll = rowfilter(x, h0)
        ll = colfilter(ll, h0)
        highs = x.new_zeros([])
    return ll, highs


def inv_j1(ll, highs, g0, g1, o_dim, ri_dim, h_dim, w_dim):
    """ Level1 inverse dtcwt.

    Have it as a separate function as can be used by the forward pass of the
    inverse transform and the backward pass of the forward transform.
    """
    if highs is None or highs.shape == torch.Size([]):
        y = rowfilter(colfilter(ll, g0), g0)
    else:
        # Get the double sampled bandpass coefficients
        lh, hl, hh = orientations_to_highs(highs, o_dim, ri_dim)

        if ll is None or ll.shape == torch.Size([]):
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


def fwd_j2plus(x, h0a, h1a, h0b, h1b, skip_hps, o_dim, ri_dim):
    """ Level 2 plus forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    if not skip_hps:
        lo = rowdfilt(x, h0b, h0a)
        hi = rowdfilt(x, h1b, h1a, highpass=True)

        ll = coldfilt(lo, h0b, h0a)
        lh = coldfilt(lo, h1b, h1a, highpass=True)
        hl = coldfilt(hi, h0b, h0a)
        hh = coldfilt(hi, h1b, h1a, highpass=True)
        del lo, hi
        highs = highs_to_orientations(lh, hl, hh, o_dim, ri_dim)
    else:
        ll = rowdfilt(x, h0b, h0a)
        ll = coldfilt(ll, h0b, h0a)
        highs = x.new_zeros([])

    return ll, highs


def inv_j2plus(ll, highs, g0a, g1a, g0b, g1b, o_dim, ri_dim, h_dim, w_dim):
    """ Level2+ inverse dtcwt.

    Have it as a separate function as can be used by the forward pass of the
    inverse transform and the backward pass of the forward transform.
    """
    if highs is None or highs.shape == torch.Size([]):
        y = rowifilt(colifilt(ll, g0b, g0a), g0b, g0a)
    else:
        # Get the double sampled bandpass coefficients
        lh, hl, hh = orientations_to_highs(highs, o_dim, ri_dim)

        if ll is None or ll.shape == torch.Size([]):
            # Interpolate
            hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
            lo = colifilt(lh, g1b, g1a, True)
            del lh, hh, hl
        else:
            # Interpolate
            hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
            lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
            del lh, hl, hh

        y = rowifilt(hi, g1b, g1a, True) + rowifilt(lo, g0b, g0a)
    return y


class FWD_J1(Function):
    """ Differentiable function doing 1 level forward DTCWT """
    @staticmethod
    def forward(ctx, x, h0, h1, skip_hps, o_dim, ri_dim):
        ctx.save_for_backward(h0, h1)
        ctx.dims = get_dimensions(o_dim, ri_dim)
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]

        ll, highs = fwd_j1(x, h0, h1, skip_hps, o_dim, ri_dim)
        return ll, highs

    @staticmethod
    def backward(ctx, dl, dh):
        h0, h1 = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            o_dim, ri_dim, h_dim, w_dim = ctx.dims
            dx = inv_j1(dl, dh, h0, h1, o_dim, ri_dim, h_dim, w_dim)

        return dx, None, None, None, None, None


class FWD_J2PLUS(Function):
    """ Differentiable function doing second level forward DTCWT """
    @staticmethod
    def forward(ctx, x, h0a, h1a, h0b, h1b, skip_hps, o_dim, ri_dim):
        ctx.save_for_backward(h0a, h1a, h0b, h1b)
        ctx.dims = get_dimensions(o_dim, ri_dim)
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]

        ll, highs = fwd_j2plus(x, h0a, h1a, h0b, h1b, skip_hps, o_dim, ri_dim)
        return ll, highs

    @staticmethod
    def backward(ctx, dl, dh):
        h0a, h1a, h0b, h1b = ctx.saved_tensors
        # The colifilt and rowifilt functions use conv2d not conv2d_transpose,
        # so need to reverse the filters
        h0a, h0b = h0b, h0a
        h1a, h1b = h1b, h1a
        dx = None
        if ctx.needs_input_grad[0]:
            o_dim, ri_dim, h_dim, w_dim = ctx.dims
            dx = inv_j2plus(dl, dh, h0a, h1a, h0b, h1b,
                            o_dim, ri_dim, h_dim, w_dim)

        return dx, None, None, None, None, None, None, None


class INV_J1(Function):
    """ Differentiable function doing 1 level inverse DTCWT """
    @staticmethod
    def forward(ctx, lows, highs, g0, g1, o_dim, ri_dim):
        ctx.save_for_backward(g0, g1)
        ctx.dims = get_dimensions(o_dim, ri_dim)
        o_dim, ri_dim, h_dim, w_dim = ctx.dims
        y = inv_j1(lows, highs, g0, g1, o_dim, ri_dim, h_dim, w_dim)
        return y

    @staticmethod
    def backward(ctx, dy):
        g0, g1 = ctx.saved_tensors
        dl = None
        dh = None
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]
        if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            dl, _ = fwd_j1(dy, g0, g1, True, o_dim, ri_dim)
        elif ctx.needs_input_grad[1] and not ctx.needs_input_grad[0]:
            _, dh = fwd_j1(dy, g0, g1, False, o_dim, ri_dim)
        elif ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            dl, dh = fwd_j1(dy, g0, g1, False, o_dim, ri_dim)

        return dl, dh, None, None, None, None


class INV_J2PLUS(Function):
    """ Differentiable function doing level 2 onwards inverse DTCWT """
    @staticmethod
    def forward(ctx, lows, highs, g0a, g1a, g0b, g1b, o_dim, ri_dim):
        ctx.save_for_backward(g0a, g1a, g0b, g1b)
        ctx.dims = get_dimensions(o_dim, ri_dim)
        o_dim, ri_dim, h_dim, w_dim = ctx.dims
        y = inv_j2plus(lows, highs, g0a, g1a, g0b, g1b,
                       o_dim, ri_dim, h_dim, w_dim)
        return y

    @staticmethod
    def backward(ctx, dy):
        g0a, g1a, g0b, g1b = ctx.saved_tensors
        g0a, g0b = g0b, g0a
        g1a, g1b = g1b, g1a
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]
        if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            dl, _ = fwd_j2plus(dy, g0a, g1a, g0b, g1b, True, o_dim, ri_dim)
        elif ctx.needs_input_grad[1] and not ctx.needs_input_grad[0]:
            _, dh = fwd_j2plus(dy, g0a, g1a, g0b, g1b, False, o_dim, ri_dim)
        elif ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            dl, dh = fwd_j2plus(dy, g0a, g1a, g0b, g1b, False, o_dim, ri_dim)

        return dl, dh, None, None, None, None, None, None
