from __future__ import absolute_import
import torch
import torch.nn.functional as F

from pytorch_wavelets.dtcwt.transform_funcs import fwd_j1, inv_j1
from pytorch_wavelets.dtcwt.transform_funcs import fwd_j1_rot, inv_j1_rot
from pytorch_wavelets.dtcwt.transform_funcs import fwd_j2plus, inv_j2plus
from pytorch_wavelets.dtcwt.transform_funcs import fwd_j2plus_rot, inv_j2plus_rot


def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


class SmoothMagFn(torch.autograd.Function):
    """ Class to do complex magnitude """
    @staticmethod
    def forward(ctx, x, y, b):
        r = torch.sqrt(x**2 + y**2 + b**2)
        if x.requires_grad:
            dx = x/r
            dy = y/r
            ctx.save_for_backward(dx, dy)

        return r - b

    @staticmethod
    def backward(ctx, dr):
        dx = None
        if ctx.needs_input_grad[0]:
            drdx, drdy = ctx.saved_tensors
            dx = drdx * dr
            dy = drdy * dr
        return dx, dy, None


class ScatLayerj1_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. """

    @staticmethod
    def forward(ctx, x, h0o, h1o, mode, bias, combine_colour):
        #  bias = 1e-2
        #  bias = 0
        ctx.in_shape = x.shape
        batch, ch, r, c = x.shape
        assert r % 2 == c % 2 == 0
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.combine_colour = combine_colour

        ll, reals, imags = fwd_j1(x, h0o, h1o, False, 1, mode)
        ll = F.avg_pool2d(ll, 2)
        if combine_colour:
            r = torch.sqrt(reals[:,:,0]**2 + imags[:,:,0]**2 +
                           reals[:,:,1]**2 + imags[:,:,1]**2 +
                           reals[:,:,2]**2 + imags[:,:,2]**2 + bias**2)
            r = r[:, :, None]
        else:
            r = torch.sqrt(reals**2 + imags**2 + bias**2)

        if x.requires_grad:
            drdx = reals/r
            drdy = imags/r
            ctx.save_for_backward(h0o, h1o, drdx, drdy)
        else:
            z = x.new_zeros(1)
            ctx.save_for_backward(h0o, h1o, z, z)

        r = r - bias
        del reals, imags
        if combine_colour:
            Z = torch.cat((ll, r[:, :, 0]), dim=1)
        else:
            Z = torch.cat((ll[:, None], r), dim=1)

        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode

        if ctx.needs_input_grad[0]:
            #  h0o, h1o, θ = ctx.saved_tensors
            h0o, h1o, drdx, drdy = ctx.saved_tensors
            # Use the special properties of the filters to get the time reverse
            h0o_t = h0o
            h1o_t = h1o

            # Level 1 backward (time reversed biorthogonal analysis filters)
            if ctx.combine_colour:
                dYl, dr = dZ[:,:3], dZ[:,3:]
                dr = dr[:, :, None]
            else:
                dYl, dr = dZ[:,0], dZ[:,1:]
            ll = 1/4 * F.interpolate(dYl, scale_factor=2, mode="nearest")
            reals = dr * drdx
            imags = dr * drdy

            dX = inv_j1(ll, reals, imags, h0o_t, h1o_t, 1, 3, 4, mode)

        return (dX,) + (None,) * 5


class ScatLayerj1_rot_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. Uses the rotationally symmetric
    filters, i.e. a slightly more expensive operation."""

    @staticmethod
    def forward(ctx, x, h0o, h1o, h2o, mode, bias, combine_colour):
        mode = int_to_mode(mode)
        ctx.mode = mode
        #  bias = 0
        ctx.in_shape = x.shape
        ctx.combine_colour = combine_colour
        batch, ch, r, c = x.shape
        assert r % 2 == c % 2 == 0

        # Level 1 forward (biorthogonal analysis filters)
        ll, reals, imags = fwd_j1_rot(x, h0o, h1o, h2o, False, 1, mode)
        ll = F.avg_pool2d(ll, 2)
        if combine_colour:
            r = torch.sqrt(reals[:,:,0]**2 + imags[:,:,0]**2 +
                           reals[:,:,1]**2 + imags[:,:,1]**2 +
                           reals[:,:,2]**2 + imags[:,:,2]**2 + bias**2)
            r = r[:, :, None]
        else:
            r = torch.sqrt(reals**2 + imags**2 + bias**2)
        if x.requires_grad:
            drdx = reals/r
            drdy = imags/r
            ctx.save_for_backward(h0o, h1o, h2o, drdx, drdy)
        else:
            z = x.new_zeros(1)
            ctx.save_for_backward(h0o, h1o, h2o, z, z)
        r = r - bias
        del reals, imags
        if combine_colour:
            Z = torch.cat((ll, r[:, :, 0]), dim=1)
        else:
            Z = torch.cat((ll[:, None], r), dim=1)

        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode

        if ctx.needs_input_grad[0]:
            # Don't need to do time reverse as these filters are symmetric
            #  h0o, h1o, h2o, θ = ctx.saved_tensors
            h0o, h1o, h2o, drdx, drdy = ctx.saved_tensors

            # Level 1 backward (time reversed biorthogonal analysis filters)
            if ctx.combine_colour:
                dYl, dr = dZ[:,:3], dZ[:,3:]
                dr = dr[:, :, None]
            else:
                dYl, dr = dZ[:,0], dZ[:,1:]
            ll = 1/4 * F.interpolate(dYl, scale_factor=2, mode="nearest")

            reals = dr * drdx
            imags = dr * drdy
            dX = inv_j1_rot(ll, reals, imags, h0o, h1o, h2o, 1, 3, 4, mode)

        return (dX,) + (None,) * 6


class ScatLayerj2_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. """

    @staticmethod
    def forward(ctx, x, h0o, h1o, h0a, h0b, h1a, h1b, mode, bias, combine_colour):
        #  bias = 1e-2
        #  bias = 0
        ctx.in_shape = x.shape
        batch, ch, r, c = x.shape
        assert r % 8 == c % 8 == 0
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.combine_colour = combine_colour

        # First order scattering
        s0, reals, imags = fwd_j1(x, h0o, h1o, False, 1, mode)
        if combine_colour:
            s1_j1 = torch.sqrt(reals[:,:,0]**2 + imags[:,:,0]**2 +
                               reals[:,:,1]**2 + imags[:,:,1]**2 +
                               reals[:,:,2]**2 + imags[:,:,2]**2 + bias**2)
            s1_j1 = s1_j1[:, :, None]
            if x.requires_grad:
                dsdx1 = reals/s1_j1
                dsdy1 = imags/s1_j1
            s1_j1 = s1_j1 - bias

            s0, reals, imags = fwd_j2plus(s0, h0a, h1a, h0b, h1b, False, 1, mode)
            s1_j2 = torch.sqrt(reals[:,:,0]**2 + imags[:,:,0]**2 +
                               reals[:,:,1]**2 + imags[:,:,1]**2 +
                               reals[:,:,2]**2 + imags[:,:,2]**2 + bias**2)
            s1_j2 = s1_j2[:, :, None]
            if x.requires_grad:
                dsdx2 = reals/s1_j2
                dsdy2 = imags/s1_j2
            s1_j2 = s1_j2 - bias
            s0 = F.avg_pool2d(s0, 2)

            # Second order scattering
            s1_j1 = s1_j1[:, :, 0]
            s1_j1, reals, imags = fwd_j1(s1_j1, h0o, h1o, False, 1, mode)
            s2_j1 = torch.sqrt(reals**2 + imags**2 + bias**2)
            if x.requires_grad:
                dsdx2_1 = reals/s2_j1
                dsdy2_1 = imags/s2_j1
            q = s2_j1.shape
            s2_j1 = s2_j1.view(q[0], 36, q[3], q[4])
            s2_j1 = s2_j1 - bias
            s1_j1 = F.avg_pool2d(s1_j1, 2)
            if x.requires_grad:
                ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b,
                                      dsdx1, dsdy1, dsdx2, dsdy2,
                                      dsdx2_1, dsdy2_1)
            else:
                z = x.new_zeros(1)
                ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b,
                                      z, z, z, z, z, z)

            del reals, imags
            Z = torch.cat((s0, s1_j1, s1_j2[:,:,0], s2_j1), dim=1)

        else:
            s1_j1 = torch.sqrt(reals**2 + imags**2 + bias**2)
            if x.requires_grad:
                dsdx1 = reals/s1_j1
                dsdy1 = imags/s1_j1
            s1_j1 = s1_j1 - bias

            s0, reals, imags = fwd_j2plus(s0, h0a, h1a, h0b, h1b, False, 1, mode)
            s1_j2 = torch.sqrt(reals**2 + imags**2 + bias**2)
            if x.requires_grad:
                dsdx2 = reals/s1_j2
                dsdy2 = imags/s1_j2
            s1_j2 = s1_j2 - bias
            s0 = F.avg_pool2d(s0, 2)

            # Second order scattering
            p = s1_j1.shape
            s1_j1 = s1_j1.view(p[0], 6*p[2], p[3], p[4])

            s1_j1, reals, imags = fwd_j1(s1_j1, h0o, h1o, False, 1, mode)
            s2_j1 = torch.sqrt(reals**2 + imags**2 + bias**2)
            if x.requires_grad:
                dsdx2_1 = reals/s2_j1
                dsdy2_1 = imags/s2_j1
            q = s2_j1.shape
            s2_j1 = s2_j1.view(q[0], 36, q[2]//6, q[3], q[4])
            s2_j1 = s2_j1 - bias
            s1_j1 = F.avg_pool2d(s1_j1, 2)
            s1_j1 = s1_j1.view(p[0], 6, p[2], p[3]//2, p[4]//2)

            if x.requires_grad:
                ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b,
                                      dsdx1, dsdy1, dsdx2, dsdy2,
                                      dsdx2_1, dsdy2_1)
            else:
                z = x.new_zeros(1)
                ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b,
                                      z, z, z, z, z, z)

            del reals, imags
            Z = torch.cat((s0[:, None], s1_j1, s1_j2, s2_j1), dim=1)

        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode

        if ctx.needs_input_grad[0]:
            # Input has shape N, L, C, H, W
            o_dim = 1
            h_dim = 3
            w_dim = 4

            # Retrieve phase info
            (h0o, h1o, h0a, h0b, h1a, h1b, dsdx1, dsdy1, dsdx2, dsdy2, dsdx2_1,
             dsdy2_1) = ctx.saved_tensors

            # Use the special properties of the filters to get the time reverse
            h0o_t = h0o
            h1o_t = h1o
            h0a_t = h0b
            h0b_t = h0a
            h1a_t = h1b
            h1b_t = h1a

            # Level 1 backward (time reversed biorthogonal analysis filters)
            if ctx.combine_colour:
                ds0, ds1_j1, ds1_j2, ds2_j1 = \
                    dZ[:,:3], dZ[:,3:9], dZ[:,9:15], dZ[:,15:]
                ds1_j2 = ds1_j2[:, :, None]

                ds1_j1 = 1/4 * F.interpolate(ds1_j1, scale_factor=2, mode="nearest")
                q = ds2_j1.shape
                ds2_j1 = ds2_j1.view(q[0], 6, 6, q[2], q[3])

                # Inverse second order scattering
                reals = ds2_j1 * dsdx2_1
                imags = ds2_j1 * dsdy2_1
                ds1_j1 = inv_j1(
                    ds1_j1, reals, imags, h0o_t, h1o_t, o_dim, h_dim, w_dim, mode)
                ds1_j1 = ds1_j1[:, :, None]

                # Inverse first order scattering j=2
                ds0 = 1/4 * F.interpolate(ds0, scale_factor=2, mode="nearest")
                #  s = ds1_j2.shape
                #  ds1_j2 = ds1_j2.view(s[0], 6, s[1]//6, s[2], s[3])
                reals = ds1_j2 * dsdx2
                imags = ds1_j2 * dsdy2
                ds0 = inv_j2plus(
                    ds0, reals, imags, h0a_t, h1a_t, h0b_t, h1b_t,
                    o_dim, h_dim, w_dim, mode)

                # Inverse first order scattering j=1
                reals = ds1_j1 * dsdx1
                imags = ds1_j1 * dsdy1
                dX = inv_j1(
                    ds0, reals, imags, h0o_t, h1o_t, o_dim, h_dim, w_dim, mode)
            else:
                ds0, ds1_j1, ds1_j2, ds2_j1 = \
                    dZ[:,0], dZ[:,1:7], dZ[:,7:13], dZ[:,13:]
                p = ds1_j1.shape
                ds1_j1 = ds1_j1.view(p[0], p[2]*6, p[3], p[4])
                ds1_j1 = 1/4 * F.interpolate(ds1_j1, scale_factor=2, mode="nearest")
                q = ds2_j1.shape
                ds2_j1 = ds2_j1.view(q[0], 6, q[2]*6, q[3], q[4])

                # Inverse second order scattering
                reals = ds2_j1 * dsdx2_1
                imags = ds2_j1 * dsdy2_1
                ds1_j1 = inv_j1(
                    ds1_j1, reals, imags, h0o_t, h1o_t, o_dim, h_dim, w_dim, mode)
                ds1_j1 = ds1_j1.view(p[0], 6, p[2], p[3]*2, p[4]*2)

                # Inverse first order scattering j=2
                ds0 = 1/4 * F.interpolate(ds0, scale_factor=2, mode="nearest")
                #  s = ds1_j2.shape
                #  ds1_j2 = ds1_j2.view(s[0], 6, s[1]//6, s[2], s[3])
                reals = ds1_j2 * dsdx2
                imags = ds1_j2 * dsdy2
                ds0 = inv_j2plus(
                    ds0, reals, imags, h0a_t, h1a_t, h0b_t, h1b_t,
                    o_dim, h_dim, w_dim, mode)

                # Inverse first order scattering j=1
                reals = ds1_j1 * dsdx1
                imags = ds1_j1 * dsdy1
                dX = inv_j1(
                    ds0, reals, imags, h0o_t, h1o_t, o_dim, h_dim, w_dim, mode)

        return (dX,) + (None,) * 9


class ScatLayerj2_rot_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT bandpass biorthogonal and qshift filters . """

    @staticmethod
    def forward(ctx, x, h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b, mode, bias, combine_colour):
        #  bias = 1e-2
        #  bias = 0
        ctx.in_shape = x.shape
        batch, ch, r, c = x.shape
        assert r % 8 == c % 8 == 0
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.combine_colour = combine_colour

        # First order scattering
        s0, reals, imags = fwd_j1_rot(x, h0o, h1o, h2o, False, 1, mode)
        if combine_colour:
            s1_j1 = torch.sqrt(reals[:,:,0]**2 + imags[:,:,0]**2 +
                               reals[:,:,1]**2 + imags[:,:,1]**2 +
                               reals[:,:,2]**2 + imags[:,:,2]**2 + bias**2)
            s1_j1 = s1_j1[:, :, None]
            if x.requires_grad:
                dsdx1 = reals/s1_j1
                dsdy1 = imags/s1_j1
            s1_j1 = s1_j1 - bias

            s0, reals, imags = fwd_j2plus_rot(s0, h0a, h1a, h0b, h1b, h2a, h2b, False, 1, mode)
            s1_j2 = torch.sqrt(reals[:,:,0]**2 + imags[:,:,0]**2 +
                               reals[:,:,1]**2 + imags[:,:,1]**2 +
                               reals[:,:,2]**2 + imags[:,:,2]**2 + bias**2)
            s1_j2 = s1_j2[:, :, None]
            if x.requires_grad:
                dsdx2 = reals/s1_j2
                dsdy2 = imags/s1_j2
            s1_j2 = s1_j2 - bias
            s0 = F.avg_pool2d(s0, 2)

            # Second order scattering
            s1_j1 = s1_j1[:, :, 0]
            s1_j1, reals, imags = fwd_j1_rot(s1_j1, h0o, h1o, h2o, False, 1, mode)
            s2_j1 = torch.sqrt(reals**2 + imags**2 + bias**2)
            if x.requires_grad:
                dsdx2_1 = reals/s2_j1
                dsdy2_1 = imags/s2_j1
            q = s2_j1.shape
            s2_j1 = s2_j1.view(q[0], 36, q[3], q[4])
            s2_j1 = s2_j1 - bias
            s1_j1 = F.avg_pool2d(s1_j1, 2)
            if x.requires_grad:
                ctx.save_for_backward(h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b,
                                      dsdx1, dsdy1, dsdx2, dsdy2, dsdx2_1,
                                      dsdy2_1)
            else:
                z = x.new_zeros(1)
                ctx.save_for_backward(h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b,
                                      z, z, z, z, z, z)

            del reals, imags
            Z = torch.cat((s0, s1_j1, s1_j2[:, :, 0], s2_j1), dim=1)
        else:
            s1_j1 = torch.sqrt(reals**2 + imags**2 + bias**2)
            if x.requires_grad:
                dsdx1 = reals/s1_j1
                dsdy1 = imags/s1_j1
            s1_j1 = s1_j1 - bias

            s0, reals, imags = fwd_j2plus_rot(s0, h0a, h1a, h0b, h1b, h2a, h2b, False, 1, mode)
            s1_j2 = torch.sqrt(reals**2 + imags**2 + bias**2)
            if x.requires_grad:
                dsdx2 = reals/s1_j2
                dsdy2 = imags/s1_j2
            s1_j2 = s1_j2 - bias
            s0 = F.avg_pool2d(s0, 2)

            # Second order scattering
            p = s1_j1.shape
            s1_j1 = s1_j1.view(p[0], 6*p[2], p[3], p[4])
            s1_j1, reals, imags = fwd_j1_rot(s1_j1, h0o, h1o, h2o, False, 1, mode)
            s2_j1 = torch.sqrt(reals**2 + imags**2 + bias**2)
            if x.requires_grad:
                dsdx2_1 = reals/s2_j1
                dsdy2_1 = imags/s2_j1
            q = s2_j1.shape
            s2_j1 = s2_j1.view(q[0], 36, q[2]//6, q[3], q[4])
            s2_j1 = s2_j1 - bias
            s1_j1 = F.avg_pool2d(s1_j1, 2)
            s1_j1 = s1_j1.view(p[0], 6, p[2], p[3]//2, p[4]//2)

            if x.requires_grad:
                ctx.save_for_backward(h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b,
                                      dsdx1, dsdy1, dsdx2, dsdy2, dsdx2_1,
                                      dsdy2_1)
            else:
                z = x.new_zeros(1)
                ctx.save_for_backward(h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b,
                                      z, z, z, z, z, z)

            del reals, imags
            Z = torch.cat((s0[:, None], s1_j1, s1_j2, s2_j1), dim=1)

        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode

        if ctx.needs_input_grad[0]:
            # Input has shape N, L, C, H, W
            o_dim = 1
            h_dim = 3
            w_dim = 4

            # Retrieve phase info
            (h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b, dsdx1, dsdy1, dsdx2,
             dsdy2, dsdx2_1, dsdy2_1) = ctx.saved_tensors

            # Use the special properties of the filters to get the time reverse
            h0o_t = h0o
            h1o_t = h1o
            h2o_t = h2o
            h0a_t = h0b
            h0b_t = h0a
            h1a_t = h1b
            h1b_t = h1a
            h2a_t = h2b
            h2b_t = h2a

            # Level 1 backward (time reversed biorthogonal analysis filters)
            if ctx.combine_colour:
                ds0, ds1_j1, ds1_j2, ds2_j1 = \
                    dZ[:,:3], dZ[:,3:9], dZ[:,9:15], dZ[:,15:]
                ds1_j2 = ds1_j2[:, :, None]

                # Inverse second order scattering
                ds1_j1 = 1/4 * F.interpolate(ds1_j1, scale_factor=2, mode="nearest")
                q = ds2_j1.shape
                ds2_j1 = ds2_j1.view(q[0], 6, 6, q[2], q[3])

                # Inverse second order scattering
                reals = ds2_j1 * dsdx2_1
                imags = ds2_j1 * dsdy2_1
                ds1_j1 = inv_j1_rot(
                    ds1_j1, reals, imags, h0o_t, h1o_t, h2o_t,
                    o_dim, h_dim, w_dim, mode)
                ds1_j1 = ds1_j1[:, :, None]

                # Inverse first order scattering j=2
                ds0 = 1/4 * F.interpolate(ds0, scale_factor=2, mode="nearest")
                #  s = ds1_j2.shape
                #  ds1_j2 = ds1_j2.view(s[0], 6, s[1]//6, s[2], s[3])
                reals = ds1_j2 * dsdx2
                imags = ds1_j2 * dsdy2
                ds0 = inv_j2plus_rot(
                    ds0, reals, imags, h0a_t, h1a_t, h0b_t, h1b_t, h2a_t, h2b_t,
                    o_dim, h_dim, w_dim, mode)

                # Inverse first order scattering j=1
                reals = ds1_j1 * dsdx1
                imags = ds1_j1 * dsdy1
                dX = inv_j1_rot(
                    ds0, reals, imags, h0o_t, h1o_t, h2o_t,
                    o_dim, h_dim, w_dim, mode)
            else:
                ds0, ds1_j1, ds1_j2, ds2_j1 = \
                    dZ[:,0], dZ[:,1:7], dZ[:,7:13], dZ[:,13:]

                # Inverse second order scattering
                p = ds1_j1.shape
                ds1_j1 = ds1_j1.view(p[0], p[2]*6, p[3], p[4])
                ds1_j1 = 1/4 * F.interpolate(ds1_j1, scale_factor=2, mode="nearest")
                q = ds2_j1.shape
                ds2_j1 = ds2_j1.view(q[0], 6, q[2]*6, q[3], q[4])
                reals = ds2_j1 * dsdx2_1
                imags = ds2_j1 * dsdy2_1
                ds1_j1 = inv_j1_rot(
                    ds1_j1, reals, imags, h0o_t, h1o_t, h2o_t,
                    o_dim, h_dim, w_dim, mode)
                ds1_j1 = ds1_j1.view(p[0], 6, p[2], p[3]*2, p[4]*2)

                # Inverse first order scattering j=2
                ds0 = 1/4 * F.interpolate(ds0, scale_factor=2, mode="nearest")
                #  s = ds1_j2.shape
                #  ds1_j2 = ds1_j2.view(s[0], 6, s[1]//6, s[2], s[3])
                reals = ds1_j2 * dsdx2
                imags = ds1_j2 * dsdy2
                ds0 = inv_j2plus_rot(
                    ds0, reals, imags, h0a_t, h1a_t, h0b_t, h1b_t, h2a_t, h2b_t,
                    o_dim, h_dim, w_dim, mode)

                # Inverse first order scattering j=1
                reals = ds1_j1 * dsdx1
                imags = ds1_j1 * dsdy1
                dX = inv_j1_rot(
                    ds0, reals, imags, h0o_t, h1o_t, h2o_t,
                    o_dim, h_dim, w_dim, mode)

        return (dX,) + (None,) * 12
