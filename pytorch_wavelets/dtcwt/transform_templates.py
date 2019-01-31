from torch.autograd import Function

## Forward Templates

level1_fwd = """# Level 1 forward (biorthogonal analysis filters)
        Lo = rowfilter(input, h0o)
        if not ctx.skip_hps[0]:
            Hi = rowfilter(input, h1o)
            LoHi = colfilter(Lo, h1o)
            HiLo = colfilter(Hi, h0o)
            HiHi = colfilter(Hi, h1o)
            deg15, deg165 = q2c(LoHi, ctx.ri_dim)
            deg45, deg135 = q2c(HiHi, ctx.ri_dim)
            deg75, deg105 = q2c(HiLo, ctx.ri_dim)
            Yh1 = torch.stack(
                [deg15, deg45, deg75, deg105, deg135, deg165], dim=ctx.o_dim)
        else:
            Yh1 = torch.tensor([], device=input.device)
        LoLo = colfilter(Lo, h0o)
        if ctx.include_scale[0]:
            Ys1 = LoLo
        else:
            Ys1 = torch.tensor([], device=input.device)
        """

level1_hps_bwd = """# Level 1 backward (time reversed biorthogonal analysis filters)
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
            """

level2plus_fwd = """# Level {j} forward (quater shift analysis filters)
        r, c = LoLo.shape[2:]
        if r % 4 != 0:
            LoLo = torch.cat((LoLo[:,:,0:1], LoLo, LoLo[:,:,-1:]), dim=2)
        if c % 4 != 0:
            LoLo = torch.cat((LoLo[:,:,:,0:1], LoLo, LoLo[:,:,:,-1:]), dim=3)

        Lo = rowdfilt(LoLo, h0b, h0a)
        if not ctx.skip_hps[{i}]:
            Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
            LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
            HiLo = coldfilt(Hi, h0b, h0a)
            HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

            deg15, deg165 = q2c(LoHi, ctx.ri_dim)
            deg45, deg135 = q2c(HiHi, ctx.ri_dim)
            deg75, deg105 = q2c(HiLo, ctx.ri_dim)
            Yh{j} = torch.stack(
                [deg15, deg45, deg75, deg105, deg135, deg165], dim=ctx.o_dim)
        else:
            Yh{j} = torch.tensor([], device=input.device)
        LoLo = coldfilt(Lo, h0b, h0a)
        if ctx.include_scale[{i}]:
            Ys{j} = LoLo
        else:
            Ys{j} = torch.tensor([], device=input.device)
"""

level2plus_bwd = """# Level {j} backward (time reversed quater shift analysis filters)
            if not ctx.skip_hps[{i}]:
                dev = grad_Yh{j}.device
                deg15, deg165 = torch.unbind(torch.index_select(
                    grad_Yh{j}, ctx.o_dim, torch.tensor([0, 5], device=dev)), dim=ctx.o_dim)
                deg45, deg135 = torch.unbind(torch.index_select(
                    grad_Yh{j}, ctx.o_dim, torch.tensor([1, 4], device=dev)), dim=ctx.o_dim)
                deg75, deg105 = torch.unbind(torch.index_select(
                    grad_Yh{j}, ctx.o_dim, torch.tensor([2, 3], device=dev)), dim=ctx.o_dim)
                lh = c2q(deg15, deg165, ctx.ri_dim)
                hl = c2q(deg75, deg105, ctx.ri_dim)
                hh = c2q(deg45, deg135, ctx.ri_dim)
                Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
                Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
                ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            else:
                ll = rowifilt(colifilt(Lo, h0b_t, h0a_t), h0b_t, h0a_t)
            {checkshape}
"""
level2plus_bwd_scale = """# Level {j} backward (time reversed quater shift analysis filters)
            if not ctx.skip_hps[{i}]:
                dev = grad_Yh{j}.device
                deg15, deg165 = torch.unbind(torch.index_select(
                    grad_Yh{j}, ctx.o_dim, torch.tensor([0, 5], device=dev)), dim=ctx.o_dim)
                deg45, deg135 = torch.unbind(torch.index_select(
                    grad_Yh{j}, ctx.o_dim, torch.tensor([1, 4], device=dev)), dim=ctx.o_dim)
                deg75, deg105 = torch.unbind(torch.index_select(
                    grad_Yh{j}, ctx.o_dim, torch.tensor([2, 3], device=dev)), dim=ctx.o_dim)
                lh = c2q(deg15, deg165, ctx.ri_dim)
                hl = c2q(deg75, deg105, ctx.ri_dim)
                hh = c2q(deg45, deg135, ctx.ri_dim)
                Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
                Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
                ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            else:
                ll = rowifilt(colifilt(Lo, h0b_t, h0a_t), h0b_t, h0a_t)
            {checkshape}
            if ctx.include_scale[{i2}]:
                ll = (ll + grad_Ys{j2})/2
"""
bwd_checkshape_hps = """r, c = ll.shape[2:]
            r1, c1 = grad_Yh{j2}.shape[ctx.h_dim], grad_Yh{j2}.shape[ctx.w_dim]
            if r != r1 * 2:
                ll = ll[:,:,1:-1]
            if c != c1 * 2:
                ll = ll[:,:,:,1:-1]"""
bwd_checkshape_nohps = """r, c = ll.shape[2:]
            r1, c1 = in_shape[ctx.h_dim], in_shape[ctx.w_dim]
            if r != r1 * 2:
                ll = ll[:,:,1:-1]
            if c != c1 * 2:
                ll = ll[:,:,:,1:-1]"""

xfm = """
class xfm{J}{scale}(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b, skip_hps, include_scale, o_dim, ri_dim):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.o_dim = (o_dim % 6)
        ctx.ri_dim = (ri_dim % 6)
        if ctx.o_dim < ctx.ri_dim:
            ctx.ri_dim -= 1
        if ctx.o_dim >= 3 and ctx.ri_dim >= 3:
            ctx.h_dim = 2
        elif ctx.o_dim >= 4 or ctx.ri_dim >= 4:
            ctx.h_dim = 3
        else:
            ctx.h_dim = 4
        if ctx.o_dim >= 4 and ctx.ri_dim >= 4:
            ctx.w_dim = 3
        elif ctx.o_dim >=4 or ctx.ri_dim >= 4:
            ctx.w_dim = 4
        else:
            ctx.w_dim = 5

        ctx.in_shape = input.shape
        ctx.skip_hps = skip_hps
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

        {level1}
        {level2plus}Yl = LoLo
        return {fwd_lo_out}, {fwd_out}

    @staticmethod
    def backward(ctx, {bwd_lo_in}, {bwd_in}):
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
            {bwd_lo_init}
            {level2plusbwd}
            {level1bwd}

        return (grad_input,) + (None,) * 10

"""

## Inverse Templates
level1_hps_fwd_inv = """# Level 1 inverse with biorthogonal synthesis filters
        if yh1 is not None and yh1.shape != torch.Size([0]):
            {checkshape}dev = yh1.device
            deg15, deg165 = torch.unbind(torch.index_select(
                yh1, ctx.o_dim, torch.tensor([0, 5], device=dev)), dim=ctx.o_dim)
            deg45, deg135 = torch.unbind(torch.index_select(
                yh1, ctx.o_dim, torch.tensor([1, 4], device=dev)), dim=ctx.o_dim)
            deg75, deg105 = torch.unbind(torch.index_select(
                yh1, ctx.o_dim, torch.tensor([2, 3], device=dev)), dim=ctx.o_dim)
            lh = c2q(deg15, deg165, ctx.ri_dim)
            hl = c2q(deg75, deg105, ctx.ri_dim)
            hh = c2q(deg45, deg135, ctx.ri_dim)
            Hi = colfilter(hh, g1o) + colfilter(hl, g0o)
            if ll is not None and ll.shape != torch.Size([0]):
                Lo = colfilter(lh, g1o) + colfilter(ll, g0o)
            else:
                Lo = colfilter(lh, g1o)
            y = rowfilter(Hi, g1o) + rowfilter(Lo, g0o)
        else:
            y = rowfilter(colfilter(ll, g0o), g0o)"""

level1_bwd_inv = """# Level 1 inverse gradient - same as fwd transform
            # with time reverse biorthogonal synthesis filters
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            {hps}"""

level1_hps_bwd_inv = """if ctx.needs_input_grad[1]:
                Hi = rowfilter(grad_y, g1o_t)
                LoHi = colfilter(Lo, g1o_t)
                HiLo = colfilter(Hi, g0o_t)
                HiHi = colfilter(Hi, g1o_t)
                deg15, deg165 = q2c(LoHi, ctx.ri_dim)
                deg45, deg135 = q2c(HiHi, ctx.ri_dim)
                deg75, deg105 = q2c(HiLo, ctx.ri_dim)
                grad_yh1 = torch.stack(
                    [deg15, deg45, deg75, deg105, deg135, deg165], dim=ctx.o_dim)
"""

level2plus_fwd_inv = """# Level {j} inverse transform with quater shift synthesis filters
        if yh{j} is not None and yh{j}.shape != torch.Size([0]):
            {checkshape}dev = yh{j}.device
            deg15, deg165 = torch.unbind(torch.index_select(
                yh{j}, ctx.o_dim, torch.tensor([0, 5], device=dev)), dim=ctx.o_dim)
            deg45, deg135 = torch.unbind(torch.index_select(
                yh{j}, ctx.o_dim, torch.tensor([1, 4], device=dev)), dim=ctx.o_dim)
            deg75, deg105 = torch.unbind(torch.index_select(
                yh{j}, ctx.o_dim, torch.tensor([2, 3], device=dev)), dim=ctx.o_dim)
            lh = c2q(deg15, deg165, ctx.ri_dim)
            hl = c2q(deg75, deg105, ctx.ri_dim)
            hh = c2q(deg45, deg135, ctx.ri_dim)
            Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
            if ll is not None and ll.shape != torch.Size([0]):
                Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
            else:
                Lo = colifilt(lh, g1b, g1a, True)
            ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        else:
            ll = rowifilt(colifilt(ll, g0b, g0a), g0b, g0a)
        """

fwd_checkshape_hps = """r, c = ll.shape[2:]
            r1, c1 = yh{j}.shape[ctx.h_dim], yh{j}.shape[ctx.w_dim]
            if r != r1 * 2:
                ll = ll[:,:,1:-1]
            if c != c1 * 2:
                ll = ll[:,:,:,1:-1]
            """

level2plus_bwd_inv = """# Level {j} inverse gradient - same as fwd transform
            # but with time-reverse quater shift synthesis filters
            r, c = LoLo.shape[2:]
            if r % 4 != 0:
                LoLo = torch.cat((LoLo[:,:,0:1], LoLo, LoLo[:,:,-1:]), dim=2)
            if c % 4 != 0:
                LoLo = torch.cat((LoLo[:,:,:,0:1], LoLo, LoLo[:,:,:,-1:]), dim=3)
            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            if ctx.needs_input_grad[{j}]:
                Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
                LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
                HiLo = coldfilt(Hi, g0b_t, g0a_t)
                HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

                deg15, deg165 = q2c(LoHi, ctx.ri_dim)
                deg45, deg135 = q2c(HiHi, ctx.ri_dim)
                deg75, deg105 = q2c(HiLo, ctx.ri_dim)
                grad_yh{j} = torch.stack(
                    [deg15, deg45, deg75, deg105, deg135, deg165], dim=ctx.o_dim)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)"""

ifm = """
class ifm{J}(Function):

    @staticmethod
    def forward(ctx, yl, {yh_in}, g0o, g1o, g0a, g0b, g1a, g1b, o_dim, ri_dim):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.o_dim = (o_dim % 6)
        ctx.ri_dim = (ri_dim % 6)
        if ctx.o_dim < ctx.ri_dim:
            ctx.ri_dim -= 1
        # Get the height and width dimensions
        if ctx.o_dim >= 3 and ctx.ri_dim >= 3:
            ctx.h_dim = 2
        elif ctx.o_dim >= 4 or ctx.ri_dim >= 4:
            ctx.h_dim = 3
        else:
            ctx.h_dim = 4
        if ctx.o_dim >= 4 and ctx.ri_dim >= 4:
            ctx.w_dim = 3
        elif ctx.o_dim >=4 or ctx.ri_dim >= 4:
            ctx.w_dim = 4
        else:
            ctx.w_dim = 5
        ll = yl
        {level2plus}{level1}

        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        {grad_yh_init}

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if True in ctx.needs_input_grad:
            {level1bwd}
            {level2plusbwd}if ctx.needs_input_grad[0]:
                grad_yl = LoLo

        return (grad_yl, {grad_yh_ret}) + (None,) * 8

"""

#  FAST_FILTS = True
# Use the above templates to create all the code for DTCWT functions with layers
import os
file = os.path.join(os.path.dirname(__file__), 'transform_funcs.py')
f = open(file, 'w')
f.write("'''This file was automatically generated by running transform_templates.py'''\n")
f.write('import torch\n')
f.write('from torch.autograd import Function\n')
f.write('from pytorch_wavelets.dtcwt.lowlevel import colfilter, rowfilter\n')
f.write('from pytorch_wavelets.dtcwt.lowlevel import coldfilt, rowdfilt\n')
f.write('from pytorch_wavelets.dtcwt.lowlevel import colifilt, rowifilt, q2c, c2q\n')
for J in range(1,8):
    # Don't do the inverse for skip hps
    yh_in = ', '.join(['yh{}'.format(j) for j in range(1,J+1)])
    level2plus = "\n\n        ".join(
        [level2plus_fwd_inv.format(
            j=j,
            checkshape=(fwd_checkshape_hps.format(j=j) if j<J else ""))
         for j in range(J,1,-1)])
    level1 = level1_hps_fwd_inv.format(checkshape=fwd_checkshape_hps.format(j=1) if J>1 else "")
    if level2plus != "":
        level1 = "\n        " + level1
    grad_yh_init = '\n        '.join(['grad_yh{} = None'.format(j) for j in
                                      range(1,J+1)])
    level1bwd = level1_bwd_inv.format(hps=(level1_hps_bwd_inv))

    level2plusbwd = "\n            ".join(
        [level2plus_bwd_inv.format(j=j) for j in range(2,J+1)])
    if level2plusbwd != '':
        level2plusbwd = level2plusbwd + '\n            '

    grad_yh_ret = ", ".join(['grad_yh{}'.format(j) for j in range(1,J+1)])

    f.write(ifm.format(
        J=J,
        yh_in=yh_in,
        level2plus=level2plus,
        level1=level1,
        grad_yh_init=grad_yh_init,
        level1bwd=level1bwd,
        level2plusbwd=level2plusbwd,
        grad_yh_ret=grad_yh_ret,
    ))

    # Do the forward transform
    for s in ('scale', ''):
        fwd_out = ", ".join(['Yh{j}'.format(j=j) for j in range(1,J+1)])
        bwd_in = ", ".join(['grad_Yh{j}'.format(j=j) for j in range(1,J+1)])
        level2plus = '\n        '.join([level2plus_fwd.format(j=j, i=j-1) for j in range(2,J+1)])
        if level2plus != '':
            level2plus = level2plus + '\n        '
        if s == '':
            fwd_lo_out = 'Yl'
            bwd_lo_in = 'grad_LoLo'
            bwd_lo_init = 'll = grad_LoLo'
            level2plusbwd = '\n            '.join(
                [level2plus_bwd.format(j=j, i=j-1, checkshape=(bwd_checkshape_hps.format(j2=j-1)))
                 for j in range(J,1,-1)])
            include_scale = ''
            include_scale2 = ''
        else:
            fwd_lo_out = ', '.join(['Ys{j}'.format(j=j) for j in range(1,J+1)])
            bwd_lo_in = ', '.join(['grad_Ys{j}'.format(j=j) for j in range(1,J+1)])
            bwd_lo_init = '''ll = grad_Ys{j}'''.format(j=J)
            {bwd_lo_init}
            level2plusbwd = '\n            '.join(
                [level2plus_bwd_scale.format(j=j, i=j-1,i2=j-2,j2=j-1, checkshape=(bwd_checkshape_hps.format(j2=j-1)))
                 for j in range(J,1,-1)])
            include_scale = ', include_scale'
            include_scale2 = 'ctx.include_scale = include_scale'
        if level2plusbwd != '':
            level2plusbwd = '\n            ' + level2plusbwd

        f.write(xfm.format(
            scale=s,
            level1=level1_fwd,
            level2plus=level2plus,
            fwd_lo_out=fwd_lo_out,
            fwd_out=fwd_out,
            bwd_in=bwd_in,
            bwd_lo_in=bwd_lo_in,
            bwd_lo_init=bwd_lo_init,
            level1bwd=(level1_hps_bwd),
            level2plusbwd=level2plusbwd,
            J=J))

f.close()
