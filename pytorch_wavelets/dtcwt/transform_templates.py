from torch.autograd import Function

## Forward Templates

level1_fwd = """# Level 1 forward (biorthogonal analysis filters)
        Lo = rowfilter(input, h0o)
        if not ctx.skip_hps[0]:
            Hi = rowfilter(input, h1o)
            LoHi = colfilter(Lo, h1o)
            HiLo = colfilter(Hi, h0o)
            HiHi = colfilter(Hi, h1o)
            deg15, deg165 = q2c(LoHi)
            deg45, deg135 = q2c(HiHi)
            deg75, deg105 = q2c(HiLo)
            Yh1 = torch.stack(
                [deg15, deg45, deg75, deg105, deg135, deg165], dim=2)
        else:
            Yh1 = torch.tensor([])
        LoLo = colfilter(Lo, h0o)
        """

level1_hps_bwd = """# Level 1 backward (time reversed biorthogonal analysis filters)
            if not ctx.skip_hps[0]:
                lh = c2q(grad_Yh1[:,:,0:6:5])
                hl = c2q(grad_Yh1[:,:,2:4:1])
                hh = c2q(grad_Yh1[:,:,1:5:3])
                Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
                Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
                grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)
            else:
                grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)
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

            deg15, deg165 = q2c(LoHi)
            deg45, deg135 = q2c(HiHi)
            deg75, deg105 = q2c(HiLo)
            Yh{j} = torch.stack(
                [deg15, deg45, deg75, deg105, deg135, deg165], dim=2)
        else:
            Yh{j} = torch.tensor([])
        LoLo = coldfilt(Lo, h0b, h0a)
"""

level2plus_bwd = """# Level {j} backward (time reversed quater shift analysis filters)
            if not ctx.skip_hps[{j}]:
                lh = c2q(grad_Yh{j}[:,:,0:6:5])
                hl = c2q(grad_Yh{j}[:,:,2:4:1])
                hh = c2q(grad_Yh{j}[:,:,1:5:3])
                Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
                Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
                ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            else:
                ll = rowifilt(colifilt(Lo, h0b_t, h0a_t), h0b_t, h0a_t)
            {checkshape}
"""
bwd_checkshape_hps = """r, c = ll.shape[2:]
            r1, c1 = grad_Yh{j2}.shape[3:5]
            if r != r1 * 2:
                ll = ll[:,:,1:-1]
            if c != c1 * 2:
                ll = ll[:,:,:,1:-1]"""
bwd_checkshape_nohps = """r, c = ll.shape[2:]
            r1, c1 = in_shape[3:5]
            if r != r1 * 2:
                ll = ll[:,:,1:-1]
            if c != c1 * 2:
                ll = ll[:,:,:,1:-1]"""

xfm = """
class xfm{J}(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b, skip_hps):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.in_shape = input.shape
        ctx.skip_hps = skip_hps
        batch, ch, r, c = input.shape

        # If the row/col count of X is not divisible by 2 then we need to
        # extend X
        if r % 2 != 0:
            input = torch.cat((input, input[:,:,-1:]), dim=2)
        if c % 2 != 0:
            input = torch.cat((input, input[:,:,:,-1:]), dim=3)

        {level1}
        {level2plus}Yl = LoLo
        return Yl, {fwd_out}

    @staticmethod
    def backward(ctx, grad_LoLo, {bwd_in}):
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
            ll = grad_LoLo{level2plusbwd}
            {level1bwd}

        return (grad_input,) + (None,) * 7

"""

## Inverse Templates
level1_hps_fwd_inv = """# Level 1 inverse with biorthogonal synthesis filters
        if yh1 is not None and yh1.shape != torch.Size([0]):
            {checkshape}lh = c2q(yh1[:,:,0:6:5])
            hl = c2q(yh1[:,:,2:4:1])
            hh = c2q(yh1[:,:,1:5:3])
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
                deg15, deg165 = q2c(LoHi)
                deg45, deg135 = q2c(HiHi)
                deg75, deg105 = q2c(HiLo)
                grad_yh1 = torch.stack(
                    [deg15, deg45, deg75, deg105, deg135, deg165], dim=2)
"""

level2plus_fwd_inv = """# Level {j} inverse transform with quater shift synthesis filters
        if yh{j} is not None and yh{j}.shape != torch.Size([0]):
            {checkshape}lh = c2q(yh{j}[:,:,0:6:5])
            hl = c2q(yh{j}[:,:,2:4:1])
            hh = c2q(yh{j}[:,:,1:5:3])
            Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
            if ll is not None and ll.shape != torch.Size([0]):
                Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
            else:
                Lo = colifilt(lh, g1b, g1a, True)
            ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        else:
            ll = rowifilt(colifilt(Lo, g0b, g0a), g0b, g0a)
        """

fwd_checkshape_hps = """r, c = ll.shape[2:]
            r1, c1 = yh{j}.shape[3:5]
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
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            if ctx.needs_input_grad[{j}]:
                Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
                LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
                HiLo = coldfilt(Hi, g0b_t, g0a_t)
                HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

                deg15, deg165 = q2c(LoHi)
                deg45, deg135 = q2c(HiHi)
                deg75, deg105 = q2c(HiLo)
                grad_yh{j} = torch.stack(
                    [deg15, deg45, deg75, deg105, deg135, deg165], dim=2)"""

ifm = """
class ifm{J}(Function):

    @staticmethod
    def forward(ctx, yl, {yh_in}, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
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
            {level2plusbwd}grad_yl = LoLo

        return grad_yl, {grad_yh_ret}, None, None, None, None, None, None

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
    fwd_out = ", ".join(
        ['Yh{j}'.format(j=j) for j in range(1,J+1)])
    bwd_in = ", ".join(
        ['grad_Yh{j}'.format(j=j) for j in range(1,J+1)])
    level2plus = '\n        '.join(
        [level2plus_fwd.format(j=j, i=j-1) for j in range(2,J+1)])
    if level2plus != '':
        level2plus = level2plus + '\n        '
    level2plusbwd = '\n            '.join(
        [level2plus_bwd.format(
            j=j,
            checkshape=(bwd_checkshape_hps.format(j2=j-1)))
         for j in range(J,1,-1)])
    if level2plusbwd != '':
        level2plusbwd = '\n            ' + level2plusbwd

    f.write(xfm.format(
        level1=level1_fwd,
        level2plus=level2plus,
        fwd_out=fwd_out,
        bwd_in=bwd_in,
        level1bwd=(level1_hps_bwd),
        level2plusbwd=level2plusbwd,
        J=J))

f.close()
