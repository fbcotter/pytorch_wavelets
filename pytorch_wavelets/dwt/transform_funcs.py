import torch
from torch.autograd import Function
from .lowlevel import rowdfilt, coldfilt, rowifilt, colifilt


class WT(Function):
    @staticmethod
    def forward(ctx, x, h0, h1):
        ctx.save_for_backward(h0, h1)

        Lo = rowdfilt(x, h0)
        LoLo = coldfilt(Lo, h0)
        Hi = rowdfilt(x, h1)
        LoHi = coldfilt(Lo, h1)
        HiLo = coldfilt(Hi, h0)
        HiHi = coldfilt(Hi, h1)

        return LoLo, torch.stack((LoHi, HiHi, HiLo), dim=2)

    @staticmethod
    def backward(ctx, grad_y):
        return None, None, None


class IWT(Function):
    @staticmethod
    def forward(ctx, yl, yh, g0, g1):
        ctx.save_for_backward(g0, g1)

        ll = yl
        lh, hh, hl = yh[:,:,0], yh[:,:,1], yh[:,:,2]
        Hi = colifilt(hh, g1) + colifilt(hl, g0)
        Lo = colifilt(lh, g1) + colifilt(ll, g0)
        ll = rowifilt(Hi, g1) + rowifilt(Lo, g0)

        return ll

    @staticmethod
    def backward(ctx, grad_y):
        return None, None, None

class WT2(Function):
    @staticmethod
    def forward(ctx, x, h0, h1):
        ctx.save_for_backward(h0, h1)

        Lo = rowdfilt(x, h0,)
        LoLo = coldfilt(Lo, h0)
        Hi = rowdfilt(x, h1)
        LoHi = coldfilt(Lo, h1)
        HiLo = coldfilt(Hi, h0)
        HiHi = coldfilt(Hi, h1)

        return LoLo, torch.stack((LoHi, HiHi, HiLo), dim=2)

    @staticmethod
    def backward(ctx, grad_y):
        return None, None, None
