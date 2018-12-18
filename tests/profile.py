import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import argparse
import py3nvml
import torch.nn.functional as F

parser = argparse.ArgumentParser(
    'Profile the forward and inverse dtcwt in pytorch')
parser.add_argument('--no_grad', action='store_true',
                    help='Dont calculate the gradients')
parser.add_argument('--ref', action='store_true',
                    help='Compare to doing the DTCWT with ffts')
parser.add_argument('-c', '--convolution', action='store_true',
                    help='Profile an 11x11 convolution')
parser.add_argument('-f', '--forward', action='store_true',
                    help='Only do forward transform (default is fwd and inv)')
parser.add_argument('-i', '--inverse', action='store_true',
                    help='Only do inverse transform (default is fwd and inv)')
parser.add_argument('-j', type=int, default=2,
                    help='number of scales of transform to do')
parser.add_argument('--no_hp', action='store_true')
parser.add_argument('-s', '--size', default=0, type=int,
                    help='spatial size of input')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='which device to test')
parser.add_argument('--batch', default=16, type=int,
                    help='Number of images in parallel')


def forward(size, no_grad, J, no_hp=False, dev='cuda'):
    x = torch.randn(*size, requires_grad=(not no_grad)).to(dev)
    xfm = DTCWTForward(J=J, skip_hps=no_hp).to(dev)
    Yl, Yh = xfm(x)
    if not no_grad:
        Yl.backward(torch.ones_like(Yl))


def inverse(size, no_grad, J, no_hp=False, dev='cuda'):
    yl = torch.randn(size[0], size[1], size[2] >> (J-1), size[3] >> (J-1),
                     requires_grad=(not no_grad)).to(dev)
    yh = [torch.randn(size[0], size[1], 6, size[2] >> j, size[3] >> j, 2,
                      requires_grad=(not no_grad)).to(dev)
          for j in range(1,J+1)]
    ifm = DTCWTInverse(J=J).to(dev)
    Y = ifm((yl, yh))
    if not no_grad:
        Y.backward(torch.ones_like(Y))
    return Y


def end_to_end(size, no_grad, J, no_hp=False, dev='cuda'):
    x = torch.randn(*size, requires_grad=(not no_grad)).to(dev)
    xfm = DTCWTForward(J=J, skip_hps=no_hp).to(dev)
    ifm = DTCWTInverse(J=J).to(dev)
    Yl, Yh = xfm(x)
    Y = ifm((Yl, Yh))
    if not no_grad:
        Y.backward(torch.ones_like(Y))
    return Y


def reference_conv(size, no_grad=False, dev='cuda'):
    x = torch.randn(*size, requires_grad=(not no_grad)).to(dev)
    w = torch.randn(10,1,11,11).to(dev)
    y = F.conv2d(x, w, padding=5, groups=10)
    if not no_grad:
        y.backward(torch.ones_like(y))
    return y


def reference_fftconv(size, J, no_grad=False, dev='cuda'):
    x = torch.randn(*size, requires_grad=(not no_grad)).to(dev)
    # Make the rough assumption that the wavelet sizes are 9*J spatial size
    sz = 9*J
    xp = F.pad(x, (0, sz-1, 0, sz-1))
    FX = torch.rfft(xp, 2)
    FX = torch.unsqueeze(FX, dim=2)
    FW = torch.randn(1, 1, 12*J+1, FX.shape[-3], FX.shape[-2], 2, device=dev)
    FYr = FX[..., 0] * FW[..., 0] - FX[..., 1] * FW[..., 1]
    FYi = FX[..., 0] * FW[..., 1] + FX[..., 1] * FW[..., 0]
    FY = torch.stack((FYr, FYi), dim=-1)
    FY = FY.view(FY.shape[0], -1, *FY.shape[-3:])
    Y = torch.irfft(FY, 2, signal_sizes=xp.shape[-2:])
    if not no_grad:
        Y.backward(torch.ones_like(Y))


if __name__ == "__main__":
    args = parser.parse_args()
    py3nvml.grab_gpus(1)
    if args.size > 0:
        size = (args.batch, 1, args.size, args.size)
    else:
        size = (args.batch, 1, 128, 128)

    if args.ref:
        print('Running dtcwt with FFTs')
        reference_fftconv(size, args.j, args.no_grad, args.device)
    elif args.convolution:
        print('Running 11x11 convolution')
        reference_conv(size, args.no_grad, args.device)
    else:
        if args.forward:
            print('Running forward transform')
            forward(size, args.no_grad, args.j, args.no_hp, args.device)
        elif args.inverse:
            print('Running inverse transform')
            inverse(size, args.no_grad, args.j, args.no_hp, args.device)
        else:
            print('Running end to end')
            end_to_end(size, args.no_grad, args.j, args.no_hp, args.device)
