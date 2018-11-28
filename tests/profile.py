import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import argparse
import py3nvml
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--forward', action='store_true')
parser.add_argument('-b', '--backward', action='store_true')
parser.add_argument('-j', type=int, default=2)
parser.add_argument('-c', '--convolution', action='store_true')
parser.add_argument('--no_hp', action='store_true')
parser.add_argument('-s', '--size', default=0, type=int, help='spatial size of input')
parser.add_argument('--reference', action='store_true')

size = (10, 10, 128, 128)


def forward(J, no_hp=False):
    x = torch.randn(*size).cuda()
    xfm = DTCWTForward(J=J, skip_hps=no_hp).cuda()
    Yl, Yh = xfm(x)


def inverse(J, no_hp=False):
    yl = torch.randn(size[0], size[1], size[2]>>(J-1), size[3]>>(J-1)).cuda()
    yh = [torch.randn(size[0], size[1], 6, size[2]>>j, size[3]>>j, 2).cuda() for j in
          range(1,J+1)]
    ifm = DTCWTInverse(J=J).cuda()
    Y = ifm((yl, yh))


def end_to_end(J, no_hp=False):
    x = torch.randn(*size).cuda()
    xfm = DTCWTForward(J=J, skip_hps=no_hp).cuda()
    ifm = DTCWTInverse(J=J).cuda()
    Yl, Yh = xfm(x)
    Y = ifm((Yl, Yh))


def reference():
    x = torch.randn(*size).cuda()
    w = torch.randn(10,1,11,11).cuda()
    y = F.conv2d(x, w, padding=5, groups=10)
    print(y.shape)


if __name__ == "__main__":
    args = parser.parse_args()
    py3nvml.grab_gpus(1)
    if args.size > 0:
        size = (10, 10, args.size, args.size)
    if args.forward:
        print('Running forward transform')
        forward(args.j, args.no_hp)
    elif args.backward:
        print('Running backward transform')
        inverse(args.j, args.no_hp)
    elif args.reference:
        print('Running reference')
        reference()
    else:
        print('Running end to end')
        end_to_end(args.j, args.no_hp)

