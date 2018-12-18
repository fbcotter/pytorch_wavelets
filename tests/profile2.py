import torch
import argparse
import py3nvml
import timeit

parser = argparse.ArgumentParser('Profile the dwt')
parser.add_argument('method', choices=['torch', 'numpy'],
                    help='Method to use to calculate dwt')
parser.add_argument('xfm', choices=['dwt', 'dtcwt'],
                    help='which transform to use')
parser.add_argument('-f', '--forward', action='store_true',
                    help='Only do forward transform (default is fwd and inv)')
parser.add_argument('-j', type=int, default=2,
                    help='number of scales of transform to do')
parser.add_argument('-s', '--size', default=0, type=int,
                    help='spatial size of input')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='which device to test')
parser.add_argument('--wave', default='db4',
                    help='which wavelet to use')
parser.add_argument('--batch', default=16, type=int,
                    help='Number of images in parallel')

if __name__ == "__main__":
    args = parser.parse_args()
    py3nvml.grab_gpus(1)
    if args.size > 0:
        size = (args.batch, 1, args.size, args.size)
    else:
        size = (args.batch, 1, 128, 128)

    if args.method == 'torch':
        if args.xfm == 'dwt':
            t = timeit.Timer('ifm(xfm(x))',
                             setup="""
import torch
from pytorch_wavelets import DWT, IDWT
x = torch.randn(*{sz}).to('{dev}')
xfm = DWT(J={J}, wave='{wave}').to('{dev}')
ifm = IDWT(wave='{wave}').to('{dev}')""".format(sz=size, dev=args.device, J=args.j,
                                                wave=args.wave))
            print('5 run average is {:.3f}s'.format(t.timeit(number=5)/5))
        else:
            t = timeit.Timer('ifm(xfm(x))',
                             setup="""
import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse
x = torch.randn(*{sz}).to('{dev}')
xfm = DTCWTForward(J={J}).to('{dev}')
ifm = DTCWTInverse(J={J}).to('{dev}')""".format(sz=size, dev=args.device, J=args.j))
            print('5 run average is {:.3f}s'.format(t.timeit(number=5)/5))
    else:
        if args.xfm == 'dwt':
            t = timeit.Timer('ifm(xfm(x))',
                             setup="""
import numpy as np
import pywt
x = np.random.randn(*{sz})
xfm = lambda a: pywt.wavedec2(a, '{wave}', level={J}, mode='reflect')
ifm = lambda a: pywt.waverec2(a, '{wave}', mode='reflect')
                                 """.format(sz=size, wave=args.wave, J=args.j))
            print('5 run average is {:.3f}s'.format(t.timeit(number=5)/5))
        else:
            t = timeit.Timer("""
for b in x:
    for c in b:
        xfm.inverse(xfm.forward(c, nlevels={J}))
""".format(J=args.j), setup="""
import numpy as np
import dtcwt
x = np.random.randn(*{sz})
xfm = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_a')
                             """.format(sz=size))
            print('5 run average is {:.3f}s'.format(t.timeit(number=5)/5))
    #  end_to_end(args.method, args.wave, size, args.j, args.device)
