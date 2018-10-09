# coding: utf-8
import torch
import torch.nn as nn
from dtcwtnet2.dtcwt import SparsifyWaveCoeffs2
from pytorch_wavelets import DTCWTForward, DTCWTInverse
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.xfm = DTCWTForward(J=3,C=3)
        self.ifm = DTCWTInverse(J=3,C=3)
        self.sparsify = SparsifyWaveCoeffs2(3,3)
    def forward(self, x):
        coeffs = self.xfm(x)
        coeffs = self.sparsify(coeffs)
        y = self.ifm(coeffs)
        return y

net = Net()
X = 100 * torch.randn(5, 3, 32, 32, dtype=torch.float)
X_noise = X + 5*torch.randn(5,3,32,32, dtype=torch.float)
y = net(X_noise)

criterion = torch.nn.MSELoss()
loss = criterion(y, X)
loss.backward()
