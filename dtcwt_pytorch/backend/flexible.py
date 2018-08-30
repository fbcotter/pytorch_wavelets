class DTCWTForward(nn.Module):
    def __init__(self, C, biort='near_sym_a', qshift='qshift_a',
                 J=3, in_channels=1, skip_hps=None):
        super().__init__()
        self.C = C
        self.biort = biort
        self.qshift = qshift
        self.skip_hps = skip_hps
        self.J = J
        h0o, g0o, h1o, g1o = _biort(biort)
        self.h0o = torch.nn.Parameter(prep_filt(h0o, in_channels), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, in_channels), False)
        self.g0o = torch.nn.Parameter(prep_filt(g0o, in_channels), False)
        self.g1o = torch.nn.Parameter(prep_filt(g1o, in_channels), False)
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        self.h0a = torch.nn.Parameter(prep_filt(h0a, in_channels), False)
        self.h0b = torch.nn.Parameter(prep_filt(h0b, in_channels), False)
        self.h1a = torch.nn.Parameter(prep_filt(h1a, in_channels), False)
        self.h1b = torch.nn.Parameter(prep_filt(h1b, in_channels), False)
        self.g0a = torch.nn.Parameter(prep_filt(g0a, in_channels), False)
        self.g0b = torch.nn.Parameter(prep_filt(g0b, in_channels), False)
        self.g1a = torch.nn.Parameter(prep_filt(g1a, in_channels), False)
        self.g1b = torch.nn.Parameter(prep_filt(g1b, in_channels), False)

        # Create the function to do the DTCWT
        if skip_hps:
            J1 = 2
        else:
            J1 = 1
        fwd_out = ", ".join(
            ['Yhr{j}, Yhi{j}'.format(j=j) for j in range(J1,J)])
        bwd_in = ", ".join(
            ['grad_Yhr{j}, grad_Yhi{j}'.format(j=j) for j in range(J1,J)])
        level2plus = '\n'.join(
            [tt.level2plus_fwd.format(j=j) for j in range(J1,J)])
        level2plusbwd = '\n'.join(
            [tt.level2plus_bwd.format(j=j) for j in range(J-1,J1-1,-1)])

        self.fwd_function = eval(tt.xfm.format(
            level1=tt.level1_fwd.format(skip_hps=skip_hps),
            level2plus=level2plus,
            fwd_out=fwd_out,
            bwd_in=bwd_in,
            level1_bwd=tt.level1_bwd.format(skip_hps=skip_hps),
            level2plus_bwd=level2plusbwd))

    def forward(self, input):
        return self.fwd_function(input)

