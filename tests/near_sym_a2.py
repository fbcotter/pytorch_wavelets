# coding: utf-8
h0, g0, h1, g1 = level1('near_sym_a', True)
h0 = np.array([[0, -.05, 0.25, 0.5, 0.25, -0.05, 0]]).T
g1 = np.array([[0 -.05, -.25, .6, -.25, -.05, 0]]).T
h0 *= 1/np.sqrt(np.sum(h0**2))
h1 *= 1/np.sqrt(np.sum(h1**2))
g0 *= 1/np.sqrt(np.sum(g0**2))
g1 *= 1/np.sqrt(np.sum(g1**2))
h0a = np.concatenate((h0, [[0]]), axis=0)
h1a = np.concatenate(([[0]], h1), axis=0)
h0b = np.concatenate(([[0]], h0), axis=0)
h1b = np.concatenate((h1, [[0]]), axis=0)
g0a = np.concatenate(([[0]], g0), axis=0)
g1a = np.concatenate((g1, [[0]]), axis=0)
g0b = np.concatenate((g0, [[0]]), axis=0)
g1b = np.concatenate(([[0]], g1), axis=0)
np.savez('/scratch/fbc23/repos/fbcotter/pytorch_wavelets/pytorch_wavelets/dtcwt/data/near_sym_a2.npz', h0a=h0a, h1a=h1a, h0b=h0b, h1b=h1b, g0a=g0a, g1a=g1a, g0b=g0b,g1b=g1b)
