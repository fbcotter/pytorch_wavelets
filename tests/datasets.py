import os
import numpy as np

def regframes(name):
    """Load the *name* registration dataset and return source and reference frame."""
    frames = np.load(os.path.join(os.path.dirname(__file__), name + '.npz'))
    return frames['f1'], frames['f2']

def mandrill():
    """Return the "mandrill" test image."""
    return np.load(os.path.join(os.path.dirname(__file__), 'mandrill.npz'))['mandrill']

def barbara():
    """Return the "barbara" test image."""
    return np.load(os.path.join(os.path.dirname(__file__), 'barbara.npz'))['barbara']
