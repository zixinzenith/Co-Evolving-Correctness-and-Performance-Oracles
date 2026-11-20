"""Scripted fake models for end-to-end tests of the loop — no API key needed.

`good` answers correctly on iteration 1 (its kernel is a hand-copy of the reference
computation). `buggy_then_good` plants one documented bug axis on the first attempt and
answers correctly afterwards — this exercises the unit-test feedback path. `always_buggy`
never recovers and exhausts the iteration budget.
"""

import numpy as np


def good(prompt, history):
    return '''
import numpy as np

def kernel(**inp):
    q, s, z = inp['w'], inp['scales'], inp['zeros']
    g = q.shape[0] // s.shape[0]
    s = np.repeat(s, g, axis=0)
    z = np.repeat(z, g, axis=0)
    out = np.clip(np.rint((q + z * s) / s), 0, 15)
    return {'q': out.astype(np.int64)}
'''


def buggy_then_good(prompt, history):
    if not history:
        return '''
import numpy as np

def kernel(**inp):
    q, s, z = inp['w'], inp['scales'], inp['zeros']
    g = q.shape[0] // s.shape[0]
    s = np.repeat(s, g, axis=0)
    z = np.repeat(z, g, axis=0)
    out = np.clip(np.floor((q + z * s) / s + 0.5), 0, 15)  # half-up, not half-even
    return {'q': out.astype(np.int64)}
'''
    return good(prompt, history)


def always_buggy(prompt, history):
    return buggy_then_good(prompt, [])  # perpetually half-up
