# implementation of different profiles for learning rate schedules

import numpy as np

profile_list = ['cos', 'rex', 'exp', 'linear']

def cos_decay(_iter, stepsize, min_val, max_val):
    assert min_val <= max_val
    z = float((_iter % stepsize)) / stepsize
    val = max_val - 0.5*float(max_val - min_val)*(1.0 - np.cos(np.pi*z))
    return val

def cos_growth(_iter, stepsize, min_val, max_val):
    assert min_val <= max_val
    z = float((_iter % stepsize)) / stepsize
    val = min_val + 0.5*float(max_val - min_val)*(1.0 - np.cos(np.pi*z))
    return val

def rex_decay(_iter, stepsize, min_val, max_val):
    assert min_val <= max_val
    mod_iter = float(_iter % stepsize)
    z = float(stepsize - mod_iter) / stepsize
    val = min_val + float(max_val - min_val) * (z / (1 - 0.9 + 0.9*z))
    return val

def rex_growth(_iter, stepsize, min_val, max_val):
    assert min_val <= max_val
    mod_iter = float(_iter % stepsize)
    z = float(stepsize - mod_iter) / stepsize
    val = max_val - float(max_val - min_val) * (z / (1 - 0.9 + 0.9*z))
    return val

def exp_decay(_iter, stepsize, min_val, max_val, exponent=4):
    assert min_val <= max_val
    z = float(_iter % stepsize) / stepsize
    val = min_val + float(max_val - min_val)*np.exp(-exponent*z)
    return val

def exp_growth(_iter, stepsize, min_val, max_val, exponent=4):
    assert min_val <= max_val
    z = float(_iter % stepsize) / stepsize
    val = max_val - float(max_val - min_val)*np.exp(-exponent*z)
    return val

def linear_decay(_iter, stepsize, min_val, max_val):
    assert min_val <= max_val
    mod_iter = float(_iter % stepsize)
    val = max_val - (float(max_val - min_val) * (float(mod_iter) / stepsize))
    return val

def linear_growth(_iter, stepsize, min_val, max_val):
    assert min_val <= max_val
    mod_iter = float(_iter % stepsize)
    val = min_val + (float(max_val - min_val) * (float(mod_iter) / stepsize))
    return val
