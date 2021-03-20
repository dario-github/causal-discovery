import numpy as np
import cupy as cp
from causal_discovery.parameter.algo import AdaptiveSize


def adaptive_size(grad_new, grad_old, eta_old, z_old):
    xp = cp.get_array_module(grad_new)
    param = AdaptiveSize()
    z = grad_new + param.alpha * z_old
    etaup = (grad_new * grad_old) >= 0

    eta = eta_old * (param.up * etaup + param.down * (1 - etaup))
    eta = xp.minimum(eta, param.eta_minimum)
    return eta, z