import logging
import numpy as np
import cupy as cp
from causal_discovery.algorithm.local_ng_cd.util.scorecond import scorecond


def estim_beta_pham(x):
    xp = cp.get_array_module(x)
    t1, t2 = x.shape
    if t1 > t2:
        logging.error(
            "error in eaastim_beta_pham(x): data must be organized in x in a row fashion"
        )
        return None

    return xp.array([-scorecond(x.T)[0].T[0, :], -scorecond(x[::-1, :].T)[0].T[0, :]])


if __name__ == "__main__":
    print("=" * 20)
    print(estim_beta_pham(np.array([1, 2, 3, 4, 5], ndmin=2)))