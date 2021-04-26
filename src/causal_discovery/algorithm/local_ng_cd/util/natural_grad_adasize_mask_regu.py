import logging

import numpy as np

from causal_discovery.algorithm.local_ng_cd.util.adaptive_size import adaptive_size
from causal_discovery.algorithm.local_ng_cd.util.estim_beta_pham import estim_beta_pham
from causal_discovery.algorithm.local_ng_cd.util.pdinv import pdinv, user_inv
from causal_discovery.parameter.algo import NaturalGradAdasizeMaskRegu
from causal_discovery.parameter.env import select_xp

xp = select_xp()


def natural_grad_Adasize_Mask_regu(x, mask, regu, ww=None):
    param = NaturalGradAdasizeMaskRegu()
    var_num, sample_size = x.shape
    mu = param.mu
    itmax = param.itmax
    tol = param.tol
    early_stopping_times = param.early_stopping_times
    y_psi0 = {}
    grad_w_0 = 0
    num_edges = xp.sum(mask, axis=(0, 1))

    # initilization of W
    if not ww:
        ww = xp.eye(var_num)
        for iter_ in range(var_num):
            idx_i = xp.flatnonzero(mask[iter_, :])
            ww[iter_, idx_i] = (
                -0.5
                * (x[iter_, :] @ x[idx_i, :].conj().T)
                @ pdinv(x[idx_i, :] @ x[idx_i, :].conj().T)
            )
        w = 0.4 * (ww + ww.conj().T)
        w = w + 0.1 * xp.random.normal(0, 1, w.shape)
        w = w + xp.diag(xp.array([1] * var_num) - xp.diag(w))
    else:
        w = ww

    z = xp.zeros((var_num, var_num))
    eta = mu * xp.ones(w.shape)
    y_psi = xp.zeros((var_num, sample_size))
    logging.info("begin natural grad adasize mask regu")
    early_stopping_cnt = 0
    check_min = 1
    for iter_ in range(itmax):
        y = w @ x
        # update W: linear ICA with marginal score function estimated from data...
        if iter_ % 12 == 0:
            for i in range(var_num):
                tem = estim_beta_pham(y[[i], :])
                y_psi[i, :] = tem[0, :]
                idxs = xp.argsort(y[i, :])
                y_psi0[i] = y_psi[i, idxs]
        else:
            for i in range(var_num):
                idxs = xp.argsort(y[i, :])
                y_psi[i, idxs] = y_psi0[i]
        # with regularization to make W small
        # grad_w_n = y_psi @ x.T / sample_size + xp.linalg.inv(w.conj().T) - 2 * regu * w
        grad_w_n = y_psi @ x.T / sample_size + user_inv(w.conj().T) - 2 * regu * w
        if iter_ == 0:
            grad_w_0 = grad_w_n
        eta, z = adaptive_size(grad_w_n, grad_w_0, eta, z)
        w = w + eta * z * mask
        check_value = xp.sum(abs(grad_w_n * mask), axis=(0, 1)) / num_edges
        check_min = min(check_value, check_min) if check_value > 0 else check_min
        # if not iter_ % (2 * early_stopping_window):
        if check_value > check_min:
            early_stopping_cnt += 1
            if early_stopping_cnt > early_stopping_times:
                logging.info(f"early stopping, value: {check_value}, target: {tol}")
                break
        else:
            early_stopping_cnt = 0  # 只有连续n次不下降，才停止
        if not iter_ % 100:
            logging.info(
                f"grad step: {iter_}/{itmax}, value: {check_value}, min: {check_min}, target: {tol}"
            )
            # 不再下降时提前停止
        if check_value < tol:
            break
        grad_w_0 = grad_w_n
    logging.info("end natural grad adasize mask regu")
    return w


if __name__ == "__main__":
    print(
        natural_grad_Adasize_Mask_regu(
            np.array([[1, 2], [2, 1]]), np.ones((2, 2)) - np.eye(2), 1e-3
        )
    )
