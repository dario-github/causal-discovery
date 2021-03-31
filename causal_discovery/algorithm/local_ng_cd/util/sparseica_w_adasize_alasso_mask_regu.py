import logging
import numpy as np
import cupy as cp
from pydantic import BaseModel
from causal_discovery.algorithm.local_ng_cd.util.natural_grad_adasize_mask_regu import natural_grad_Adasize_Mask_regu
from causal_discovery.algorithm.local_ng_cd.util.estim_beta_pham import estim_beta_pham
from causal_discovery.algorithm.local_ng_cd.util.adaptive_size import adaptive_size
from causal_discovery.algorithm.local_ng_cd.util.pdinv import user_inv
from causal_discovery.parameter.algo import SparseicaWAdasizeALassoMaskRegu
from typing import Optional, List


class ICAModel(BaseModel):
    param = SparseicaWAdasizeALassoMaskRegu()
    mu = param.mu  # learning rate
    m = param.m  # for approximate the derivative of |.|; 60 40
    itmax = param.itmax
    tol = param.tol

    beta = 0
    var_num = 0
    sample_size = 0
    num_edges = 0
    refine = True
    regu = 0
    lambda_param: float = 0
    stagnation_limit = param.stagnation_limit
    omega: Optional[List[float]] = None
    grad_new: Optional[List[float]] = None
    mask: Optional[List[float]] = None
    eta: Optional[List[float]] = None
    w: Optional[List[float]] = None
    ww: Optional[List[float]] = None
    w_old: Optional[List[float]] = None
    z: Optional[List[float]] = None


def sparseica_W_adasize_Alasso_mask_regu(x, mask, lambda_param, regu):
    """
        ICA with SCAD penalized entries of the de-mixing matrix
    """
    xp = cp.get_array_module(x)

    def initialization(new_x, ica_model: ICAModel):
        logging.info("Initialization....")
        # w_temp = xp.diag(1 / xp.std(new_x, axis=1, ddof=1))
        w_temp = natural_grad_Adasize_Mask_regu(new_x, ica_model.mask, ica_model.regu)
        omega_temp = xp.array([1 / abs(w_temp[ica_model.mask != 0])])
        # to avoid instability
        upper = 3 * xp.mean(omega_temp, axis=(0, 1))
        omega_temp = (omega_temp > upper) * upper + omega_temp * (omega_temp <= upper)

        omega = np.zeros((ica_model.var_num, ica_model.var_num))
        omega[np.where(cp.asnumpy(mask) != 0)] = cp.asnumpy(omega_temp)
        ica_model.omega = xp.asarray(omega)
        ica_model.w = w_temp
        ica_model.ww = w_temp

        ica_model.z = xp.zeros((ica_model.var_num, ica_model.var_num))
        ica_model.eta = ica_model.mu * xp.ones(ica_model.w.shape)
        ica_model.w_old = ica_model.w + xp.eye(ica_model.var_num)
        ica_model.grad_new = ica_model.w_old

    def penalization(x, new_x, std_x, ica_model: ICAModel):
        """
            add
        """
        logging.info("Starting penalization...")
        y_psi = xp.zeros((ica_model.var_num, ica_model.sample_size))
        # y_psi = xp.zeros((ica_model.var_num, tem.shape[1]))
        y_psi_0 = {}
        grad_old = None
        min_check = 1
        stagnation_cnt = 0
        regu_l1 = ica_model.regu / 2.0
        for i in range(0, ica_model.itmax):
            y = ica_model.w @ new_x
            # 对grad_new求平均
            check_value = (
                xp.sum(abs(ica_model.grad_new * ica_model.mask), axis=(0, 1))
                / ica_model.num_edges
            )
            if not i % 100:
                logging.info(
                    f"penalization train step: {i}, value: {check_value}, min_check: {min_check}, target: {ica_model.tol}"
                )
            if abs(check_value) < ica_model.tol:
                if ica_model.refine:
                    # ica_model.mask = (abs(ica_model.w) > 0.01)
                    ica_model.mask = xp.ones(ica_model.w.shape, dtype=int) * (
                        abs(ica_model.w) > 0.01
                    )
                    ica_model.mask = ica_model.mask - xp.diag(xp.diag(ica_model.mask))
                    ica_model.lambda_param = 0
                    ica_model.refine = False
                else:
                    logging.info(
                        f"last penalization train step: {i}, value: {check_value}, min_check: {min_check}, target: {ica_model.tol}"
                    )
                    break
            # 梯度连续没有下降，记次数
            if min_check < abs(check_value):
                stagnation_cnt += 1
            else:
                min_check = abs(check_value)
                stagnation_cnt = 0
            # 连续n次没有下降，停止
            if stagnation_cnt > ica_model.stagnation_limit:
                logging.info(
                    f"early stopping: {i}, value: {check_value}, min_check: {min_check}, target: {ica_model.tol}"
                )
                break
            ica_model.w_old = ica_model.w
            # update W: linear ICA with marginal score function estimated from data...
            if i % 8 == 0:
                for j in range(ica_model.var_num):
                    tem = estim_beta_pham(y[[j], :])
                    y_psi[j] = tem[0, :]
                    idxs = xp.argsort(y[j, :])
                    y_psi_0[j] = y_psi[j, idxs]
            else:
                for j in range(ica_model.var_num):
                    idxs = xp.argsort(y[j, :])
                    y_psi[j, idxs] = y_psi_0[j]

            dev = ica_model.omega * xp.tanh(ica_model.m * ica_model.w)
            # with additional regularization
            # y_psi*x'/T + inv(W') -4*beta* (diag(diag(y*y'/T)) - eye(N)) * (y*x'/T) - dev*lambda/T  - 2*regu_l1 * W
            ica_model.grad_new = (
                y_psi @ x.conj().T / ica_model.sample_size
                # + xp.linalg.inv(ica_model.w.conj().T)
                + user_inv(ica_model.w.conj().T)
                - 4 * ica_model.beta
                * (
                    xp.diag(xp.diag(y @ y.conj().T / ica_model.sample_size))
                    - xp.eye(ica_model.var_num)
                )
                @ (y @ x.conj().T / ica_model.sample_size)
                - dev * ica_model.lambda_param / ica_model.sample_size
                - 2 * regu_l1 * ica_model.w
            )
            if i == 0:
                grad_old = ica_model.grad_new
            ica_model.eta, ica_model.z = adaptive_size(
                ica_model.grad_new, grad_old, ica_model.eta, ica_model.z
            )
            ica_model.w = (
                ica_model.w + 0.9 * ica_model.eta * ica_model.z * ica_model.mask
            )
            grad_old = ica_model.grad_new

        # re-scaling
        ica_model.w = xp.diag(std_x) @ ica_model.w @ xp.diag(1 / std_x)
        ica_model.ww = xp.diag(std_x) @ ica_model.ww @ xp.diag(1 / std_x)
        y = xp.diag(std_x) @ y
        score = ica_model.omega * abs(ica_model.w)
        return y, ica_model.w, ica_model.ww, score

    ica_model = ICAModel()
    ica_model.mask = mask
    ica_model.regu = regu
    ica_model.lambda_param = lambda_param
    ica_model.var_num, ica_model.sample_size = x.shape
    # 居中
    new_x = x - xp.mean(xp.array([x]), axis=2).T @ xp.ones((1, ica_model.sample_size))
    # To avoid instability
    std_x = xp.std(new_x, ddof=1, axis=1)
    new_x = xp.diag(1 / std_x) @ x  # 归一化操作
    ica_model.num_edges = xp.sum(mask, axis=(0, 1))

    initialization(new_x, ica_model)
    return penalization(x, new_x, std_x, ica_model)


if __name__ == "__main__":
    from peon.log import config_log
    config_log(
        "local_ng_cd", "local_ng_cd", log_root="/root/logs", print_terminal=True, enable_monitor=True,
    )
    sparseica_W_adasize_Alasso_mask_regu(
        cp.random.rand(50, 500),
        cp.ones((50, 50)) - cp.eye(50),
        8,
        1e-3,
    )
