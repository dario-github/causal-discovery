import numpy as np
import pandas as pd
import pytest

from causal_discovery.algorithm.local_ng_cd.local_ng_cd import local_ng_cd
from causal_discovery.algorithm.local_ng_cd.util.betaAlasso_grad_2step import (
    betaAlasso_grad_2step,
)
from causal_discovery.algorithm.local_ng_cd.util.estim_beta_pham import estim_beta_pham
from causal_discovery.algorithm.local_ng_cd.util.natural_grad_adasize_mask_regu import (
    natural_grad_Adasize_Mask_regu,
)
from causal_discovery.algorithm.local_ng_cd.util.pdinv import pdinv, user_inv, user_pinv
from causal_discovery.algorithm.local_ng_cd.util.scorecond import scorecond
from causal_discovery.algorithm.local_ng_cd.util.sparseica_w_adasize_alasso_mask_regu import (
    sparseica_W_adasize_Alasso_mask_regu,
)
from causal_discovery.parameter.algo import LocalNgCdParam
from causal_discovery.parameter.env import select_xp

xp = select_xp()


def test_beta_adaptive_lasso_grad_2step():
    betaAlasso_grad_2step(
        xp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        xp.array([[3, 5, 7, 9]]),
        0.01,
        4,
    )


def test_estim_beta_pham():
    estim_beta_pham(xp.array([1, 2, 3, 4, 5], ndmin=2))


def test_natural_grad_Adasize_Mask_regu():
    natural_grad_Adasize_Mask_regu(
        xp.array([[1, 2], [2, 1]]), xp.ones((2, 2)) - xp.eye(2), 1e-3
    )


def test_pdinv():
    pdinv(xp.array([[1, 2], [1, 1]]))
    user_inv(xp.array([[1, 2], [1, 1]]))
    user_pinv(xp.array([[1, 2], [1, 1]]))


def test_scorecond():
    scorecond(xp.array([[1, 2, 3, 4, 5]]).T)


def test_sparseica_W_adasize_Alasso_mask_regu():
    sparseica_W_adasize_Alasso_mask_regu(
        xp.random.rand(50, 500), xp.ones((50, 50)) - xp.eye(50), 8, 1e-3,
    )


def test_main():
    cov_matrix = [
        [0, 0.6, 0.4, 0, 0, 0, 0.5],
        [0, 0, 0, 0.6, 0, 0, 0],
        [0, 0, 0, 0.8, 0, 0, 0.6],
        [0, 0, 0, 0, 0.5, 0.7, 0],
        [0, 0.6, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.7, 0],
    ]
    cov_matrix = np.array(cov_matrix).T
    var_num = cov_matrix.shape[0]
    noise = 2 * np.random.rand(1000, var_num) - 1
    simul_data = np.linalg.inv(np.eye(var_num) - cov_matrix) @ noise.conj().T
    matrix_data = pd.DataFrame(simul_data.T)
    # 暴露出来允许修改的参数
    param = LocalNgCdParam()
    param.candidate_two_step = False
    param.target_index = list(matrix_data.columns).index(3)
    # 调用主函数计算因果
    local_ng_cd(xp.asarray(matrix_data.T.to_numpy()), param, synthesize=True)


if __name__ == "__main__":
    pytest.main(["-v", "-s", "-q", "test_local_ng_cd.py"])
