import logging
import numpy as np
from causal_discovery.algorithm.local_ng_cd.util.pdinv import pdinv, user_inv, user_pinv
from causal_discovery.parameter.algo import BetaAdaptiveLassoGrad2Step
from causal_discovery.parameter.env import select_xp

xp = select_xp()

def betaAlasso_grad_2step(
    x, y, var_noise: float, lambda_value: float
):
    """
    搜索给定 lambda_value 的自适应套索（ALasso；adaptive Lasso Zou, 2006）解。

    目的为找到节点 x 上的无向邻接，这是一种正则化的回归方法，
    对每个单独的系数进行了适当的惩罚。使用 ALasso，每个节点 x_i 在 x 中的
    其余节点上回归。


    :param x: p*n 矩阵
    :param y: 1*n 矩阵
    :param var_noise: 噪声的方差，标量
    :param lambda_value: log(T) / 2，T是样本数
    :return beta_al: 由第一步（传统 ALasso ）后得到的 ALasso 系数 beta 组成，
                    第一步收敛后，重复更新 \\hat{beta} 并且重复 ALasso 步骤来提升效果。
    :return beta_new_n: 由第一步后的惩罚项（ beta_al / \\hat{beta} ）组成。
    :return beta2_al: 由第二步后的结果组成
    :return beta2_new_n: 由第二步后的结果组成
    注意：在第一步使用了梯度方法与牛顿方法的结合。

    作者：张坤
    """
    param = BetaAdaptiveLassoGrad2Step()
    var_noise_back = var_noise

    trad_1 = param.trad_1
    trad_2 = 1 - trad_1
    N, T = x.shape  # 得到矩阵维度
    tol = param.tol  # 1E-10
    beta_min = param.beta_min
    beta_min_2 = param.beta_min_2  # ASK 为什么要设1E-2，和beta_min差这么多
    sum_adjust_beta = []
    pl = []  # penalization

    # centering the data
    logging.info("centering the data")
    # 归一
    x = x - xp.tile(xp.mean(x.conj().T, axis=0), (T, 1)).conj().T  # 用T份副本创建1 m
    y = y - xp.mean(y)
    # (1, T) @ (T, N) = (1, N)
    beta_hat = (
        y @ x.conj().T
    )  # NOTION inv结果与matlab差距很大，以[-2, -1]对比：python -8.65，matlab -3.60
    try:
        inv_xx = pdinv(x @ x.conj().T)
    except np.linalg.LinAlgError:
        inv_xx = user_pinv(x @ x.conj().T)
    except Exception as e:
        raise e
    beta_hat = beta_hat @ inv_xx

    var_noise = var_noise or xp.var(y - beta_hat @ x)
    x_new = xp.diagflat(beta_hat) @ x
    error_value = 1
    beta_new_o = xp.ones((N, 1))
    # beta_new_o = 1E-5 * ones(N,1)
    # store for curve plotting
    logging.info("store for curve plotting")
    sum_adjust_beta.append(sum(abs(beta_new_o)))
    pl.append(
        (y - beta_new_o.conj().T @ x_new)
        @ (y - beta_new_o.conj().T @ x_new).conj().T
        / 2
        / var_noise
        + lambda_value * sum(abs(beta_new_o))
    )
    beta_new_n = None
    while error_value > tol:
        sigma = xp.diagflat(1 / abs(beta_new_o))  # 这里用diagflat是因为beta_new_o是N * 1的向量
        #     Sigma = diag(1./beta_new_o) # this is wrong?
        #     beta_new_n = inv(x_new*x_new.conj().T + var_noise*lambda_value * Sigma) * (x_new*y.conj().T)
        # # with gradient trad-off!
        beta_new_n = (
            user_inv(x_new @ x_new.conj().T + var_noise * lambda_value * sigma)
            @ (x_new @ y.conj().T)
            * trad_1
            + beta_new_o * trad_2
        )
        beta_new_n = xp.sign(beta_new_n) * xp.maximum(abs(beta_new_n), beta_min)
        error_value = xp.linalg.norm(beta_new_n - beta_new_o)
        beta_new_o = beta_new_n
        sum_adjust_beta.append(sum(abs(beta_new_n)))
        pl.append(
            (y - beta_new_n.conj().T @ x_new)
            @ (y - beta_new_n.conj().T @ x_new).conj().T
            / 2
            / var_noise
            + lambda_value * sum(abs(beta_new_n))
        )
        logging.info(f"Error: {error_value}, tol: {tol}")
    Ind = xp.flatnonzero(abs(beta_new_n) > param.ampl_coef * beta_min)
    logging.info(f"{len(Ind)} inds in step-1")
    beta_new_n = beta_new_n * (abs(beta_new_n) > param.ampl_coef * beta_min)
    beta_al = beta_new_n * beta_hat.conj().T

    # figure, plot(sum_adjust_beta, pl, .conj().Tr.-.conj().T)

    ## step 2
    logging.info("=== step 2 ===")
    N2 = len(Ind)
    x2 = x[Ind, :]
    beta2_hat = y @ x2.conj().T
    try:
        inv_xx_2 = pdinv(x2 @ x2.conj().T)
    except np.linalg.LinAlgError:
        inv_xx_2 = user_pinv(x2 @ x2.conj().T)
    except Exception as e:
        raise e
    beta2_hat = beta2_hat @ inv_xx_2
    if not var_noise_back:
        var_noise = xp.var(y - beta2_hat @ x2)

    x2_new = xp.diagflat(beta2_hat) @ x2
    beta2_new_o = xp.ones((N2, 1))
    sum_adjust_beta2 = []
    pl2 = []
    sum_adjust_beta2.append(sum(abs(beta2_new_o)))
    pl2.append(
        (y - beta2_new_o.conj().T @ x2_new)
        @ (y - beta2_new_o.conj().T @ x2_new).conj().T
        / 2
        / var_noise
        + lambda_value * sum(abs(beta2_new_o))
    )
    error_value = 1
    Iter = 1
    logging.info("calc beta-2")
    while error_value > tol:
        if not Iter % 10:
            logging.info(f"Iter: {Iter}, Error: {error_value}, Target: {tol}")
        sigma = xp.diagflat(1 / abs(beta2_new_o))
        #     Sigma = diag(1./beta2_new_o) # this is wrong?
        #     if det(x2_new*x2_new.conj().T + var_noise*lambda_value * Sigma) < 1E-6 # 0.01
        #         pause
        #     end
        beta2_new_n = pdinv(
            x2_new @ x2_new.conj().T + var_noise * lambda_value * sigma
        ) @ (
            x2_new @ y.conj().T
        )  # NOTION pdinv也会放大误差
        beta2_new_n = xp.sign(beta2_new_n) * xp.maximum(abs(beta2_new_n), beta_min)
        error_value = xp.linalg.norm(beta2_new_n - beta2_new_o)
        beta2_new_o = beta2_new_n
        sum_adjust_beta2.append(sum(abs(beta2_new_n)))
        pl2.append(
            (y - beta2_new_n.conj().T @ x2_new)
            @ (y - beta2_new_n.conj().T @ x2_new).conj().T
            / 2
            / var_noise
            + lambda_value * sum(abs(beta2_new_n))
        )
        Iter += 1
        if Iter > param.iter_limit:  # TODO 原始值100
            break
    logging.info(f"Iter: {Iter}, Error: {error_value}, Target: {tol}")
    
    beta2_new_n = beta2_new_n * (abs(beta2_new_n) > beta_min_2)
    beta2_al = xp.zeros((N, 1))

    beta2_al[Ind] = beta2_new_n * beta2_hat.conj().T
    return beta_al, beta_new_n, beta2_al, beta2_new_n


if __name__ == "__main__":
    a, b, c, d = betaAlasso_grad_2step(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        np.array([[3, 5, 7, 9]]),
        0.01,
        4,
    )
    for x in [a, b, c, d]:
        print("=" * 20)
        print(x)
