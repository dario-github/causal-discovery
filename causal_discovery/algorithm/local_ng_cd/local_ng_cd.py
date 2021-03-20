#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Filename:
#
# Description:
#  Local linear causal discovery method for discovering all direct causes and
#  effects of the target_value.
#  Input:
#    X: Data matrix (variable number * sample size);
#    target_variable: the sequential number of the variable for which local
#        non-Gaussian causal discovery to be performed.
#  Output:
#    Edges_trust: the causal edges into and out of the target variable and those
#        causal edges into the children of the target variale, which are
#        trustable although we are not analyzing all variables together; its
#        i-th row is the i-th edge, which goes from Edges_trust(i,1) to
#        Edges_trust(i,2) with coefficient Edges_trust(i,3).
#    B_orig: the 'causal' adjacency matrix estimated from the target
#        variable and its discovered Markov blanket, some of which are
#        trustable.
#    y_m: the estimated independent subspace components for the target
#        variable and its discovered Markov blanket.
#    Ind: the indices of the variables in the Markov blanket of the target
#        variable; the first entry is target_variable.
# Version:       1.0

import math
import logging
import numpy as np
import cupy as cp
# cp.cuda.Device(1).use()
from scipy.stats import pearsonr
from causal_discovery.algorithm.local_ng_cd.util.betaAlasso_grad_2step import betaAlasso_grad_2step
from causal_discovery.algorithm.local_ng_cd.util.sparseica_w_adasize_alasso_mask_regu import (
    sparseica_W_adasize_Alasso_mask_regu,
)
from causal_discovery.parameter.algo import LocalNgCdParam


def get_edges_trust(b_orig, ind_mb):
    b_orig, ind_mb = cp.asnumpy(b_orig), cp.asnumpy(ind_mb)
    # 目标的原因
    ind_into = np.flatnonzero(b_orig[0, :] != 0)
    logging.info(f"have {len(ind_into)} direct causal")
    edges_trust = None
    if len(ind_into) > 0:
        edges_trust = np.zeros(shape=(len(ind_into), 3))
        for i in range(len(ind_into)):
            edges_trust[i, 0] = ind_mb[ind_into[i]]
            edges_trust[i, 1] = ind_mb[0]
            edges_trust[i, 2] = b_orig[0, ind_into[i]]
    # 间接原因
    for i in range(len(ind_into)):
        ind_tmp = np.flatnonzero(b_orig[ind_into[i], :] != 0)
        # the conidered child has other causes
        if len(ind_tmp) < 2:
            continue
        for j in range(1, len(ind_tmp)):
            edges_trust = np.concatenate(
                (
                    edges_trust,
                    np.array(
                        [
                            ind_mb[ind_tmp[j]],
                            ind_mb[ind_into[i]],
                            b_orig[ind_into[i], ind_tmp[j]],
                        ],
                        ndmin=2,
                    ),
                ),
                axis=0,
            )

    # 目标的结果
    ind_from = np.flatnonzero(b_orig[:, 0] != 0)
    if len(ind_from) < 1:
        logging.info("No direct reason.")
        return edges_trust or []
    logging.info(f"have {len(ind_from)} direct reason")
    if edges_trust is None:
        edges_trust = np.zeros(shape=(0, 3))
    for i in range(len(ind_from)):
        edges_trust = np.concatenate(
            (
                edges_trust,
                np.array(
                    [ind_mb[0], ind_mb[ind_from[i]], b_orig[ind_from[i], 0]], ndmin=2
                ),
            ),
            axis=0,
        )
    # other edges into the children of target_variable
    for i in range(len(ind_from)):
        ind_tmp = np.flatnonzero(b_orig[ind_from[i], :] != 0)
        # the conidered child has other causes
        if len(ind_tmp) < 2:
            continue
        for j in range(1, len(ind_tmp)):
            edges_trust = np.concatenate(
                (
                    edges_trust,
                    np.array(
                        [
                            ind_mb[ind_tmp[j]],
                            ind_mb[ind_from[i]],
                            b_orig[ind_from[i], ind_tmp[j]],
                        ],
                        ndmin=2,
                    ),
                ),
                axis=0,
            )
    return edges_trust


def corr_filter(x, target_variable, alpha, candidate_two_step):
    # 相关性筛选放在这里做
    pval = np.array([pearsonr(cp.asnumpy(x_i), cp.asnumpy(x[target_variable, :]))[1] for x_i in x])
    ind_corr = np.flatnonzero(pval < alpha)
    logging.info(f"first have {ind_corr.shape[0]} ind")
    ind_corr = np.delete(ind_corr, np.where(ind_corr == target_variable), axis=0)

    # 计算2跳相关性，如果想提高效率可跳过
    ind_corr_final = ind_corr
    if candidate_two_step:
        for i in range(len(ind_corr)):
            pval2 = np.array(
                [pearsonr(cp.asnumpy(x_i), cp.asnumpy(x[ind_corr[i], :]))[1] for x_i in x]
            )  # x中变量再和1跳变量计算相关性
            ind_corr2 = np.flatnonzero(pval2 < alpha)
            ind_corr_final = np.concatenate((ind_corr_final, ind_corr2))
            ind_corr_final = np.unique(ind_corr_final)

    # 确保ind_corr_final不包含目标变量
    ind_corr_final = np.delete(
        ind_corr_final, np.where(ind_corr_final == target_variable), axis=0
    )
    return np.array(ind_corr_final)


def few_point_operation(x, ind_corr_final, target_variable, var_num, sample_size):
    xp = cp.get_array_module(x)
    # 样本量t太小时，预选t/4的特征
    if sample_size < 3 * (var_num + 1):
        ind_t, _ = zip(*sorted(enumerate([pearsonr(cp.asnumpy(x[ind, :]), cp.asnumpy(x[target_variable, :]))[1]
                                          for ind in ind_corr_final]), key=lambda x: abs(x[1]), reverse=True))
        ind_t = np.array(ind_t)
        # pre-select N/4 features
        ind_corr_final = ind_corr_final[ind_t[: math.floor(sample_size / 4)]]
    return xp.asarray(cp.asnumpy(ind_corr_final))


def local_ng_cd(x, param: LocalNgCdParam, synthesize: bool = False):
    """[局部线性非高斯因果发现算法]

    Args:
        x ([type]): [numpy或cupy的array格式，行索引为变量名，列索引为样本ID]
        param (LocalNgCdParam): [参数类，见parameter.algo.LocalNgCdParam]
        synthesize (bool, optional): [是否输出复合权重]. Defaults to False.

    Returns:
        edges_trust, [synthesize_effect]: [可信边三元组，[原因，结果，因果效应强度]]
    """
    xp = cp.get_array_module(x)
    candidate_two_step: bool = param.candidate_two_step  # NOTION 是否用2跳的相关性
    target_index: int = param.target_index

    _, sample_size = x.shape
    logging.info(f"data shape: {x.shape}")
    # 剔除连续序列的变量(x.max(axis=1) == x.min(axis=1))
    # logging.info(f"del {list(np.where(x.max(axis=1) == x.min(axis=1))[0])}")
    # x = x[x.max(axis=1) != x.min(axis=1)]

    # 先用相关性找1跳关系，再继续找2跳关系，构成目标变量的马尔科夫毯集
    # 这里设定p值
    alpha = param.alpha

    # 相关性筛选
    ind_corr_final = corr_filter(x, target_index, alpha, candidate_two_step)
    valid_var_num = len(ind_corr_final)
    if candidate_two_step:
        logging.info(f"second have {valid_var_num} ind")

    # 小样本处理
    ind_corr_final = few_point_operation(
        x, ind_corr_final, target_index, valid_var_num, sample_size
    )

    # 搜索给定 lambda_value 的自适应套索解
    _, _, beta2_alt, _ = betaAlasso_grad_2step(
        x[ind_corr_final, :],
        x[[target_index], :],
        0.6 ** 2 * xp.var(x[target_index, :], axis=0, ddof=1),
        xp.log(sample_size) / 2,
    )

    # further selected indices for Markov blanket
    selected = xp.flatnonzero(abs(beta2_alt) > param.mb_beta_threshold)
    ind_mb = ind_corr_final[selected]
    ind_mb = xp.concatenate((xp.array([target_index]), ind_mb), axis=0)  # 这里已经把目标变量放在第一位了
    var_num_mb = len(ind_mb)
    logging.info(f"{var_num_mb - 1} ind-mb")
    if var_num_mb < 2:
        logging.info("no valid ind-mb.")
        return [] if not synthesize else [[], []]

    # Now perform independent subspace analysis
    mask = xp.ones((var_num_mb, var_num_mb)) - xp.eye(var_num_mb)

    # 执行受约束的ICA
    regu = param.ica_regu
    _, w_m, _, _ = sparseica_W_adasize_Alasso_mask_regu(
        x[ind_mb, :], mask, math.log(sample_size), regu
    )

    b_orig = xp.eye(var_num_mb) - w_m
    b_orig = b_orig * (abs(b_orig) > param.b_orig_trust_value)
    edges_trust = get_edges_trust(b_orig, ind_mb)
    if not synthesize:
        return edges_trust
    else:
        synthesize_effect = np.sum([np.linalg.matrix_power(cp.asnumpy(b_orig), num) for num in range(1, 6)], axis=0)
        synthesize_effect = sorted(list(enumerate(synthesize_effect[0, :])), key=lambda x: -abs(x[1]))
        return edges_trust, synthesize_effect


if __name__ == "__main__":
    import scipy.io as sio
    import pickle
    from peon.log import config_log
    import torch
    from causal_discovery.parameter.env import set_gpu
    if torch.cuda.is_available():
        gpu_num = set_gpu(15e2)
        cp.cuda.Device(gpu_num).use()
    from causal_discovery.parameter.algo import LocalNgCdParam
    config_log(
        "local_ng_cd", "local_ng_cd", print_terminal=True, enable_monitor=True,
    )
    param = LocalNgCdParam()
    param.candidate_two_step = False
    param.target_index = 3
    # data = sio.loadmat("./test_data/industry_M003802298.mat")["x"]
    # data = sio.loadmat("/root/work/git_code/causal_discovery/test/test_data/simul_data.mat")["XX"]
    data = pickle.load(open("/root/work/lab_test/time_series/data/tmp.pkl", "rb"))
    for x in local_ng_cd(cp.asarray(data), param):
        print(x)