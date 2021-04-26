# Filename:
#
# Description:
# % Syntaxe [psi, entropy] = scorecond(data, bdwidth, cova)
# %
# % Estimate the conditional score function defined as minus the
# % gradient of the conditional density of of a random variable X_p
# % given x_{p-1}, \dots, x_{q+1}. The values of these random variables are
# % provided in the n x p array data.
# %
# % The estimator is based on the partial derivatives of the
# % conditional entropy with respect to the data values, the entropy
# % being estimated through the kernel density estimate _after a
# % prewhitening operation_, with the kernel being the density of the
# % sum of 3 independent uniform random variables in [.5,.5]. The kernel
# % bandwidth is set to bdwidth*(standard deviation) and the density is
# % evaluated at bandwidth apart. bdwidth defaults to
# %   2*(11*sqrt(pi)/20)^((p-q)/(p-q+4))*(4/(3*n))^(1/(p-q+4)
# % (n = sample size), which is optimal _for estimating a normal density_
# %
# % If cova (a p x p matrix) is present, it is used as the covariance
# % matrix of the data and the mean is assume 0. This prevents
# % recomputation of the mean and covariance if the data is centered
# % and/or prewhitenned.
# %
# % The score function is computed at the data points and returned in
# % psi.
import logging
import math
import numpy as np
# import scipy.linalg
from scipy.sparse import csr_matrix
from causal_discovery.algorithm.local_ng_cd.util.pdinv import user_inv
from causal_discovery.parameter.env import select_xp, to_numpy

xp = select_xp()

def scorecond(data, q=None, bdwidth=None, cova=None):
    n, p = data.shape

    if not q:
        q = 0

    if p < q + 1:
        logging.error("Sorry: not enough variables")
        raise ValueError("Sorry: not enough variables")

    if not cova:
        tmp = xp.sum(data, axis=0) / n
        data = data - xp.ones((n, 1), dtype=int) * tmp[0]
        cova = data.conj().T @ data / n

    # prewhitening
    t = xp.linalg.cholesky(cova).T
    # data = data @ xp.linalg.inv(t)
    data = data @ user_inv(t)

    if q > 0:
        # delete first q columns
        data[:, 0 : q - 1] = []
        p = p - q

    if not bdwidth:
        bdwidth = (
            2
            * (11 * math.sqrt(math.pi) / 20) ** (p / (p + 4))
            * (4 / (3 * n)) ** (1 / (p + 4))
        )

    # Grouping the data into cells, idx gives the index of the cell
    # containing a datum, r gives its relative distance to the leftmost
    # border of the cell
    r = data / bdwidth
    idx = xp.floor(r).astype(xp.int16)
    r = r - idx
    tmp = idx.min(axis=0)
    idx = idx - xp.ones((n, 1), dtype=int) * tmp[0]  # 这里巨tm坑(0 <= idx)

    # The array ker contains the contributions to the probability of cells
    # The array kerp contains the gradient of these contributions
    # The array ix contains the indexes of the cells, arranged in
    # _lexicographic order_
    ker = (
        xp.array(
            [(1 - r[:, 0]) ** 2 / 2, 0.5 + r[:, 0] * (1 - r[:, 0]), r[:, 0] ** 2 / 2]
        )
        .conj()
        .T
    )
    index_cell = xp.array([idx[:, 0], idx[:, 0] + 1, idx[:, 0] + 2]).conj().T
    kerp = xp.array([1 - r[:, 0], 2 * r[:, 0] - 1, -r[:, 0]]).conj().T
    mx = idx.max(axis=0) + 3  # NOTION 由+2改为+3，在idx+2基础上+1，表示元素数量
    m = xp.cumprod(mx, axis=0)  # 最大值的累乘？
    nr = xp.array(range(n), ndmin=2).T
    for i in range(1, p):
        ii = i * xp.ones((1, 3 ** (i - 1)), dtype=int)
        kerp = (
            xp.array(
                [
                    kerp * (1 - r[nr, ii]) ** 2 / 2,
                    kerp * (0.5 + r[nr, ii]) * (1 - r[nr, ii]),
                    kerp * r[nr, ii] ** 2 / 2,
                    ker * (1 - r[:, ii]),
                    ker * (2 * r[:, ii] - 1),
                    -ker * r[:, ii],
                ]
            )
            .conj()
            .T
        )
        nr = xp.concatenate((nr, xp.arange(0, n)))
        ker = (
            xp.array(
                [
                    ker * (1 - r[:, ii]) ** 2 / 2,
                    ker * (0.5 + r[:, ii]) * (1 - r[:, ii]),
                    ker * r[:, ii] ** 2 / 2,
                ]
            )
            .conj()
            .T
        )
        mi = m[i - 1]
        index_cell = (
            xp.array(
                [
                    index_cell + mi * (idx[:, ii] - 1),
                    index_cell + mi * idx[:, ii],
                    index_cell + mi * (idx[:, ii] + 1),
                ]
            )
            .conj()
            .T
        )

    # joint prob. of cells
    # xp.array([1, 1, 1], ndmin=2)
    pr = xp.asarray(
        csr_matrix(
            (
                to_numpy(xp, ker).flatten(order="F"),
                (
                    to_numpy(xp, index_cell).flatten(order="F"),
                    np.array([0] * index_cell.size, dtype=int),
                ),
            ),
            shape=(to_numpy(xp, m[p - 1]), 1),  # TODO m[p - 1][0], 暂时先+1处理，需要搞清楚为什么
        ).toarray()
        / n
    )
    # to contain log(cond. prob.)
    logp = xp.zeros((to_numpy(xp, m[p - 1]), 1))
    if p > 1:
        # marginal prob. (Mi = M(p-1))
        pm = xp.sum(pr.reshape(mi, mx[p][0], order="F").copy(), axis=1)
        pm = pm[:, xp.ones((1, to_numpy(xp, mx[p])))].reshape(to_numpy(xp, m[p]), 1, order="F").copy()
        # avoid 0
        logp[xp.flatnonzero(pr)] = (
            xp.log(pr[xp.flatnonzero(pr)]) / pm[xp.flatnonzero(pr)]
        )
    else:
        logp[xp.flatnonzero(pr)] = xp.log(pr[xp.flatnonzero(pr)])

    # compute the conditional entropy (if asked)
    entropy = (
        xp.log(bdwidth * t[-1, -1]) - pr.conj().T @ logp if q is not None else None
    )

    # Compute the conditional score
    psi = xp.sum(logp[index_cell[nr.flatten(), :]][:, :, 0] * kerp, axis=1)
    psi = psi.reshape(n, p, order="F").copy() / bdwidth
    tmp = xp.sum(psi, axis=0) / n
    psi = psi - xp.ones((n, 1), dtype=int) * tmp[0]
    # correction
    lam = psi.conj().T @ data / n
    lam = xp.tril(lam) + xp.tril(lam, -1).conj().T
    lam[p - 1, p - 1] = lam[p - 1, p - 1] - 1

    if q > 0:
        # psi = xp.array([xp.zeros((n, q)), psi - data * lam]) @ xp.linalg.inv(t.conj().T)
        psi = xp.array([xp.zeros((n, q)), psi - data * lam]) @ user_inv(t.conj().T)
    else:
        # psi = (psi - data * lam) @ xp.linalg.inv(t.conj().T)
        psi = (psi - data * lam) @ user_inv(t.conj().T)
    return psi, entropy


if __name__ == "__main__":
    print(scorecond(np.array([[1, 2, 3, 4, 5]]).T))
