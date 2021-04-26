import numpy as np
from causal_discovery.parameter.env import select_xp, to_numpy

xp = select_xp()


def pdinv(A):
    """
    PDINV 计算正定阵的逆
    """
    num_data = A.shape[0]
    try:
        U = np.linalg.cholesky(to_numpy(xp, A)).T  # cupy对于非正定阵不报错，所以还是用numpy来求上三角阵
        # invU = eye(num_data)/U
        # invU = xp.eye(num_data) @ xp.linalg.inv(U)  # 用了伪逆
        invU = xp.eye(num_data) @ user_inv(xp.asarray(U))  # 用了伪逆
        Ainv = invU @ invU.conj().T
    except np.linalg.LinAlgError:
        # if str(e).strip() == "Matrix is not positive definite":
        if xp.isinf(A).any():  # 针对数值超出python范围的情况，取最大值替换inf
            tmp = A[xp.where(~xp.isinf(A))]
            max_num, min_num = max(tmp), min(tmp)
            A[xp.where((xp.isinf(A)) & (A > 0))] = max_num  # 正负无穷分开处理
            A[xp.where((xp.isinf(A)) & (A < 0))] = min_num
        U, S, Vh = xp.linalg.svd(
            A
        )  # NOTION 随机阵可能会报错：Intel MKL ERROR: Parameter 4 was incorrect on entry to DLASCL.
        V = Vh.T
        Ainv = V @ xp.diagflat(1 / S) @ U.conj().T
    except Exception as e:
        raise e
    return Ainv


def user_inv(matrix):
    """自定义求逆函数

    Args:
        matrix ([np.ndarray]): [numpy矩阵]

    Returns:
        [np.ndarray]: [matrix的逆阵]
    """
    if xp.__name__ == "cupy":
        try:
            return xp.linalg.inv(matrix)
        except xp.cuda.cusolver.CUSOLVERError:  # CUDA10.0的遗留问题，输入矩阵过大，需要较大的工作缓冲区
            return xp.asarray(np.linalg.inv(xp.asnumpy(matrix)))
    else:
        return xp.linalg.inv(matrix)


def user_pinv(matrix):
    """自定义求伪逆函数

    Args:
        matrix ([np.ndarray]): [numpy矩阵]

    Returns:
        [np.ndarray]: [matrix的伪逆]
    """
    return xp.linalg.pinv(matrix)


if __name__ == "__main__":
    print("=" * 20)
    print(pdinv(np.array([[1, 2], [1, 1]])))
    print(user_inv(np.array([[1, 2], [1, 1]])))
    print(user_pinv(np.array([[1, 2], [1, 1]])))