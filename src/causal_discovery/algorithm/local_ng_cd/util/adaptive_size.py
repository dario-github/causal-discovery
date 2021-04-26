from causal_discovery.parameter.algo import AdaptiveSize
from causal_discovery.parameter.env import select_xp

xp = select_xp()


def adaptive_size(grad_new, grad_old, eta_old, z_old):
    """[通过上下界参数up、down，自适应调整grad]

    Args:
        grad_new ([type]): [description]
        grad_old ([type]): [description]
        eta_old ([type]): [description]
        z_old ([type]): [description]

    Returns:
        [type]: [description]
    """
    param = AdaptiveSize()
    z = grad_new + param.alpha * z_old  # 这里alpha默认为0,
    etaup = (grad_new * grad_old) >= 0

    eta = eta_old * (param.up * etaup + param.down * (1 - etaup))
    eta = xp.minimum(eta, param.eta_minimum)  # 防止eta值过大
    return eta, z
