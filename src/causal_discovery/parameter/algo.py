from enum import Enum
from typing import TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class DataType(str, Enum):
    triple = "triple"
    matrix = "matrix"
    matrixT = "matrixT"

class TypeUpdate(str, Enum):
    square_error = "square_error"
    causal = "causal"
    none = "none"


class TypePreprocess(str, Enum):
    uniform = "uniform"
    normal = "normal"
    none = "none"


class MabLingamParam(BaseModel):
    target_ind: int = Field(0, ge=0)
    max_iter: int = Field(100, gt=0)
    sample_num: int = Field(1000, gt=0)
    fea_num: int = Field(10, gt=0)
    tolerance: float = Field(0.5, ge=0, le=1)
    fea_num_return: int = Field(7, gt=0)
    type_update: TypeUpdate = TypeUpdate.square_error
    type_preprocess: TypePreprocess = TypePreprocess.uniform
    need_revalue: bool = False
    need_weight: bool = False
    seed: int = Field(1, ge=1, le=6)
    np_seed: int = 1


class TypeScore(str, Enum):
    SEMBicScore = "SEMBicScore"
    BDeuScore = "BDeuScore"


class AdaptiveSize(BaseModel):
    alpha: float = 0
    up: float = 1.05
    down: float = 0.8
    eta_minimum: float = 0.03

class BetaAdaptiveLassoGrad2Step(BaseModel):
    trad_1: float = 0.2
    tol: float = 1e-2
    beta_min: float = 1e-12
    beta_min_2: float = 1e-2
    iter_limit: int = 1e3
    ampl_coef: float = 1e4


class NaturalGradAdasizeMaskRegu(BaseModel):
    mu: float = 3e-3
    itmax: int = 18000
    tol: float = 1e-4
    early_stopping_times: int = 300  # 连续n次检测到不再下降就停止


class SparseicaWAdasizeALassoMaskRegu(BaseModel):
    mu: float = 1e-3  # learning rate
    m: int = 40  # for approximate the derivative of |.|; 60 40
    itmax: int = 10000
    tol: float = 1e-6
    stagnation_limit: int = 300  # 梯度连续n次没有下降，停止

class LocalNgCdParam(BaseModel):
    target_index: int = Field(0, ge=0)  # 目标变量
    candidate_two_step: bool = True  # 是否用2跳的相关性
    alpha: float = Field(5e-2, ge=0, le=1)  # 搜索MB的p值
    mb_beta_threshold: float = Field(
        5e-2, ge=0
    )  # further selected indices for Markov blanket
    ica_regu: float = Field(1e-3, gt=0)  # ICA约束值
    b_orig_trust_value: float = Field(5e-2, gt=0)


class FgesMbParam(BaseModel):
    score: TypeScore = TypeScore.SEMBicScore
    sparsity: int = 20
    cache_interval: int = 1
    max_degree: float = 1e6
    verbose: bool = True
    target_index: int = 0
    knowledge: T = None
