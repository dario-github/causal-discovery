# 一、简介

因果发现算法工具包，目前包含：

- fges_mb: 详见`docs/algo/算法伪代码.doc`，page-3：FGES算法
- local_ng_cd：详见`docs/algo/Local_NG_CD分享文档.doc`
- mab_lingam：详见`docs/algo/LiNGAM分享文档.docx`

其中：

- local_ng_cd与mab_lingam是线性模型，没有区分离散与连续数据，统一当做连续值处理。
- fges_mb是图模型，支持添加先验知识，需要预先指定数据是离散或连续。

建议先使用**local_ng_cd**测试数据集效果（速度最快，算法最新，结果渐进正确，考虑了未知混杂因子）

测试使用仿真数据集：`tests/test_data/simul_data.csv`，列索引为指标ID，行索引是样本ID。也可以自行生产仿真数据，命令详见下文。

# 二、使用

## 安装

### pip安装
```sh
python3.7 -m pip install causal-discovery
```

### gpu支持

需手动查看cuda版本并安装相应版本cupy，可以不安装，默认调用numpy在CPU计算

```sh
# 查看支持的cuda版本
ls /usr/local/ | grep cuda

# 安装对应cuda版本的cupy，以cuda10.0为例
python3.7 -m poetry add cupy-cuda100
```

## 快速入门

### 命令行调用

```sh
# 查看参数说明
python3.7 -m causal_discovery fast-simul-data --help
python3.7 -m causal_discovery run-local-ng-cd --help

# 生产仿真数据的参数样例
python3.7 -m causal_discovery fast-simul-data --cov-matrix '[[0, 1, 0], [0, 0, 0.5], [1, 0, 0]]' --sample-size 10

# 生产默认的仿真数据集
python3.7 -m causal_discovery fast-simul-data

# 调用默认仿真数据集
python3.7 -m causal_discovery run-local-ng-cd simul_data.csv 3 matrixT
```

控制台日志最后一条为计算结果保存路径，如未指定`output`目录，默认为当前目录

### 计算结果说明

仿真数据集`simul_data.csv`调用`local_ng_cd`计算后，结果分为两个文件：

1）可信边`edges_trust.json`；可信边是原因直接指向结果的路径（1跳）

```
# 三列，原因、结果、因果效应强度
# 因果效应强度越大，表示直接因果关系越深，正负分别表示正向、负向影响。
causal  reason  effect
2       3       0.7705689874891608
1       3       0.5863603810291644
5       1       0.0993025854935757
3       4       0.5015018174923119
3       5       0.7071753114627015
6       5       0.6977965771255858
```

2）复合权重`synthesize_effect.json`，复合权重是指从原因到结果的所有有向边权重的和，计算n跳复合权重就是计算邻接阵B的n次幂

```
# 三列，原因、结果、复合因果效应强度（5跳内）
causal  reason  effect
2       3       0.7700866938213671
1       3       0.6950546424688089
3       3       0.34082384182310194
5       3       -0.19710467189008646
4       3       0.06902072305646559
```

## 性能

> 建议使用conda自带的numpy库，包含Inter提供的MKL，大幅提高矩阵运算速度（在求逆函数提高约50倍）

`numpy`、`cupy`、`torch`在500 * 500 随机阵求逆的性能对比

|函数|mean|std|
|-|-|-|
|numpy.linalg.inv|71.8 ms|± 64.9 ms|
|**cupy.linalg.inv**|**1.39 ms**|**± 41.5 µs**|
|torch.inverse|6.02 ms|± 6.26 µs|

## 参数说明

### 命令行简化版

```sh
Usage: __main__.py [OPTIONS] INPUT_FILE TARGET
                   DATA_TYPE:[triple|matrix|matrixT]

  [因果发现算法：Local-NG-CD, 作者：张坤, 年份：2020]
  
Args:
    input_file (str): [输入文件地址，csv格式]
    target (str): [目标变量名]
    data_type (DataType): [数据类型：triple（三元组[样本索引，变量名，值]）、matrix（矩阵，行索引为变量名，
    列索引为样本索引）、matrixT（矩阵，行索引为样本索引，列索引为变量名）]
    sep (str, optional): [csv分隔符]. Defaults to ",".
    index_col (str, optional): [读取csv的index索引]. Defaults to None.
    header (str, optional): [读取csv的header索引]. Defaults to None.
    output_dir (str, optional): [输出目录]. Defaults to "./output".
    log_root (str, optional): [日志目录]. Defaults to "./logs".
    verbose (bool, optional): [是否打印日志到控制台]. Defaults to True.
    candidate_two_step (bool, optional): [是否启用2跳关系筛选]. Defaults to False.

Raises:
    DataTypeError: [数据类型错误]

Arguments:
  INPUT_FILE                      [required]
  TARGET                          [required]
  DATA_TYPE:[triple|matrix|matrixT]
                                  [required]

Options:
  --sep TEXT                      [default: ,]
  --index-col TEXT
  --header INTEGER
  --output-dir TEXT               [default: ./output]
  --log-root TEXT                 [default: ./logs]
  --verbose / --no-verbose        [default: True]
  --candidate-two-step / --no-candidate-two-step
                                  [default: False]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.

  --help                          Show this message and exit.
```

### 参数配置完整版

#### Local_NG_CD

```python
# 引用方式
from causal_discovery.parameter.algo import LocalNgCdParam

# 参数详情
target_index: int = Field(0, ge=0)               # 目标变量索引，默认0，如非必要不用修改
candidate_two_step: bool = True                  # 是否用2跳的相关性筛选更多指标，True会用两跳相关性，更多变量。
alpha: float = Field(5e-2, ge=0, le=1)           # 相关性筛选时的p值，值越小表示越严格，一般用0.05或0.01表示95%、99%置信度
mb_beta_threshold: float = Field(5e-2, ge=0)     # 用ALasso回归获取因子权重，用来筛选是否为无向边的阈值，值越大表示越严格
ica_regu: float = Field(1e-3, gt=0)              # ICA时，用来约束稀疏度的惩罚项，值越小，得到的关系图越稀疏
b_orig_trust_value: float = Field(5e-2, gt=0)    # 得到邻接阵B后，用来进一步筛选的权重阈值，默认0.05，值越大表示越严格
```

#### MAB_LinGAM

```python
# 引用方式
from causal_discovery.parameter.algo import MabLingamParam

# 参数详情
target_ind: 0               # 目标变量索引，默认0，如非必要不用修改
max_iter: 100               # 单次ICALiNGAM的迭代次数，
sample_num: 1000            # 样本的连续采样数，是输入矩阵的第一维
fea_num: 30                 # 特征的采样数，是输入矩阵的第二维
tolerance: 0.5              # 阈值，大于这个值视为特征与目标变量构成因果关系
fea_num_return: 7           # 返回有因果关系的特征的数量
type_update: square_error   # 找到有因果关系的特征后，消除特征对目标变量的影响的方式，取值：square_error（最小二乘法）,causal（因果方式）,none（不处理）
type_preprocess: uniform    # 特征数据预处理,取值：uniform(0-1分布),normal（正态分布）,none（不处理）
need_revalue: False         # 消除特征对目标变量的影响后，是否重新更新目标变量的分布，取值：True（是）,False（否）
need_weight: False          # 是否需要输出因果关系的强度，取值：True（是）,False（否）
seed: 1                     # random随机种子
np_seed: 1                  # numpy随机种子
```

#### FGES_MB

```python
# 引用方式
from causal_discovery.parameter.algo import FgesMbParam

# 参数详情
target_index: int = 0                      # 目标变量索引，默认0，如非必要不用修改
score: TypeScore = TypeScore.SEMBicScore   # 离散：BDeuScore，连续：SEMBicScore，离散连续混合：暂未支持
sparsity: int = 20                         # 图的稀疏度惩罚，值越大表示越稀疏
cache_interval: int = 1                    # score缓存间隔，如果内存撑不住，增大该参数的值减少缓存量
max_degree: float = 1e6                    # 最大度限制，如果大于节点数表示无限制
verbose: bool = True                       # 是否打印加边删边细节
knowledge: T = None                        # 先验知识，类型为algorithm.fges_mb.utils.knowledge.Knowledge类，None表示无先验知识
```


# 三、开发

## 环境准备

### 创建虚拟环境
  
```sh
# python版本：>=3.7
cd $PROJECT_DIR
python3.7 -m pip install -U pip setuptools
python3.7 -m pip install poetry
python3.7 -m poetry install
```

## 构建文档

```sh
poetry install --extra doc
invoke doc
```

## 调用方式
   
### python
```python
# 算法主函数
from causal_discovery.algorithm import local_ng_cd, fges_mb, mab_lingam  

# 参数类
from causal_discovery.parameter.algo import LocalNgCdParam, FgesMbParam, MabLingamParam
```

### 命令行（参数详情见上文）

```sh
# 查看参数说明
python3.7 -m causal_discovery run-local-ng-cd --help

# 调用示例
python3.7 -m causal_discovery run-local-ng-cd simul_data.csv 3 matrixT
```