[![CI](https://github.com/dario-github/causal_discovery/actions/workflows/main.yml/badge.svg)](https://github.com/dario-github/causal_discovery/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/dario-github/causal_discovery/branch/main/graph/badge.svg?token=M5BFSZ0NG5)](https://codecov.io/gh/dario-github/causal_discovery)
[![version](https://img.shields.io/badge/version-1.0.5-green.svg?maxAge=259200)](#)
![visitors](https://visitor-badge.glitch.me/badge?page_id=dario-github.causal_discovery&left_color=gray&right_color=green)

[English](./README.md) / [简体中文](./README.zh.md)

<a name="index">**Index**</a>

<a href="0">1. Introduction</a>

<a href="1">2. Usage</a>

&emsp;<a href="#2">Installation</a>

&emsp;&emsp;<a href="#3">Installing via pip</a>

&emsp;&emsp;<a href="#4">GPU Support</a>

&emsp;<a href="#5">Quick Start</a>

&emsp;&emsp;<a href="#6">Command Line Usage</a>

&emsp;&emsp;<a href="#7">Explanation of Output</a>

&emsp;<a href="#8">Performance</a>

&emsp;<a href="#9">Parameter Description</a>

&emsp;&emsp;<a href="#10">Simplified Command Line Version</a>

&emsp;&emsp;<a href="#11">Complete Parameter Configuration</a>

&emsp;&emsp;&emsp;<a href="#12">Local_NG_CD</a>

<a href="13">3. Development</a>

&emsp;<a href="#14">Environment Setup</a>

&emsp;&emsp;<a href="#15">Creating a Virtual Environment</a>

&emsp;<a href="#16">Building Documentation</a>

&emsp;<a href="#17">Method of Invocation</a>

&emsp;&emsp;<a href="#18">Python</a>

&emsp;&emsp;<a href="#19">Command Line (see above for parameter details)</a>


# <a name="0">I. Introduction</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>


The causal discovery algorithm toolkit currently includes:

- local_ng_cd: see docs/algo/Local_NG_CD.doc for details

Note that:

- local_ng_cd is a linear model that does not distinguish between discrete and continuous data, and treats them uniformly as continuous values.

It is recommended to use local_ng_cd to test the performance on the dataset first (it is the fastest and the algorithm is the newest, and the results are asymptotically correct, taking into account unknown confounding factors).

See the following text for detailed usage instructions.

# <a name="1">II. Usage</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

## <a name="2">Installation</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

### <a name="3">pip Installation</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

```sh
python3.7 -m pip install causal-discovery
```

### <a name="4">GPU Support</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

It is necessary to check the CUDA version manually and install the corresponding version of CuPy. If CuPy is not installed, NumPy will be used for CPU computation by default.

```sh
# Check the supported CUDA version
ls /usr/local/ | grep cuda

# Install the corresponding version of CuPy, for example, CUDA 10.0
python3.7 -m poetry add cupy-cuda100

```

## <a name="5">Quick Start</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

### <a name="6">Command Line Usage</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

```sh
# Check the parameter instructions
python3.7 -m causal_discovery fast-simul-data --help
python3.7 -m causal_discovery run-local-ng-cd --help

# Example of parameters for generating simulated data
python3.7 -m causal_discovery fast-simul-data --cov-matrix '[[0, 1, 0], [0, 0, 0.5], [1, 0, 0]]' --sample-size 10

# Generate a default simulated data set (the first row represents the column index indicating the variable names, and each row represents a sampling record)
python3.7 -m causal_discovery fast-simul-data

# Call the default simulated data set
python3.7 -m causal_discovery run-local-ng-cd simul_data.csv 3 matrixT
```

The last line of the console log is the path where the calculation result is saved. If the 'output' directory is not specified, it defaults to the current directory.

### <a name="7">Calculation Results Description</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

After calling `local_ng_cd` with the simulation dataset `simul_data.csv`, the result is divided into two files:

1) Trustworthy edges `edges_trust.json`; trustworthy edges are the paths that directly lead from the cause to the effect (1 hop).

    - Three columns, cause, effect, and causal effect strength.

    - The larger the causal effect strength, the deeper the direct causal relationship is. Positive and negative values indicate positive and negative effects, respectively.

```
causal  reason  effect
2       3       0.7705689874891608
1       3       0.5863603810291644
5       1       0.0993025854935757
3       4       0.5015018174923119
3       5       0.7071753114627015
6       5       0.6977965771255858
```

2) Composite weight `synthesize_effect.json`. The composite weight is the sum of all directed edge weights from the cause to the effect. The n-step composite weight can be calculated by computing the nth power of the adjacency matrix B.

    - Three columns, cause, effect, and composite causal effect strength (within 5 hops).

```
causal  reason  effect
2       3       0.7700866938213671
1       3       0.6950546424688089
3       3       0.34082384182310194
5       3       -0.19710467189008646
4       3       0.06902072305646559
```

## <a name="8">Performance</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

> It is recommended to use the numpy library provided by conda, which includes MKL provided by Inter and greatly improves the speed of matrix operations (about 50 times faster in the inverse function)

Performance comparison of `numpy`, `cupy`, and `torch` for inverting a 500 x 500 random matrix

|Function|mean|std|
|-|-|-|
|numpy.linalg.inv|71.8 ms|± 64.9 ms|
|**cupy.linalg.inv**|**1.39 ms**|**± 41.5 µs**|
|torch.inverse|6.02 ms|± 6.26 µs|


## <a name="9">Parameter Description</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

### <a name="10">Simplified Command Line Version</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>


```sh
Usage: __main__.py [OPTIONS] INPUT_FILE TARGET
                   DATA_TYPE:[triple|matrix|matrixT]

  [Causal Discovery Algorithm: Local-NG-CD, Author: Kun Zhang, Year: 2020]
  
Args:
    input_file (str): [Input file address in csv format]
    target (str): [Name of the target variable]
    data_type (DataType): [Data type: triple (triplet [sample index, variable name, value]), 
                           matrix (matrix, row index as variable name, column index as sample index),
                           matrixT (matrix, row index as sample index, column index as variable name)]
    sep (str, optional): [Csv delimiter]. Defaults to ",".
    index_col (str, optional): [Index index for reading csv]. Defaults to None.
    header (str, optional): [Header index for reading csv]. Defaults to None.
    output_dir (str, optional): [Output directory]. Defaults to "./output".
    log_root (str, optional): [Log directory]. Defaults to "./logs".
    verbose (bool, optional): [Whether to print logs to the console]. Defaults to True.
    candidate_two_step (bool, optional): [Whether to enable 2-step relationship filtering]. Defaults to False.

Raises:
    DataTypeError: [Data type error]

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
                                  Install completion for the specified shell
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.

  --help                          Show this message and exit.
```

### <a name="11">Complete Version of Parameter Configuration</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

#### <a name="12">Local_NG_CD</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>


```python
# Importing method
from causal_discovery.parameter.algo import LocalNgCdParam

# Parameter Details
target_index: int = Field(0, ge=0)               # Target variable index, default 0, unless necessary, no need to modify
candidate_two_step: bool = True                  # Whether to use the 2-step correlation filtering to obtain more variables. If True, the 2-step correlation is used to filter more variables.
alpha: float = Field(5e-2, ge=0, le=1)           # p-value used in correlation filtering. The smaller the value, the more stringent. Generally, 0.05 or 0.01 is used to represent 95% or 99% confidence level
mb_beta_threshold: float = Field(5e-2, ge=0)     # A threshold used to determine whether the edge is undirected when obtaining factor weights using ALasso regression. The larger the value, the more stringent.
ica_regu: float = Field(1e-3, gt=0)              # A penalty term used to constrain the sparsity when using ICA. The smaller the value, the sparser the resulting graph.
b_orig_trust_value: float = Field(5e-2, gt=0)    # A weight threshold used for further filtering after obtaining the adjacency matrix B. The default value is 0.05, and the larger the value, the more stringent.
```

# <a name="13">III. Development</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

## <a name="14">Environment Setup</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

### <a name="15">Creating a Virtual Environment</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

  
```sh
# python version: >=3.7
cd $PROJECT_DIR
python3.7 -m pip install -U pip setuptools
python3.7 -m pip install poetry
python3.7 -m poetry install
```

## <a name="16">Building the Documentation

</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

```sh
poetry install --extra doc
invoke doc
```

## <a name="17">Calling Method

</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>
   
### <a name="18">python</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>
```python
# Algorithm Main Function
from causal_discovery.algorithm import local_ng_cd, fges_mb, mab_lingam  

# Parameter Class
from causal_discovery.parameter.algo import LocalNgCdParam, FgesMbParam, MabLingamParam
```

### <a name="19">Command Line (See above for details on the parameters)</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>

```sh
# Viewing Parameter Descriptions
python3.7 -m causal_discovery run-local-ng-cd --help

# Calling Example
python3.7 -m causal_discovery run-local-ng-cd simul_data.csv 3 matrixT
```
