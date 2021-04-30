#!/usr/bin/env python
# encoding:utf-8
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from scipy import sparse

from causal_discovery.algorithm.local_ng_cd.local_ng_cd import local_ng_cd
from causal_discovery.data_prepare.matrix_data import get_matrix_data
from causal_discovery.parameter.algo import DataType, LocalNgCdParam
from causal_discovery.parameter.env import select_xp
from causal_discovery.parameter.error import DataTypeError
from causal_discovery.parameter.log import config_log

xp = select_xp()
app = typer.Typer()


@app.command()
def fast_simul_data(
    output_dir: str = "./", cov_matrix: str = None, sample_size: int = 2000
):
    """生产仿真数据

    Args:
        output_dir (str, optional): [description]. Defaults to "./".
        cov_matrix (List[List[float]], optional): [description]. Defaults to None.
        sample_size (int, optional): [description]. Defaults to 2000.
    """
    # 默认协方差矩阵
    if cov_matrix is None:
        cov_matrix = [
            [0, 0.6, 0.4, 0, 0, 0, 0.5],
            [0, 0, 0, 0.6, 0, 0, 0],
            [0, 0, 0, 0.8, 0, 0, 0.6],
            [0, 0, 0, 0, 0.5, 0.7, 0],
            [0, 0.6, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.7, 0],
        ]
    else:
        try:
            cov_matrix = json.loads(cov_matrix)
        except json.JSONDecodeError:
            raise json.JSONDecodeError("Unknow json type")
        else:
            assert isinstance(cov_matrix, List)
    cov_matrix = np.array(cov_matrix).T

    assert (
        len(cov_matrix.shape) == 2 and cov_matrix.shape[0] == cov_matrix.shape[1]
    )  # 2维，正方形矩阵
    assert cov_matrix.dtype in [np.int64, np.float64]  # 数值型

    var_num = cov_matrix.shape[0]
    noise = 2 * np.random.rand(sample_size, var_num) - 1
    simul_data = np.linalg.inv(np.eye(var_num) - cov_matrix) @ noise.conj().T
    pd.DataFrame(simul_data.T).to_csv(Path(output_dir) / "simul_data.csv", index=False)
    print("-" * 10, "cov matrix", "-" * 10)
    print(sparse.csr_matrix(cov_matrix))
    print("-" * 32)
    print("file:", (Path(output_dir) / "simul_data.csv").absolute().as_posix())
    return pd.DataFrame(simul_data.T)


@app.command()
def run_local_ng_cd(
    input_file: str,
    target: str,
    data_type: DataType,
    sep: str = ",",
    index_col: str = "None",
    header: int = 0,
    fill_methods: str = "ffill,bfill",
    output_dir: str = "./output",
    log_root: str = "./logs",
    verbose: bool = True,
    candidate_two_step: bool = False,
):
    """[因果发现算法：Local-NG-CD, 作者：张坤, 年份：2020]

    Args:\n
        input_file (str): [输入文件地址，csv格式]\n
        target (str): [目标变量名]\n
        data_type (DataType): [数据类型：triple（三元组[样本索引，变量名，值]）、matrix（矩阵，行索引为变量名，列索引为样本索引）、matrixT（矩阵，行索引为样本索引，列索引为变量名）]\n
        sep (str, optional): [csv分隔符]. Defaults to ",".\n
        index_col (str, optional): [读取csv的index索引]. Defaults to None.\n
        header (str, optional): [读取csv的header索引]. Defaults to None.\n
        fill_methods (str, optional): [pandas填充数据的方式，表示依次执行填充动作]. Defaults to "ffill,bfill".\n
        output_dir (str, optional): [输出目录]. Defaults to "./output".\n
        log_root (str, optional): [日志目录]. Defaults to "./logs".\n
        verbose (bool, optional): [是否打印日志到控制台]. Defaults to True.\n
        candidate_two_step (bool, optional): [是否启用2跳关系筛选]. Defaults to False.\n

    Raises:\n
        DataTypeError: [数据类型错误]
    """
    config_log(
        "causal_discovery",
        "local_ng_cd",
        log_root=log_root,
        print_terminal=verbose,
        enable_monitor=True,
    )

    logging.info("=" * 10 + "Start !!!" + "=" * 10)
    logging.info("Read data.")

    dataframe = pd.read_csv(
        input_file,
        index_col=([int(x) for x in index_col.split(",")] if index_col != "None" else None),
        header=header,
        sep=sep,
    )
    datatype_param = {
        DataType.triple: {"triple_df": dataframe},
        DataType.matrix: {"ret_df": dataframe.T},
        DataType.matrixT: {"ret_df": dataframe},
    }

    # matrix_data = dataframe

    # 解析三种输入数据
    if data_type in datatype_param:
        matrix_data = get_matrix_data(
            target,
            corr_filter=True,
            fill_methods=[x if x != "None" else None for x in fill_methods.split(",")],
            **datatype_param[data_type],
        ).T
    else:
        raise DataTypeError(f"Unknown datatype: {data_type}.")

    if target not in matrix_data.columns:
        raise DataTypeError(
            "Choosing matrix/matrixT datatype, but no target in indexes or columns, please check arg: index_col/header"
        )

    # 暴露出来允许修改的参数
    param = LocalNgCdParam()
    param.candidate_two_step = candidate_two_step
    param.target_index = list(matrix_data.columns).index(target)
    # 变量token -> name
    index_map = dict(enumerate(matrix_data.columns))
    # 调用主函数计算因果
    edges_trust, synthesize_effect = local_ng_cd(
        xp.asarray(matrix_data.T.to_numpy()), param, synthesize=True
    )

    edges_trust = [
        (index_map.get(int(causal), "N"), index_map.get(int(result), "N"), weight,)
        for causal, result, weight in sorted(
            edges_trust, key=lambda x: (x[1] != param.target_index, x[1], -abs(x[2]))
        )
    ]
    synthesize_effect = [
        (index_map.get(int(idx)), target, score)
        for idx, score in synthesize_effect
        if idx in index_map and idx != 0
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_name = ".".join(Path(input_file).name.split(".")[:-1])
    pd.DataFrame(
        edges_trust, columns=["causal", "reason", "effect"]
    ).drop_duplicates().to_csv(
        output_dir / f"{input_name}.edges_trust.json", sep="\t", index=None
    )
    pd.DataFrame(synthesize_effect, columns=["causal", "reason", "effect"]).to_csv(
        output_dir / f"{input_name}.synthesize_effect.json", sep="\t", index=None
    )
    logging.info(f"results were saved in {output_dir.absolute().as_posix()}")


if __name__ == "__main__":
    app()
