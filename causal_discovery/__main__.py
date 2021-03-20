import logging
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import torch
import typer
from peon.log import config_log

from causal_discovery.algorithm.local_ng_cd.local_ng_cd import local_ng_cd
from causal_discovery.data_prepare.matrix_data import get_matrix_data
from causal_discovery.parameter.algo import DataType, LocalNgCdParam
from causal_discovery.parameter.error import DataTypeError

app = typer.Typer()
xp = cp if torch.cuda.is_available() else np


@app.command()
def run_local_ng_cd(
        input_file: str,
        target: str,
        data_type: DataType,
        sep: str = ",",
        index_col: str = None,
        header: int = None,
        output_dir: str = "./output",
        log_root: str = "./logs",
        verbose: bool = True,
        candidate_two_step: bool = False
):
    """[因果发现算法：Local-NG-CD, 作者：张坤, 年份：2020]

    Args:\n
        input_file (str): [输入文件地址，csv格式]\n
        target (str): [目标变量名]\n
        data_type (DataType): [数据类型：triple（三元组[样本索引，变量名，值]）、matrix（矩阵，行索引为变量名，列索引为样本索引）、matrixT（矩阵，行索引为样本索引，列索引为变量名）]\n
        sep (str, optional): [csv分隔符]. Defaults to ",".\n
        index_col (str, optional): [读取csv的index索引]. Defaults to None.\n
        header (str, optional): [读取csv的header索引]. Defaults to None.\n
        output_dir (str, optional): [输出目录]. Defaults to "./output".\n
        log_root (str, optional): [日志目录]. Defaults to "./logs".\n
        verbose (bool, optional): [是否打印日志到控制台]. Defaults to True.\n
        candidate_two_step (bool, optional): [是否启用2跳关系筛选]. Defaults to False.\n

    Raises:\n
        DataTypeError: [数据类型错误]
    """
    config_log(
        "local_ng_cd",
        "local_ng_cd",
        log_root=log_root,
        print_terminal=verbose,
        enable_monitor=True,
    )
    if data_type == DataType.triple:
        triple_data = pd.read_csv(input_file, index_col=index_col, header=header, sep=sep)
        matrix_data = get_matrix_data(target, triple_df=triple_data, corr_filter=True, used_cache_file="")
    elif data_type == DataType.matrix:
        matrix_data = pd.read_csv(input_file, index_col=index_col, header=header, sep=sep).T
    elif data_type == DataType.matrixT:
        matrix_data = pd.read_csv(input_file, index_col=index_col, header=header, sep=sep)
    else:
        raise DataTypeError(f"Unknown datatype: {data_type}.")
    if target not in matrix_data.columns:
        raise DataTypeError(
            "Choosing matrix/matrixT datatype, but no target in indexes or columns, please check arg: index_col/header")

    # 暴露出来允许修改的参数
    param = LocalNgCdParam()
    param.candidate_two_step = candidate_two_step
    param.target_index = list(matrix_data.columns).index(target)

    index_map = dict(enumerate(matrix_data.columns))
    edges_trust, synthesize_effect = local_ng_cd(xp.asarray(matrix_data.T.to_numpy()), param, synthesize=True)
    edges_trust = [
        (
            index_map.get(int(causal), "N"),
            index_map.get(int(result), "N"),
            weight,
        )
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
    pd.DataFrame(edges_trust, columns=["causal", "reason", "effect"]).to_csv(
        output_dir / "edges_trust.json", sep="\t", index=None)
    pd.DataFrame(synthesize_effect, columns=["causal", "reason", "effect"]).to_csv(
        output_dir / "synthesize_effect.json", sep="\t", index=None)
    logging.info(f"results were saved in {output_dir.absolute().as_posix()}")

if __name__ == "__main__":
    from causal_discovery.parameter.env import set_gpu
    if torch.cuda.is_available():
        gpu_num = set_gpu()
        cp.cuda.Device(gpu_num).use()
    app()