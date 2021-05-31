import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def filter_corr(ret_df: pd.DataFrame, filter_num=0.95):
    df = ret_df[ret_df.columns]
    logging.info("get corr df.")
    cov_df = np.corrcoef(df)
    df_index = list(df.index)
    logging.info(f"filter corr: {filter_num}")
    x_df, y_df = np.where(cov_df >= filter_num)
    filter_idx = set()
    for x, y in zip(x_df, y_df):
        if x >= y:
            continue
        if (df_index[x] in filter_idx) or (df_index[y] in filter_idx):
            continue
        # 谁与行业指标相关性高，留谁
        if x == 0 and y != 0:  # 一定是x<=y，所以只判断x==0的情况就行，保留行业指标
            filter_idx.add(df_index[y])
            continue
        filter_idx.add(
            min(
                [(cov_df[0][x], df_index[x]), (cov_df[0][y], df_index[y])],
                key=lambda z: abs(z[0]),
            )[1]
        )
    df = df.loc[filter(lambda x: x not in filter_idx, df_index)]
    return df


def numeric_and_fill(df: pd.DataFrame, fill_methods: list):
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df.loc[:, col], errors="coerce")  # 转为数值，非数值会转为NaN
    logging.info(f"before dropna: {df.shape}")
    df.dropna(how="all", axis=1, inplace=True)  # 剔除所有值都非数值的变量
    logging.info(f"after drop columns: {df.shape}")
    df.dropna(how="all", axis=0, inplace=True)  # 剔除无数据的采样记录
    logging.info(f"after drop indexes: {df.shape}")
    logging.info(f"Null num: {np.sum(np.isnan(df.to_numpy()))}")
    for method in fill_methods:
        if isinstance(method, float):
            df = df.fillna(value=method)
        elif method in {"backfill", "bfill", "pad", "ffill", None}:
            df = df.fillna(axis=1, method=method)
        else:
            logging.trace(f"Unknown fill method {method}.")
    return df


def vertical_2_horizontal(df: pd.DataFrame, fill_methods: list):
    df.columns = ["time", "id", "value"]
    logging.info(f"samples: {df.shape[0]}")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # 转为数值，非数值会转为NaN
    df.dropna(inplace=True)  # 剔除非数字项

    dates = sorted(df["time"].unique())
    date2id = {date: idx for idx, date in enumerate(dates)}
    len_date = len(date2id)
    id_date_value = {}
    for date, idx, value in tqdm(
        zip(df["time"], df["id"], df["value"]), desc="group", total=df.shape[0]
    ):
        id_date_value.setdefault(idx, [None] * len_date)
        date_id = date2id[date]
        id_date_value[idx][date_id] = value

    ret_df = pd.DataFrame(
        np.array(list(id_date_value.values())).T, columns=list(id_date_value.keys())
    )
    ret_df["time"] = dates
    for method in fill_methods:
        if isinstance(method, float):
            ret_df = ret_df.fillna(value=method)
        elif method in {"backfill", "bfill", "pad", "ffill", None}:
            ret_df = ret_df.fillna(axis=1, method=method)
        else:
            logging.trace(f"Unknown fill method {method}.")
    ret_df.drop(["time"], axis=1, inplace=True)
    return ret_df


def get_matrix_data(
    target: str,
    ret_df: pd.DataFrame = None,
    triple_df: pd.DataFrame = None,
    corr_filter: bool = False,
    used_cache_file: str = "",
    cache: bool = False,
    fill_methods: list = ["ffill", "bfill"],
    need_norm: bool = False,
):
    """
    纵表转横表，基于相关系数筛选指标
    return: 行索引为指标，列索引为日期
    """
    if not used_cache_file or not Path(used_cache_file).exists():
        if ret_df is None and triple_df is None:
            raise ValueError("Need ret_df or triple_df.")
        if ret_df is None:
            assert len(triple_df.columns) == 3, "Need triple DataFrame."
            ret_df = vertical_2_horizontal(triple_df, fill_methods)
        else:
            ret_df = numeric_and_fill(ret_df, fill_methods=fill_methods)
        pop_list = []
        logging.info("norm")
        for idx, row in ret_df.items():
            if row.max() != row.min():
                row = (row - row.min()) / (row.max() - row.min()) if need_norm else row
            else:
                pop_list.append(idx)
                continue
            ret_df[idx] = row
        # 剔除常数序列
        for idx in pop_list:
            ret_df.pop(idx)
        if cache:
            ret_df.to_pickle(used_cache_file or "./dataframe.cache.pkl")
            logging.info("save")
    else:
        ret_df = pd.read_pickle(used_cache_file)
        logging.info(f"use cache {used_cache_file}")

    target_series = ret_df.pop(target)
    ret_df.insert(0, target, target_series)  # 确保行业指数在第一个！！！不会被剔除

    if not corr_filter:
        return ret_df.T
    return filter_corr(ret_df.T)


if __name__ == "__main__":
    print(
        get_matrix_data(
            "A",
            triple_df=pd.DataFrame(
                [
                    [
                        "20200101",
                        "20200102",
                        "20200103",
                        "20200101",
                        "20200102",
                        "20200103",
                    ],
                    ["A", "A", "A", "B", "B", "B"],
                    [1, 2, 2.5, 3, 10, 222],
                ]
            ).T,
            corr_filter=True,
        )
    )
