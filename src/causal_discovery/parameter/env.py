#!/usr/bin/env python
# encoding:utf-8
import sys
import types
from functools import lru_cache
from os import environ
from time import sleep

import numpy as np


def set_gpu(memory_limit=4e3, gap_time=30, wait=True):
    """
        gpu自动设置，返回可用显存最多的卡号
        -1: 无可用
    """
    from subprocess import check_output

    def get_gpu_memory():
        os_str = (
            check_output("nvidia-smi -q -d Memory".split(" ")).decode("utf-8").split("\n")
        )
        tmp = [
            [y for y in os_str[idx : idx + 5] if "Free" in y]
            for idx, x in enumerate(os_str)
            if "GPU" in x
        ]
        tmp = [int(x[0].split()[2]) for x in tmp if x]
        return tmp

    while True:
        try:
            memory_gpu = get_gpu_memory()
            if max(memory_gpu) < memory_limit:
                print("Low Memory")
                if wait:
                    sleep(gap_time)
                else:
                    return -1
            else:
                gpu_free_num = np.argmax(memory_gpu)
                environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environ["CUDA_VISIBLE_DEVICES"] = str(gpu_free_num)
                print(f"set gpu {gpu_free_num}")
                return gpu_free_num
        except IndexError as unknown_exc:
            raise IndexError(sys.exc_info()[0]) from unknown_exc


@lru_cache(1)
def select_xp():
    """选择使用numpy还是cupy

    Returns:
        [Module]: numpy or cupy
    """
    try:
        import cupy as cp
    except (ModuleNotFoundError, ImportError):
        xp = np
        print("Using numpy(No cupy)")
    else:
        gpu_num = set_gpu(wait=False, memory_limit=1e3)  # 不等待GPU，有就用，没有就用CPU
        if gpu_num >= 0:
            cp.cuda.Device(0).use()  # 指定cuda某张卡可见，卡号重排为0
            xp = cp
            print(f"Using cupy(GPU-{gpu_num})")
        else:
            xp = np
            print("Using numpy(No usefull GPU)")
    return xp


def to_numpy(unknown_module: types.ModuleType, array):
    """转换array到numpy（cupy必须要用get方法，无法与numpy兼容，所以自定义一个函数来转换）

    Args:
        array ([np.array]): array的numpy类
    """
    if unknown_module.__name__ == "cupy":
        return unknown_module.asnumpy(array)
    if unknown_module.__name__ == "numpy":
        return array
    try:
        return np.array(array)
    except Exception as unknown_exc:
        raise Exception(sys.exc_info()[0]) from unknown_exc
