# gpu自动设置
def set_gpu(memory_limit=4e3, gap_time=30):
    """
        gpu自动设置
    """
    from os import popen, environ
    # from sys import exc_info
    from numpy import argmax
    from time import sleep

    while True:
        try:
            memory_gpu = [
                int(x.split()[2])
                for x in popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free")
                .read()
                .split("\n")
                if x
            ]
            if max(memory_gpu) < memory_limit:
                print("Low Memory")
                sleep(gap_time)
            else:
                gpu_free_num = argmax(memory_gpu)
                environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environ["CUDA_VISIBLE_DEVICES"] = str(gpu_free_num)
                print(f"set gpu {gpu_free_num}")
                return gpu_free_num
        except IndexError as e:
            raise IndexError(e)