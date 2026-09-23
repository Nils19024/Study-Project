import os
import subprocess

import numpy as np
import torch
from loguru import logger


def get_gpu_with_most_free_mem():
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return np.argmax([int(value) for value in memory.splitlines()])


use_gpu = torch.cuda.is_available()
gpu_no = None
if use_gpu:
    gpu_no = 0 if "CUDA_VISIBLE_DEVICES" in os.environ else get_gpu_with_most_free_mem()
device = torch.device("cuda:{}".format(gpu_no) if use_gpu else "cpu")
logger.info("Running on {}", device)
