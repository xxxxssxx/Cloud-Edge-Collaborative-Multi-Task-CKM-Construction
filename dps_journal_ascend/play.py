import gc
import io
import time

import numpy as np
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint, plot_loss, colorize_and_mark, generate_downsample_mask, generate_block_mask, load_yaml, get_grid_img, setup_logging
import matplotlib.pyplot as plt
import controllable_generation
from sampling import LangevinCorrector, ReverseDiffusionPredictor
import yaml                                             ##!!!!!!!
from dps.measurements import get_noise, get_operator
from dps.condition_methods import get_conditioning_method
from dps.dps_utils.img_utils import clear_color, mask_generator
from functools import partial
import accelerate
from accelerate import Accelerator
import sys
import torch.distributed as dist
import os
import torch.distributed as dist
import shutil, subprocess, psutil


# 检查 npu-smi 是否能找到
npu_smi_path = shutil.which("npu-smi")
if npu_smi_path:
    print(f"npu-smi found at: {npu_smi_path}")
    # 执行命令
    subprocess.run(["npu-smi", "info"])
else:
    print("npu-smi not found in PATH.")

# accelerator = Accelerator()
# rank = accelerator.process_index
# setup_logging(rank, log_path="trainlog.log")
# print(f"Accelerate version: {accelerate.__version__}")

# world_size  = accelerator.num_processes

# local_obj = {"rank": rank, "value": rank * 10}

# # —— 核心：all_gather_object ————————————————
# gathered = [None] * world_size            # 先占位
# dist.all_gather_object(gathered, local_obj)
# # ———————————————————————————————————————————

# if accelerator.is_main_process:
#     print(f"World size = {world_size}")
#     for i, obj in enumerate(gathered):
#         print(f"From rank {i}: {obj}")

# utils.py

# def _parse_npu_smi_info():
#     """
#     仅适配旧版 ASCII 表格格式（示例见截图）
#     返回:
#         {npu_id: dict(power=…, temp=…, aicore=…, mem=(used, total),
#                       hbm=(used, total)) , …}
#     """
#     smi_bin = shutil.which("npu-smi")
#     if smi_bin is None:
#         logging.warning("npu-smi not found; skip NPU logging")
#         return {}

#     try:
#         txt = subprocess.check_output([smi_bin, "info"],
#                                       stderr=subprocess.STDOUT).decode()
#     except Exception as e:
#         logging.warning(f"{smi_bin} failed: {e}")
#         return {}

#     stats = {}
#     for line in txt.splitlines():
#         if not line.startswith("|"):          # 只处理表格行
#             continue
#         cols = [c.strip() for c in line.split("|")[1:-1]]
#         if len(cols) < 8 or not cols[0].isdigit():  # 不是数据行
#             continue
#         npu_id = int(cols[0])
#         power   = float(cols[3])                     # W
#         temp    = float(cols[4])                     # ℃
#         aicore  = float(cols[5])                     # %
#         mem_used, mem_tot = (float(x) for x in cols[6].split("/") )
#         hbm_used, hbm_tot = (float(x) for x in cols[7].split("/") )
#         stats[npu_id] = dict(
#             power=power, temp=temp, aicore=aicore,
#             mem=(mem_used, mem_tot), hbm=(hbm_used, hbm_tot)
#         )
#     return stats
# def log_resource_util(step):

#     cpu = psutil.cpu_percent(interval=None)
#     npu = _parse_npu_smi_info()

#     parts = [
#         f"NPU{idx}: {d['power']:>4.0f}W {d['temp']:>2.0f}°C "
#         f"AiCore{d['aicore']:>3.0f}% "
#         f"Mem{d['mem'][0]:>5.0f}/{d['mem'][1]:.0f}MB "
#         f"HBM{d['hbm'][0]:>5.0f}/{d['hbm'][1]:.0f}MB"
#         for idx, d in sorted(npu.items())
#     ]
#     logging.info(f"step {step:<6}| CPU {cpu:>5.1f}% | " + "  ".join(parts))


# log_resource_util(200)
# log_resource_util(400)

# if shutil.which("npu-smi") is None:
#     logging.warning("npu-smi is not available, please check your environment")
# else:   
#     logging.info("npu-smi is available")
#     logging.info(f"npu-smi path: {shutil.which('npu-smi')}")