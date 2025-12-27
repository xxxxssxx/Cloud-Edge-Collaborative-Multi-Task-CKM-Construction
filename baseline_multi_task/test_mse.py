#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run MSE on the whole CKM_gain_128 *eval* split and report
average MSE (ref_img vs. recon_img).  Multi-card evaluation is handled
"""

import gc
import io
import os
import time

import numpy as np
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import load_yaml, setup_logging
import matplotlib.pyplot as plt
from dps.measurements import get_noise, get_operator
from dps.dps_utils.img_utils import mask_generator
from functools import partial
from accelerate import Accelerator
import sys
import torch.distributed as dist
import os
import importlib.util
import ml_collections
from PIL import Image, ImageDraw, ImageFont
import pandas as pd   
import torch.distributed as dist
from itertools import chain
import hashlib

# ----------------------- helper ---------------------------------- #

def load_py_config(file_path: str):
    spec = importlib.util.spec_from_file_location("config_module", file_path)
    cfg  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg.get_config()

def chw_to_hwc(x: torch.Tensor) -> np.ndarray:
    """(C,H,W) -> (H,W,C) & CPU numpy float32"""
    return x.permute(1, 2, 0).contiguous().cpu().numpy()

def seed_from_name(name):
    # 取 sha256 前 8 个十六进制位，转成 int
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)

# -------------------- main evaluation ----------------------------- #

def evaluate_baseline_multitask_mse(config: ml_collections.ConfigDict,
                             workdir="./",
                             ):
    """
    Run UNet on the whole eval set and print/return the average MSE.
    """
    BASE_SEED = 12345
    
    accelerator = Accelerator(even_batches=False, split_batches=False)
    config.device = accelerator.device
    rank = accelerator.process_index
    setup_logging(rank, log_path="logtest.log")

    logging.info("Starting test multi-task.\n")
    # configs about the forward operator
    # 从这里能看出来，网络是和前向运算相关的。
    IP_box_config = load_yaml("./configs/inpainting_box_config.yaml")
    IP_random_config = load_yaml("./configs/inpainting_random_config.yaml")
    SR_config = load_yaml("./configs/super_resolution_config.yaml")
    JDR_config = load_yaml("./configs/joint_degradation_recovery_config.yaml")
    task_config = IP_box_config
    measure_config = task_config['measurement']

    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    UNet = mutils.create_model(config)

    optimizer = losses.get_optimizer(config, UNet.parameters())
    ema = ExponentialMovingAverage(UNet.parameters(),
                                decay=config.model.ema_rate)
    
    ckpt_num = config.test.checkpoint_num
    ckpt_filename = workdir + f"checkpoints/checkpoint_{ckpt_num}.pth"
    checkpoint = torch.load(ckpt_filename, map_location=config.device)

    accelerator.unwrap_model(UNet).load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    ema.load_state_dict(checkpoint['ema'])

    ema.store(UNet.parameters())
    ema.copy_to(UNet.parameters())
    state = dict(model=UNet, ema=ema, optimizer=optimizer)

    # Device setting
    logging.info(f"device: {config.device}\n")

    UNet.eval()

    # Prepare Operator and noise
    operator = get_operator(device=config.device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logging.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}\n")

    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)
    UNet, eval_ds = accelerator.prepare(UNet, eval_ds)

    # ---------- 5.  逐批评估 ----------
    mse_sum_local  = torch.tensor(0.0, device=config.device)
    mse_gain_sum_local = torch.tensor(0.0, device=config.device)
    mse_sin_sum_local = torch.tensor(0.0, device=config.device)
    n_img_local    = torch.tensor(0,   device=config.device, dtype=torch.long)

    results_local = []             # [(idx, filename, mse_value), ...]
    test_dir  = os.path.join(workdir,
                                config.eval.eval_folder,
                                "test_visualization")
    os.makedirs(test_dir, exist_ok=True)

    count = 0
    for batch in eval_ds:

        ground_truth = batch['image'].to(config.device)  # torch.tensor
        building_batch = ground_truth[:, 2:3, :, :].to(torch.float32)  # (B, 1, H, W)
        ground_truth = ground_truth[:, 0:2, :, :]  # 保留增益和角度
        
        # AoA转换为sin
        p_min = 1/19.0  # 20 / (180-(-200))
        AoA_batch = ground_truth[:, 1:2, :, :].to(torch.float32)  # (B,1,H,W)
        angle_batch = torch.zeros_like(AoA_batch, dtype=torch.float32)
        angle_batch[building_batch==0] = ((AoA_batch[building_batch==0] - p_min) / (1.0 - p_min) * 360.0 - 180.0)
        rad_batch = torch.deg2rad(angle_batch)
        sin_batch = torch.zeros_like(AoA_batch, dtype=torch.float32)
        sin_batch[building_batch==0] = torch.sin(rad_batch[building_batch==0])
        
        # sin_batch中，将非建筑处的sin值从-1~1映射至0.3~1，建筑处的值为0
        # pixel [0.3, 1], sin [-1, 1]   pixel = 0.35*sin + 0.65  sin = (1/0.35)*pixel - 0.65/0.35
        sin_batch[building_batch==0] = 0.35 * sin_batch[building_batch==0] + 0.65
        sin_batch[building_batch==1] = 0 
        
        ground_truth[:, 1:2, :, :] = sin_batch

        ground_truth = scaler(ground_truth)  # 这里的scaler是一个函数，将数据缩放到[-1,1]之间

        # Forward measurement model (Ax + n)
        if measure_config['operator']['name'] == "inpainting":
            B, _, H, W = ground_truth.shape
            masks = []
            for i in range(B):
                # 用 filename 更稳；若没有就用 index
                name_i = batch["filename"][i] if "filename" in batch else str(batch["index"][i].item())
                seed_i = BASE_SEED ^ seed_from_name(name_i)

                np_rng_i = np.random.default_rng(seed_i)
                torch_gen_i = torch.Generator(device=ground_truth.device).manual_seed(seed_i)

                mg_i = mask_generator(
                    **measure_config['mask_opt'],
                    np_rng=np_rng_i,
                    torch_gen=torch_gen_i
                )
                m_i, ok_i = mg_i(ground_truth[i:i+1, 0:1, :, :])
                if not ok_i:
                    raise ValueError(f"Mask generation failed for sample {name_i}.")
                masks.append(m_i)

            mask = torch.cat(masks, dim=0)                 # [B, 1, H, W]
            degrade_image = operator.forward(ground_truth, mask=mask.repeat(1, ground_truth.shape[1], 1, 1))
        else:
            degrade_image = operator.forward(ground_truth)
        degrade_image = noiser(degrade_image)

        # 把degrade_image通过网络
        eval_batch = {
                'ground_truth': ground_truth,
                'degrade_image': degrade_image,
            }
        eval_step_fn = losses.get_step_fn(train=False, optimize_fn=None)
        eval_loss, pred = eval_step_fn(state, eval_batch, accelerator)

        # 反归一化到 [0,1]
        pred = inverse_scaler(pred)
        ground_truth = inverse_scaler(ground_truth)
        degrade_image = inverse_scaler(degrade_image)

        lb = 0.135
        toolow = pred.abs() < lb
        # 把过低的地方置零
        pred[toolow] = 0.0
        
        ground_truth = torch.clamp(ground_truth, 0.0, 1.0)
        pred = torch.clamp(pred, 0.0, 1.0)
        degrade_image = torch.clamp(degrade_image, 0.0, 1.0)

        # 计算 mse
        if measure_config['operator']['name'] == 'inpainting':
            # mask == 1 代表已观测像素，0 代表被遮挡（需重建）
            target_region = (1.0 - mask)           # 同 shape, float
            sq_err_map    = (pred - ground_truth) ** 2 * target_region
            sq_err_map_gain = sq_err_map[:, 0:1, :, :]
            sq_err_map_sin = sq_err_map[:, 1:2, :, :]
            pixel_cnt     = target_region.sum(dim=(1, 2, 3)).clamp_min(1.0)
            mse = sq_err_map.sum(dim=(1, 2, 3)) / pixel_cnt
            mse_gain = sq_err_map_gain.sum(dim=(1, 2, 3)) / (pixel_cnt/2)  # 应只对图像的长宽维度计数。pixel_cnt这个变量在通道数上也计数了。
            mse_sin = sq_err_map_sin.sum(dim=(1, 2, 3)) / (pixel_cnt/2)
        else:
            mse = ((pred - ground_truth) ** 2).mean(dim=(1, 2, 3))   # 全图
            mse_gain = ((pred[:, 0:1, :, :] - ground_truth[:, 0:1, :, :]) ** 2).mean(dim=(1, 2, 3))
            mse_sin = ((pred[:, 1:2, :, :] - ground_truth[:, 1:2, :, :]) ** 2).mean(dim=(1, 2, 3))
        
        degrade_image_gain = degrade_image[:, 0:1, :, :]
        degrade_image_sin = degrade_image[:, 1:2, :, :]
        if measure_config['operator']['name'] == "inpainting":
            red = torch.tensor([1.0, 0.0, 0.0],            # shape [3]
                device=degrade_image_gain.device,
                dtype=degrade_image_gain.dtype)[None, :, None, None]   # [1,3,1,1]
            mask3 = mask.expand(-1, 3, -1, -1)            # [B, 3, H, W]
            degrade_image_gain_rgb = degrade_image_gain.repeat(1, 3, 1, 1)          # [B, 3, H, W]
            degrade_image_sin_rgb = degrade_image_sin.repeat(1, 3, 1, 1)          # [B, 3, H, W]
            degrade_image_gain = torch.where(mask3 == 0, red, degrade_image_gain_rgb)
            degrade_image_sin = torch.where(mask3 == 0, red, degrade_image_sin_rgb)

        logging.info(f"mse example: {mse.cpu().numpy()}")
        logging.info(f"img  range: {pred.min():.2f} ~ {pred.max():.2f}")
        logging.info(f"ref  range: {ground_truth.min():.2f} ~ {ground_truth.max():.2f}")

        mse_sum_local += mse.sum()  # scalar
        mse_gain_sum_local  += mse_gain.sum()
        mse_sin_sum_local  += mse_sin.sum()
        
        n_img_local   += torch.tensor(mse.numel(), device=config.device)
        # logging.info("debug1")

        filenames = batch["filename"]       # list[str]  length == batch
        indices   = batch["index"]          # tensor     length == batch

        for i in range(len(filenames)):
            
            mse_val = mse[i].item()
            mse_gain_val = mse_gain[i].item()
            mse_sin_val = mse_sin[i].item()

            results_local.append(
                (indices[i].item(), filenames[i], mse_val, mse_gain_val, mse_sin_val)
            )  # 长为batch数量的数组。每个元素是一个5元组。

            # ---------- residual image ----------

            resid = torch.abs(pred[i] - ground_truth[i])          # (C,H,W) in [0,1]
            resid_gain = resid[0:1, :, :]
            resid_sin = resid[1:2, :, :]
            resid_np_gain = (resid_gain.cpu().numpy() * 255).astype("uint8").squeeze()
            resid_np_sin = (resid_sin.cpu().numpy() * 255).astype("uint8").squeeze()
            resid_pil_gain = Image.fromarray(resid_np_gain)
            resid_pil_sin = Image.fromarray(resid_np_sin)

            # annotate with MSE (top-left corner)
            draw_gain = ImageDraw.Draw(resid_pil_gain)
            draw_gain.text((4, 4), f"MSE:{mse_val:.3e}", fill=255)   # white text
            draw_sin = ImageDraw.Draw(resid_pil_sin)
            draw_sin.text((4, 4), f"MSE:{mse_val:.3e}", fill=255)   # white text

            resid_pil_gain.save(os.path.join(test_dir, f"{indices[i]}_residual_gain.png"))
            resid_pil_sin.save(os.path.join(test_dir, f"{indices[i]}_residual_sin.png"))
            
            if measure_config['operator']['name'] == "inpainting":
                plt.imsave(os.path.join(test_dir, f"{indices[i]}_input_gain.png"), chw_to_hwc(degrade_image_gain[i]), vmin=0, vmax=1)
                plt.imsave(os.path.join(test_dir, f"{indices[i]}_input_sin.png"), chw_to_hwc(degrade_image_sin[i]), vmin=0, vmax=1)
            else:
                plt.imsave(os.path.join(test_dir, f"{indices[i]}_input_gain.png"), degrade_image_gain[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                plt.imsave(os.path.join(test_dir, f"{indices[i]}_input_sin.png"), degrade_image_sin[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ground_truth_gain = ground_truth[:, 0:1, :, :]
            ground_truth_sin = ground_truth[:, 1:2, :, :]
            plt.imsave(os.path.join(test_dir, f"{indices[i]}_label_gain.png"), ground_truth_gain[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.imsave(os.path.join(test_dir, f"{indices[i]}_label_sin.png"), ground_truth_sin[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            pred_gain = pred[:, 0:1, :, :]
            pred_sin = pred[:, 1:2, :, :]
            plt.imsave(os.path.join(test_dir, f"{indices[i]}_recon_gain.png"), pred_gain[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.imsave(os.path.join(test_dir, f"{indices[i]}_recon_sin.png"), pred_sin[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)

            logging.info(f"Processed batch {count}, sample {i}")

        logging.info(f"Processed batch {count}")
        count += 1

    # accelerator.wait_for_everyone()  # 等待所有进程完成
    # convert to tensors for easy all_gather
    world_size = accelerator.num_processes
    # logging.info("debug3")
    if world_size > 1:
        list_indices_local = [r[0] for r in results_local]      # 变成 Python list
        indices_gather = [None] * world_size  # 占位
        dist.all_gather_object(indices_gather, list_indices_local)  # 收集，变量可不同长度
        indices_flat = list(chain.from_iterable(indices_gather))  # 扁平化
        all_indices = torch.tensor(indices_flat, device=config.device, dtype=torch.long)  # 转张量，整形

        list_mse_local = [r[2] for r in results_local]      # 变成 Python list
        mse_gather = [None] * world_size  # 占位
        dist.all_gather_object(mse_gather, list_mse_local)  # 收集，变量可不同长度
        mse_flat = list(chain.from_iterable(mse_gather))  # 扁平化
        all_mse = torch.tensor(mse_flat, device=config.device)  # 转张量
        
        list_mse_gain_local = [r[3] for r in results_local]      # 变成 Python list
        mse_gain_gather = [None] * world_size  # 占位
        dist.all_gather_object(mse_gain_gather, list_mse_gain_local)  # 收集，变量可不同长度
        mse_gain_flat = list(chain.from_iterable(mse_gain_gather))  # 扁平化
        all_mse_gain = torch.tensor(mse_gain_flat, device=config.device)  # 转张量
        
        list_mse_sin_local = [r[4] for r in results_local]      # 变成 Python list
        mse_sin_gather = [None] * world_size  # 占位
        dist.all_gather_object(mse_sin_gather, list_mse_sin_local)  # 收集，变量可不同长度
        mse_sin_flat = list(chain.from_iterable(mse_sin_gather))  # 扁平化
        all_mse_sin = torch.tensor(mse_sin_flat, device=config.device)  # 转张量
        
        # filenames are strings; easiest: keep them in a python list on each rank
        # then gather via torch.distributed.all_gather_object

        filenames_gather = [None] * world_size
        filenames_local = [r[1] for r in results_local]  # list[str] on this rank
        dist.all_gather_object(filenames_gather, filenames_local)   # -> filenames_gather == [list_on_rank0, list_on_rank1, ...]
        all_filenames = list(chain.from_iterable(filenames_gather))
        # logging.info("debug4")
    else:
        all_indices   = torch.tensor([r[0] for r in results_local], device=config.device, dtype=torch.long)
        all_mse       = torch.tensor([r[2] for r in results_local], device=config.device)
        all_mse_gain  = torch.tensor([r[3] for r in results_local], device=config.device)
        all_mse_sin   = torch.tensor([r[4] for r in results_local], device=config.device)
        all_filenames = [r[1] for r in results_local]

    # ---------- 6.  汇总各卡 ----------
    packed = torch.stack([mse_sum_local, n_img_local, mse_gain_sum_local, mse_sin_sum_local])  # 先stack成一个变量再reduce，避免竞争
    packed = accelerator.reduce(packed, reduction="sum")
    if accelerator.is_main_process:
        mse_mean = packed[0].item() / packed[1].item()
        mse_gain_mean = packed[2].item() / packed[1].item()
        mse_sin_mean = packed[3].item() / packed[1].item()
        logging.info(f"\n==== Average MSE over eval set: {mse_mean:.6e} ====\n")
        logging.info(f"\n==== Average MSE gain over eval set: {mse_gain_mean:.6e} ====\n")
        logging.info(f"\n==== Average MSE sin over eval set: {mse_sin_mean:.6e} ====\n")

    if accelerator.is_main_process:
        df = pd.DataFrame({
            "index":     all_indices.cpu().numpy(),
            "filename":  all_filenames,   # flatten
            "mse":       all_mse.cpu().numpy(),
            "mse_gain":  all_mse_gain.cpu().numpy(),
            "mse_sin":   all_mse_sin.cpu().numpy(),
        }).sort_values("index")                    # restore original order

        csv_path = os.path.join(workdir, "mse_per_sample.csv")
        df.to_csv(csv_path, index=False)
        # lightweight TXT version
        df.to_csv(csv_path.replace(".csv", ".txt"), index=False, sep="\t")

        # Top-N worst reconstructions
        top_k = 20
        logging.info(f"\n--- Worst {top_k} samples by MSE ---")
        logging.info(df.nlargest(top_k, "mse")[["filename", "mse"]])


# ------------------ script entry --------------------------------- #

if __name__ == "__main__":


    config_path = "./configs/CKM_gain_AoA_128_ncsnpp_deep_continuous.py"
    config = load_py_config(config_path)
    # 如需临时修改 batch_size、eval.batch_size 等可在此覆盖
    config.eval.batch_size = 6
    config.test.checkpoint_num = 10
    evaluate_baseline_multitask_mse(config, workdir="./")

# linux:
'''
cd /opt/dpcvol/models/xsx/score_sde_pytorch_ascend
pip install certifi==2024.7.4
pip install ml_collections
accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision=no \
  test_mse.py \
'''

# windows:
'''
accelerate launch `
 --num_processes=1 `
 --num_machines=1 `
 --machine_rank=0 `
 --mixed_precision=no `
test_mse.py
'''