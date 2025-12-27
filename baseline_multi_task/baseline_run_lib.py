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
from utils import save_checkpoint, restore_checkpoint, plot_loss, colorize_and_mark, generate_downsample_mask, generate_block_mask, load_yaml, get_grid_img, setup_logging, BS_building_detection
import matplotlib.pyplot as plt
from dps.measurements import get_noise, get_operator
from dps.condition_methods import get_conditioning_method
from functools import partial
from dps.dps_utils.img_utils import clear_color, mask_generator
from accelerate import Accelerator
import sys
import torch.distributed as dist
import random

FLAGS = flags.FLAGS

def train(config, workdir):
    """
    Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and logs. 
    """

    accelerator = Accelerator(gradient_accumulation_steps=config.training.grad_accum_steps)
    config.device = accelerator.device
    print(f"device: {config.device}")
    rank = accelerator.process_index
    setup_logging(rank, log_path="logtrain.log")

    # Must be plain integer seconds, ≥120 and ≤7200
    logging.info("start\n")
    os.environ["HCCL_EXEC_TIMEOUT"] = "3600"  # this really works

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    # 从这里能看出来，网络是和前向运算相关的。
    IP_box_config = load_yaml("./configs/inpainting_box_config.yaml")
    IP_random_config = load_yaml("./configs/inpainting_random_config.yaml")
    SR_config = load_yaml("./configs/super_resolution_config.yaml")
    JDR_config = load_yaml("./configs/joint_degradation_recovery_config.yaml")
    # 将他们组成一个列表
    task_configs = [IP_box_config]  # 单任务时只把其中一个放入任务备选列表
    # task_configs = [IP_box_config, IP_random_config, SR_config, JDR_config]  # 多任务时把所有任务都放入任务备选列表
    

    # Initialize model.
    UNet = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, UNet.parameters())

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Build data iterators from our torch-based dataset loaders
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization)  # 这里已经应用了transform
    
    # prepare components for distributed training
    UNet, optimizer, train_ds, eval_ds = accelerator.prepare(UNet, optimizer, train_ds, eval_ds)
    ema = ExponentialMovingAverage(UNet.parameters(), decay=config.model.ema_rate)

    if not os.path.exists(checkpoint_meta_dir):
        logging.warning(f"No checkpoint found at {checkpoint_meta_dir}.")
        initial_iter_step = 0
        initial_update_step = 0
        epoch_avg_val_losses = []
    else:
        # 1. Load only on rank 0
        if accelerator.is_main_process:
            raw = torch.load(checkpoint_meta_dir, map_location=config.device)
        else:
            raw = None

        # 2. Broadcast the Python object list [raw] from rank 0 → everyone
        if dist.is_available() and dist.is_initialized():
            obj = [raw]
            dist.broadcast_object_list(obj, src=0)
            raw = obj[0]

        # 3. Now every rank has the same dict in `raw`
        checkpoint = raw

        accelerator.unwrap_model(UNet).load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ema.load_state_dict(checkpoint['ema'])
        initial_iter_step = checkpoint['iter_step']
        initial_update_step = checkpoint['update_step']
        epoch_avg_val_losses = checkpoint["epoch_avg_val_losses"]

    state = dict(optimizer=optimizer, model=UNet, ema=ema, iter_step=initial_iter_step, update_step=initial_update_step, epoch_avg_val_losses=epoch_avg_val_losses)

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_step_fn(train=False, optimize_fn=optimize_fn)

    logging.info(f"Starting training loop at update step {initial_update_step}, iter step {initial_iter_step}.\n")
    for iter_step in range(initial_iter_step, config.training.num_iter_step + 1):
        # Because we use a PyTorch DataLoader, next(train_iter) yields a dict {'image':..., 'label':...}
        try:
            batch_dict = next(train_iter)
        except StopIteration:
            # If we run out of data, reinitialize the iterator
            train_iter = iter(train_ds)
            batch_dict = next(train_iter)
        batch = batch_dict['image'].to(config.device).float()
        building_batch = batch[:, 2:3, :, :].to(torch.float32)  # (B, 1, H, W)
        batch = batch[:, 0:2, :, :]  # 保留增益和角度
        
        # AoA转换为sin
        p_min = 1/19.0  # 20 / (180-(-200))
        AoA_batch = batch[:, 1:2, :, :].to(torch.float32)  # (B,1,H,W)
        angle_batch = torch.zeros_like(AoA_batch, dtype=torch.float32)
        angle_batch[building_batch==0] = ((AoA_batch[building_batch==0] - p_min) / (1.0 - p_min) * 360.0 - 180.0)
        rad_batch = torch.deg2rad(angle_batch)
        sin_batch = torch.zeros_like(AoA_batch, dtype=torch.float32)
        sin_batch[building_batch==0] = torch.sin(rad_batch[building_batch==0])
        
        # sin_batch中，将非建筑处的sin值从-1~1映射至0.3~1，建筑处的值为0
        # pixel [0.3, 1], sin [-1, 1]   pixel = 0.35*sin + 0.65  sin = (1/0.35)*pixel - 0.65/0.35
        sin_batch[building_batch==0] = 0.35 * sin_batch[building_batch==0] + 0.65
        sin_batch[building_batch==1] = 0 
        
        batch[:, 1:2, :, :] = sin_batch

        batch = scaler(batch)  # 这里的scaler是一个函数，将数据缩放到[-1,1]之间
        
        # Prepare Operator and noise
        task_config  = random.choice(task_configs)  # 随机任务
        measure_config = task_config['measurement']
        operator = get_operator(device=config.device, **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])
        logging.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}\n")
            # 如果inpainting， 对measurement_cond_fn重新包装，传入mask
        if measure_config['operator'] ['name'] == 'inpainting':
            mask_gen = mask_generator(
                **measure_config['mask_opt']
            )
            mask, success = mask_gen(batch[:, 0:1, :, :])  # [B, 1, H, W]
            if not success:
                raise ValueError("Mask generation failed.")
            degrade_batch = operator.forward(batch, mask=mask.repeat(1, batch.shape[1], 1, 1))  # -1~1
        else:
            degrade_batch = operator.forward(batch)  # -1~1
        degrade_batch = noiser(degrade_batch)

        # 把batch degrade_batch拼成字典
        batch_train = {
            'ground_truth': batch,
            'degrade_image': degrade_batch,
        }

        # one training step
        state['iter_step'] = iter_step + 1
        loss, _ = train_step_fn(state, batch_train, accelerator)

        # log the  training loss
        if state['update_step'] % config.training.log_freq == 0 and accelerator.sync_gradients:
            logging.info(f"step: {state['update_step']}, training_loss: {loss.item():.5e}\n")

        # # Save a checkpoint to resume later
        if accelerator.is_main_process and state['update_step'] != 0 and state['update_step'] % config.training.snapshot_freq_for_preemption == 0 and accelerator.sync_gradients:
            # 只在主进程保存
            logging.info(f"iter step: {state['iter_step']}, update step: {state['update_step']}\n")
            save_checkpoint(checkpoint_meta_dir, state, accelerator)
            logging.info(f"checkpoint-meta saved.\n")
        accelerator.wait_for_everyone()

        # Evaluate
        if state['update_step'] % config.training.eval_freq == 0 and accelerator.sync_gradients:
            try:
                eval_batch_dict = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_ds)
                eval_batch_dict = next(eval_iter)
            eval_batch = eval_batch_dict['image'].to(config.device).float()
            eval_building_batch = eval_batch[:, 2:3, :, :].to(torch.float32)  # (B, 1, H, W)
            eval_batch = eval_batch[:, 0:2, :, :]  # 保留增益和角度
            
            # AoA转换为sin
            p_min = 1/19.0  # 20 / (180-(-200))
            eval_AoA_batch = eval_batch[:, 1:2, :, :].to(torch.float32)  # (B,1,H,W)
            eval_angle_batch = torch.zeros_like(eval_AoA_batch, dtype=torch.float32)
            eval_angle_batch[eval_building_batch==0] = ((eval_AoA_batch[eval_building_batch==0] - p_min) / (1.0 - p_min) * 360.0 - 180.0)
            eval_rad_batch = torch.deg2rad(eval_angle_batch)
            eval_sin_batch = torch.zeros_like(eval_AoA_batch, dtype=torch.float32)
            eval_sin_batch[eval_building_batch==0] = torch.sin(eval_rad_batch[eval_building_batch==0])
            
            # sin_batch中，将非建筑处的sin值从-1~1映射至0.3~1，建筑处的值为0
            # pixel [0.3, 1], sin [-1, 1]   pixel = 0.35*sin + 0.65  sin = (1/0.35)*pixel - 0.65/0.35
            eval_sin_batch[eval_building_batch==0] = 0.35 * eval_sin_batch[eval_building_batch==0] + 0.65
            eval_sin_batch[eval_building_batch==1] = 0 
            
            eval_batch[:, 1:2, :, :] = eval_sin_batch
            eval_batch = scaler(eval_batch)  # 这里的scaler是一个函数，将数据缩放到[-1,1]之间
            # 如果inpainting， 对measurement_cond_fn重新包装，传入mask
            if measure_config['operator'] ['name'] == 'inpainting':
                mask_gen = mask_generator(
                    **measure_config['mask_opt']
                )
                mask, success = mask_gen(eval_batch[:, 0:1, :, :])  # [B, 1, H, W]
                if not success:
                    raise ValueError("Mask generation failed.")
                eval_degrade_batch = operator.forward(eval_batch, mask=mask.repeat(1, eval_batch.shape[1], 1, 1))  # -1~1
            else:
                eval_degrade_batch = operator.forward(eval_batch)  # -1~1
            eval_degrade_batch = noiser(eval_degrade_batch)

            # 把batch degrade_batch拼成字典
            batch_eval = {
                'ground_truth': eval_batch,
                'degrade_image': eval_degrade_batch,
            }
            eval_loss, pred = eval_step_fn(state, batch_eval, accelerator)  # 有修改
            pred = inverse_scaler(pred)
            eval_batch = inverse_scaler(eval_batch)
            eval_degrade_batch = inverse_scaler(eval_degrade_batch)
            logging.info(f"step: {state['update_step']}, eval_loss: {eval_loss.item():.5e}\n")
            epoch_avg_val_losses.append(eval_loss.item())
            if accelerator.is_main_process:
                plot_loss(epoch_avg_val_losses, config.eval.eval_folder, cut=True)
            # Periodically save big checkpoint & sample
            if accelerator.is_main_process and (state['update_step'] != 0 and state['update_step'] % config.training.snapshot_freq == 0) and accelerator.sync_gradients:
                save_step = state['update_step'] // config.training.snapshot_freq
                ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth')
                save_checkpoint(ckpt_path, state, accelerator)
                logging.info(f"checkpoint_{save_step} saved.\n")
                if config.training.snapshot_sampling:
                    this_sample_dir = os.path.join(sample_dir, f"_{save_step}")
                    os.makedirs(this_sample_dir, exist_ok=True)
                    
                    eval_degrade_batch_gain = eval_degrade_batch[:, 0:1, :, :]
                    eval_degrade_batch_sin = eval_degrade_batch[:, 1:2, :, :]
                    # 如果是inpainting， 把eval_degrade_batch可视化一下
                    if measure_config['operator'] ['name'] == 'inpainting':
                        if eval_degrade_batch_gain.shape[1] == 1:                         # 单通道 → 复制成 RGB
                            eval_degrade_batch_gain_rgb = eval_degrade_batch_gain.repeat(1, 3, 1, 1)          # [B, 3, H, W]
                            eval_degrade_batch_sin_rgb = eval_degrade_batch_sin.repeat(1, 3, 1, 1)          # [B, 3, H, W]
                        elif eval_degrade_batch_gain.shape[1] == 3:                       # 已经是 RGB
                            eval_degrade_batch_gain_rgb = eval_degrade_batch_gain.clone()                     # 拷贝一份
                            eval_degrade_batch_sin_rgb = eval_degrade_batch_sin.clone()                     # 拷贝一份
                        red = torch.tensor([1.0, 0.0, 0.0],            # shape [3]
                                device=eval_degrade_batch.device,
                                dtype=eval_degrade_batch.dtype)[None, :, None, None]   # [1,3,1,1]
                        mask3 = (mask[:, 0:1, :, :]).expand(-1, 3, -1, -1)            # [B, 3, H, W]
                        eval_degrade_batch_gain = torch.where(mask3 == 0, red, eval_degrade_batch_gain_rgb)
                        eval_degrade_batch_sin = torch.where(mask3 == 0, red, eval_degrade_batch_sin_rgb)
                    grid_degrade_batch_gain = get_grid_img(eval_degrade_batch_gain)
                    grid_degrade_batch_sin = get_grid_img(eval_degrade_batch_sin)
                    grid_degrade_batch_gain = np.clip(grid_degrade_batch_gain, 0.0, 1.0)
                    grid_degrade_batch_sin = np.clip(grid_degrade_batch_sin, 0.0, 1.0)
                    
                    eval_batch_gain = eval_batch[:, 0:1, :, :]
                    eval_batch_sin = eval_batch[:, 1:2, :, :]
                    grid_batch_gain = get_grid_img(eval_batch_gain)
                    grid_batch_sin = get_grid_img(eval_batch_sin)
                    grid_batch_gain = np.clip(grid_batch_gain, 0.0, 1.0)
                    grid_batch_sin = np.clip(grid_batch_sin, 0.0, 1.0)

                    pred_gain = pred[:, 0:1, :, :]
                    pred_sin = pred[:, 1:2, :, :]
                    grid_pred_gain = get_grid_img(pred_gain)
                    grid_pred_sin = get_grid_img(pred_sin)
                    grid_pred_gain = np.clip(grid_pred_gain, 0.0, 1.0)
                    grid_pred_sin = np.clip(grid_pred_sin, 0.0, 1.0)
                    
                    if grid_degrade_batch_gain.shape[2] == 1:
                        plt.imsave(os.path.join(this_sample_dir, 'input_gain.png'), grid_degrade_batch_gain.squeeze(2), cmap='gray', vmin=0, vmax=1)
                        plt.imsave(os.path.join(this_sample_dir, 'input_sin.png'), grid_degrade_batch_sin.squeeze(2), cmap='gray', vmin=0, vmax=1)
                    else:
                        plt.imsave(os.path.join(this_sample_dir, 'input_gain.png'), grid_degrade_batch_gain, vmin=0, vmax=1)
                        plt.imsave(os.path.join(this_sample_dir, 'input_sin.png'), grid_degrade_batch_sin, vmin=0, vmax=1)
                    if grid_pred_gain.shape[2] == 1:
                        plt.imsave(os.path.join(this_sample_dir, 'label_gain.png'), grid_batch_gain.squeeze(2), cmap='gray', vmin=0, vmax=1)
                        plt.imsave(os.path.join(this_sample_dir, 'recon_gain.png'), grid_pred_gain.squeeze(2), cmap='gray', vmin=0, vmax=1)
                        plt.imsave(os.path.join(this_sample_dir, 'label_sin.png'), grid_batch_sin.squeeze(2), cmap='gray', vmin=0, vmax=1)
                        plt.imsave(os.path.join(this_sample_dir, 'recon_sin.png'), grid_pred_sin.squeeze(2), cmap='gray', vmin=0, vmax=1)
                    else:
                        plt.imsave(os.path.join(this_sample_dir, 'label_gain.png'), grid_batch_gain, vmin=0, vmax=1)
                        plt.imsave(os.path.join(this_sample_dir, 'recon_gain.png'), grid_pred_gain, vmin=0, vmax=1)
                        plt.imsave(os.path.join(this_sample_dir, 'label_sin.png'), grid_batch_sin, vmin=0, vmax=1)
                        plt.imsave(os.path.join(this_sample_dir, 'recon_sin.png'), grid_pred_sin, vmin=0, vmax=1)
                    logging.info(f"samples_{save_step} saved at {this_sample_dir}.\n")
        accelerator.wait_for_everyone()

        # # log the npu and cpu
        # if accelerator.is_main_process and state['update_step'] % config.monitoring.freq==0:
        #     log_resource_util(state['update_step'])
        # accelerator.wait_for_everyone()

    logging.info(f"Training finished. Final iter step: {state['iter_step']}, update step: {state['update_step']}.\n")