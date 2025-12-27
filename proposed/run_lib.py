# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
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
from utils import save_checkpoint, restore_checkpoint, plot_loss, colorize_and_mark, generate_downsample_mask, generate_block_mask, load_yaml, get_grid_img, setup_logging, BS_building_detection
import matplotlib.pyplot as plt
import controllable_generation
from sampling import LangevinCorrector, ReverseDiffusionPredictor
import yaml                                             ##!!!!!!!
from dps.measurements import get_noise, get_operator
from dps.condition_methods import get_conditioning_method
from dps.dps_utils.img_utils import clear_color, mask_generator
from functools import partial
from accelerate import Accelerator
import sys
import torch.distributed as dist
import os

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
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    # state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch_avg_val_losses=[])

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Build data iterators from our torch-based dataset loaders
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization)  # 这里已经应用了transform
    
    # prepare components for distributed training
    score_model, optimizer, train_ds, eval_ds = accelerator.prepare(score_model, optimizer, train_ds, eval_ds)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

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

        accelerator.unwrap_model(score_model).load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ema.load_state_dict(checkpoint['ema'])
        initial_iter_step = checkpoint['iter_step']
        initial_update_step = checkpoint['update_step']
        epoch_avg_val_losses = checkpoint["epoch_avg_val_losses"]

    state = dict(optimizer=optimizer, model=score_model, ema=ema, iter_step=initial_iter_step, update_step=initial_update_step, epoch_avg_val_losses=epoch_avg_val_losses)

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDE
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    # Build sampling function
    if config.training.snapshot_sampling:
        sampling_shape = (config.eval.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)


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
        
        # AoA转换为sin
        p_min = 1/19.0  # 20 / (180-(-200))
        AoA_batch = batch[:, 1:2, :, :].to(torch.float32)  # (B,1,H,W)
        building_batch = batch[:, 2:3, :, :].to(torch.float32)  # (B, 1, H, W)
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

        # scale the data
        batch = scaler(batch)  # 这里的scaler是一个函数，将数据缩放到[-1,1]之间
        # one training step
        state['iter_step'] = iter_step + 1
        loss = train_step_fn(state, batch, accelerator)

        # log the  training loss
        if state['update_step'] % config.training.log_freq == 0 and accelerator.sync_gradients:
            logging.info(f"step: {state['update_step']}, training_loss: {loss.item():.5e}\n")

        # Save a checkpoint to resume later
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
            # AoA转换为sin
            p_min = 1/19.0  # 20 / (180-(-200))
            eval_AoA_batch = eval_batch[:, 1:2, :, :].to(torch.float32)  # (B,1,H,W)
            eval_building_batch = eval_batch[:, 2:3, :, :].to(torch.float32)  # (B, 1, H, W)
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
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch, accelerator)  # 有修改
            logging.info(f"step: {state['update_step']}, eval_loss: {eval_loss.item():.5e}\n")
            epoch_avg_val_losses.append(eval_loss.item())
            if accelerator.is_main_process:
                plot_loss(epoch_avg_val_losses, config.eval.eval_folder, cut=True)
        accelerator.wait_for_everyone()

        # Periodically save big checkpoint & sample
        if accelerator.is_main_process and (state['update_step'] != 0 and state['update_step'] % config.training.snapshot_freq == 0) and accelerator.sync_gradients:
            save_step = state['update_step'] // config.training.snapshot_freq
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth')
            save_checkpoint(ckpt_path, state, accelerator)
            logging.info(f"checkpoint_{save_step} saved.\n")

            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                ema.restore(score_model.parameters())

                this_sample_dir = os.path.join(sample_dir, f"_{save_step}")
                
                os.makedirs(this_sample_dir, exist_ok=True)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)

                # Save gain
                gain_grid = image_grid[0:1, :, :]
                gain_png_path = os.path.join(this_sample_dir, "sample_gain.png")
                save_image(gain_grid, gain_png_path)
                
                # Save AoA
                AoA_grid = image_grid[1:2, :, :]
                aoa_png_path = os.path.join(this_sample_dir, "sample_AoA.png")
                save_image(AoA_grid, aoa_png_path)
                
                # Save building
                building_grid = image_grid[2:3, :, :]
                building_png_path = os.path.join(this_sample_dir, "sample_building.png")
                save_image(building_grid, building_png_path)
                logging.info(f"samples_{save_step} saved at {this_sample_dir}.\n")
        accelerator.wait_for_everyone()

        # # log the npu and cpu
        # if accelerator.is_main_process and state['update_step'] % config.monitoring.freq==0:
        #     log_resource_util(state['update_step'])
        # accelerator.wait_for_everyone()

    logging.info(f"Training finished. Final iter step: {state['iter_step']}, update step: {state['update_step']}.\n")
    

def dp_sampling(config, workdir):

    accelerator = Accelerator()
    config.device = accelerator.device
    rank = accelerator.process_index
    setup_logging(rank, log_path="logdps.log")

    logging.info("Starting DP sampling.\n")

    # Setup SDE
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)

    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                decay=config.model.ema_rate)
    
    ckpt_num = config.dp_sampling.checkpoint_num
    ckpt_filename = workdir + f"checkpoints/checkpoint_{ckpt_num}.pth"
    checkpoint = torch.load(ckpt_filename, map_location=config.device)

    accelerator.unwrap_model(score_model).load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    ema.load_state_dict(checkpoint['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())

    # Device setting
    logging.info(f"device: {config.device}\n")

    # configs about the forward operator
    task_config = load_yaml(config.dp_sampling.task_config)
    
    score_model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=config.device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logging.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}\n")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])  # ps, mcg...
    measurement_cond_fn = cond_method.conditioning

    sample_dir = os.path.join(workdir, config.eval.eval_folder, "multi_task/", measure_config['operator']['name'], f"_{rank}")
    os.makedirs(sample_dir, exist_ok=True)
    logging.info(f"Sample directory created: {sample_dir}")

    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)
    eval_iter = iter(eval_ds)

    predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
    corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
    snr = 0.16 #@param {"type": "number"}
    n_steps = 1 #@param {"type": "integer"}
    probability_flow = False #@param {"type": "boolean"}

    batch = next(eval_iter)  
    ref_img = batch['image'].to(config.device)  # torch.tensor

    building_batch = ref_img[:, 2:3, :, :].to(torch.float32)  # (B, 1, H, W)
    
    # AoA转换为sin
    p_min = 1/19.0  # 20 / (180-(-200))
    AoA_batch = ref_img[:, 1:2, :, :].to(torch.float32)  # (B,1,H,W)
    angle_batch = torch.zeros_like(AoA_batch, dtype=torch.float32)
    angle_batch[building_batch==0] = ((AoA_batch[building_batch==0] - p_min) / (1.0 - p_min) * 360.0 - 180.0)
    rad_batch = torch.deg2rad(angle_batch)
    sin_batch = torch.zeros_like(AoA_batch, dtype=torch.float32)
    sin_batch[building_batch==0] = torch.sin(rad_batch[building_batch==0])
    
    # sin_batch中，将非建筑处的sin值从-1~1映射至0.3~1，建筑处的值为0
    # pixel [0.3, 1], sin [-1, 1]   pixel = 0.35*sin + 0.65  sin = (1/0.35)*pixel - 0.65/0.35
    sin_batch[building_batch==0] = 0.35 * sin_batch[building_batch==0] + 0.65
    sin_batch[building_batch==1] = 0 
    
    ref_img[:, 1:2, :, :] = sin_batch

    ref_img = scaler(ref_img)  # 这里的scaler是一个函数，将数据缩放到[-1,1]之间

    # 如果inpainting， 对measurement_cond_fn重新包装，传入mask
    if measure_config['operator'] ['name'] == 'inpainting':
        mask_gen = mask_generator(
            **measure_config['mask_opt']
        )
        mask, success = mask_gen(ref_img[:, 0:1, :, :])  # [B, 1, H, W]
        mask = mask.repeat(1, 3, 1, 1)
        mask[:, 2:3, :, :] = 1  # 把 mask 在 building batch 的维度置 1
        if not success:
            raise ValueError("Mask generation failed.")
        measurement_cond_fn = partial(cond_method.conditioning, mask=mask)  # mask通过**kargs传入operator的forward函数

    # 如果ray tracing，对measurement_cond_fn重新包装，传入BS_mask和building_mask
    if measure_config['operator'] ['name'] == 'ray_tracing':
        BS_region_size = measure_config['opt']['BS_region_size']
        if BS_region_size % 2 == 0:
            raise ValueError("BS_region_size 必须为奇数，便于定义唯一中心。")
        BS_mask, building_mask = BS_building_detection(ref_img, BS_region_size)  # [B, C, H, W]
        measurement_cond_fn = partial(cond_method.conditioning, BS_mask=BS_mask, building_mask=building_mask)  # BS_mask和building_mask通过**kargs传入operator的forward函数


    pc_dpsampler = controllable_generation.get_pc_dpsampler(sde,
                                                    predictor, corrector,
                                                    measurement_cond_fn,
                                                    inverse_scaler,
                                                    snr=snr,
                                                    n_steps=n_steps,
                                                    probability_flow=probability_flow,
                                                    continuous=config.training.continuous,
                                                    record=config.dp_sampling.record,
                                                    sample_dir=sample_dir)

    # Exception) In case of inpainting or ray tracing
    if measure_config['operator'] ['name'] == 'inpainting':
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img, mask=mask)
        y_n = noiser(y)
    elif measure_config['operator'] ['name'] == 'ray_tracing':
        y = operator.forward(ref_img, BS_mask=BS_mask, building_mask=building_mask)
        y_n = y  # 该任务不加噪声
    else: 
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)


    img, distances = pc_dpsampler(score_model, ref_img, y_n)  # mask在内部产生，不要传入。返回的img已经inverse_scaler过，是在0~1的
    
    # 如果是inpainting， 把y_n可视化一下
    y_n = inverse_scaler(y_n)  # 反归一化
    y_n_gain = y_n[:, 0:1, :, :]
    y_n_sin = y_n[:, 1:2, :, :]
    if measure_config['operator']['name'] == 'inpainting':
        red = torch.tensor([1.0, 0.0, 0.0],            # shape [3]
                   device=y_n.device,
                   dtype=y_n.dtype)[None, :, None, None]   # [1,3,1,1]
        mask3 = mask.expand(-1, 3, -1, -1)            # [B, 3, H, W]
        y_n_gain_rgb = y_n_gain.repeat(1, 3, 1, 1)
        y_n_sin_rgb = y_n_sin.repeat(1, 3, 1, 1)
        y_n_gain = torch.where(mask3 == 0, red, y_n_gain_rgb)
        y_n_sin = torch.where(mask3 == 0, red, y_n_sin_rgb)
    grid_y_n_gain = get_grid_img(y_n_gain)
    grid_y_n_sin = get_grid_img(y_n_sin)
    grid_y_n_gain = np.clip(grid_y_n_gain, 0.0, 1.0)
    grid_y_n_sin = np.clip(grid_y_n_sin, 0.0, 1.0)
    
    ref_img = inverse_scaler(ref_img)  # 反归一化
    ref_img_gain = ref_img[:, 0:1, :, :]
    ref_img_sin = ref_img[:, 1:2, :, :]
    ref_img_building = ref_img[:, 2:3, :, :]
    grid_ref_img_gain = get_grid_img(ref_img_gain)
    grid_ref_img_sin = get_grid_img(ref_img_sin)
    grid_ref_img_building = get_grid_img(ref_img_building)
    grid_ref_img_gain = np.clip(grid_ref_img_gain, 0, 1)
    grid_ref_img_sin = np.clip(grid_ref_img_sin, 0, 1)
    grid_ref_img_building = np.clip(grid_ref_img_building, 0, 1)
    
    img_gain = img[:, 0:1, :, :]
    img_sin = img[:, 1:2, :, :]
    img_building = img[:, 2:3, :, :]
    grid_img_gain = get_grid_img(img_gain)
    grid_img_sin = get_grid_img(img_sin)
    grid_img_building = get_grid_img(img_building)
    grid_img_gain = np.clip(grid_img_gain, 0, 1)
    grid_img_sin = np.clip(grid_img_sin, 0, 1)
    grid_img_building = np.clip(grid_img_building, 0, 1)

    # 这个也要改，分别判断
    if grid_y_n_gain.shape[2] == 1:
        plt.imsave(os.path.join(sample_dir, 'input_gain.png'), grid_y_n_gain.squeeze(2), cmap='gray', vmin=0, vmax=1)
        plt.imsave(os.path.join(sample_dir, 'input_sin.png'), grid_y_n_sin.squeeze(2), cmap='gray', vmin=0, vmax=1)
    else:
        plt.imsave(os.path.join(sample_dir, 'input_gain.png'), grid_y_n_gain, vmin=0, vmax=1)
        plt.imsave(os.path.join(sample_dir, 'input_sin.png'), grid_y_n_sin, vmin=0, vmax=1)

    plt.imsave(os.path.join(sample_dir, 'label_gain.png'), grid_ref_img_gain.squeeze(2), cmap='gray', vmin=0, vmax=1)
    plt.imsave(os.path.join(sample_dir, 'label_sin.png'), grid_ref_img_sin.squeeze(2), cmap='gray', vmin=0, vmax=1)
    plt.imsave(os.path.join(sample_dir, 'label_building.png'), grid_ref_img_building.squeeze(2), cmap='gray', vmin=0, vmax=1)
    
    plt.imsave(os.path.join(sample_dir, 'recon_gain.png'), grid_img_gain.squeeze(2), cmap='gray', vmin=0, vmax=1)
    plt.imsave(os.path.join(sample_dir, 'recon_sin.png'), grid_img_sin.squeeze(2), cmap='gray', vmin=0, vmax=1)
    plt.imsave(os.path.join(sample_dir, 'recon_building.png'), grid_img_building.squeeze(2), cmap='gray', vmin=0, vmax=1)


    logging.info(f"DP sampling finished.\n")