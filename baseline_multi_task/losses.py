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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
import accelerate


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn

def get_loss_fn(train):
  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    degrade_image = batch["degrade_image"]
    ground_truth = batch["ground_truth"]
    pred = model_fn(degrade_image)
    losses = torch.square(pred - ground_truth)
    loss = torch.mean(losses)
    return loss, pred
  return loss_fn

def get_step_fn(train, optimize_fn=None):
  loss_fn = get_loss_fn(train)

  def step_fn(state, batch, accelerator):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    optimizer = state['optimizer']
    ema = state['ema']
    assert accelerator is not None, "accelerator must be passed in to step_fn"
    if train:
      with accelerator.accumulate(model):
        loss, pred = loss_fn(model, batch)
        loss = loss / accelerator.gradient_accumulation_steps
        accelerator.backward(loss)
        if accelerator.sync_gradients:
          optimize_fn(optimizer, model.parameters(), step=state['update_step'])
          optimizer.zero_grad(set_to_none=True)
          ema.update(model.parameters())
          state['update_step'] += 1
    else:
      with torch.no_grad():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss, pred = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss, pred

  return step_fn
