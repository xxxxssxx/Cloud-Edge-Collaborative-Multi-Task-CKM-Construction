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

# Lint as: python3
"""Training NCSN++ on CKM_gain_AoA_128."""

from configs.default_CKM_gain_AoA_128_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.fourier_scale = 16
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8  # 本地测试先改成2，正式训练改8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = False  # 端到端网络时都选False
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.embedding_type = 'fourier'
  model.init_scale = 0.0
  model.conv_size = 3

  return config
