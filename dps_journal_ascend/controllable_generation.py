from models import utils as mutils
import torch
import numpy as np
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
from abc import ABC, abstractmethod
import torch
import sde_lib
import matplotlib.pyplot as plt
from utils import get_grid_img, plot_loss, img2gif
import logging
import os

# TODO: pc dps sampler

def get_pc_dpsampler(sde,
                  predictor, corrector,
                  measurement_cond_fn,
                  inverse_scaler,
                  snr=0.1,
                  n_steps=1,
                  probability_flow=False,
                  continuous=True,
                  record=False,
                  sample_dir=None,
                  eps=1e-5):
  '''
  combine pc sampler and dpsampler. corrector_update_fn的内部是有循环的，用n_steps参数来控制做几个corrector step。
  '''
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)
  
  def pc_dpsampler(model, ref_img, y_n):  # 是否应该接收mask作为参数？不应该，因为mask已经固定到measure_fn里了，与operator绑定。
    if record:
      progress_dir_gain = os.path.join(sample_dir, "progress_gain")
      progress_dir_sin = os.path.join(sample_dir, "progress_sin")
      progress_dir_building = os.path.join(sample_dir, "progress_building")
      os.makedirs(progress_dir_gain, exist_ok=True)
      os.makedirs(progress_dir_sin, exist_ok=True)
      os.makedirs(progress_dir_building, exist_ok=True)
    
    x = sde.prior_sampling(ref_img.shape).to(ref_img.device)  # \hat{x}_{N-1} \sim prior  Initial sample. type of prior and value of noise scheduel is in sde object.
    # ref_img 仅提供形状，不参与计算
    times = torch.linspace(sde.T, eps, sde.N)  # [T==1, ..., eps==1e-8]  # 共N个元素  times是递减的
    distances = []
    for i in range(sde.N):
      
      # x_{i-1} <- x_{i}

      x_prev = x.clone().detach().requires_grad_(True)  # save the x_{i} for measurement_cond_fn
      t = times[i]  # 随着i从0到N-1，t从T到eps
      # timestep = (t * (sde.N - 1) / sde.T).long()  # 随着i从0到N-1，timestep从N-1到0 这是predictor的ancestral sampling里的实现
      timestep = sde.N-1-i  # 这种实现更稳健
      vec_t = torch.ones(ref_img.shape[0], device=ref_img.device) * t
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)

      # progressive estimate and pass the measurement y_n into perturbation kernel
      '''x_0_hat is a function of x_prev, score is a function of x_prev'''
      if isinstance(sde, sde_lib.VESDE):
          score = score_fn(x_prev, vec_t)
          x_0_hat = x_prev + score * (sde.discrete_sigmas[timestep])**2  # 这一步数值不稳定。早期sigma较大时，score的很小误差会导致x_0_hat的很大误差
          mean_y_n_n, std_y_n_n = sde.marginal_prob(y_n, vec_t)  # perturbation kernel
          y_n_n = mean_y_n_n +  torch.randn_like(mean_y_n_n) * std_y_n_n[:, None, None, None]
          sigma = sde.discrete_sigmas.to(t.device)[timestep]
          timestep_vec = torch.full((ref_img.shape[0],), timestep, dtype=torch.long, device=torch.device('cpu'))  # (batch_size,)
          adjacent_sigma = torch.where(timestep_vec == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep_vec - 1])
          sigma_4_dsg = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))  # sigma in reverse transition kernel
      elif isinstance(sde, sde_lib.VPSDE):
          score = score_fn(x_prev, vec_t)
          x_0_hat = 1/sde.sqrt_alphas_cumprod[timestep] * (x_prev + (1 - sde.alphas_cumprod[timestep]) * score )
          mean_y_n_n, std_y_n_n = sde.marginal_prob(y_n, vec_t)  # perturbation kernel
          y_n_n = mean_y_n_n +  torch.randn_like(mean_y_n_n) * std_y_n_n[:, None, None, None]
          sigma_4_dsg = torch.sqrt((1-sde.alphas_cumprod_prev[timestep]) * (1-sde.alphas[timestep]) / (1-sde.alphas_cumprod[timestep]))  # sigma in reverse transition kernel
      else:
        raise NotImplementedError

      # x_apo: x_{i-1}^{'} 
      x_apo, x_mean = predictor_update_fn(x_prev, vec_t, model=model)  # 内部把时间转换成索引了
      x_apo, x_mean = corrector_update_fn(x_apo, vec_t, model=model)

      x, distance = measurement_cond_fn(
        x_prev=x_prev,  # x_{i}
        # DSG的时候用x_mean, DPS的时候用x_apo
        x_t=x_apo,  # x_{i-1}^{'}, output of reverse transition kernel(output of one step denoise), need to be corrected by forward model
        x_0_hat=x_0_hat,  # \hat{x}_0, progressive estimate based on x_{i}
        measurement=y_n,  # y_n = A(x_0) + n
        noisy_measurement=y_n_n,  # pass the measurement y_n into perturbation kernel，not need by ps, but need by mcg
        sigma_4_dsg = sigma_4_dsg,  # need by dsg
        )  
        # distance: || y_n - A(\hat{x}_0) ||_{2}^{2}

      x = x.detach()
      distances.append(distance.item())

      # 可视化
      if record and i % 50 == 0:
        x_gain = x[:, 0:1, :, :]
        x_sin = x[:, 1:2, :, :]
        x_building = x[:, 2:3, :, :]
        grid_x_gain = get_grid_img(x_gain)
        grid_x_sin = get_grid_img(x_sin)
        grid_x_building = get_grid_img(x_building)
        file_path_gain = os.path.join(progress_dir_gain, f"x_{str(i).zfill(5)}.png")
        file_path_sin = os.path.join(progress_dir_sin, f"x_{str(i).zfill(5)}.png")
        file_path_building = os.path.join(progress_dir_building, f"x_{str(i).zfill(5)}.png")
        plt.imsave(file_path_gain, grid_x_gain.squeeze(2), cmap='gray')
        plt.imsave(file_path_sin, grid_x_sin.squeeze(2), cmap='gray')
        plt.imsave(file_path_building, grid_x_building.squeeze(2), cmap='gray')
        plot_loss(distances, sample_dir, cut=False)
    if record:
      img2gif(progress_dir_gain)
      img2gif(progress_dir_sin)
      img2gif(progress_dir_building)

    return inverse_scaler(x), distances
  # compute graph: x_prev → x_0_hat → norm, finally norm w.r.t. x_prev. x_apo does not require grad, it only gets a minus operation.

  return pc_dpsampler


      # pbar.set_postfix({'distance': distance.item()}, refresh=False)
      # if record:
      #     if idx % 10 == 0:
      #         file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
      #         plt.imsave(file_path, clear_color(img))

      # return img     