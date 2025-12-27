import ml_collections
import torch
from accelerate import Accelerator

def get_default_configs():
  config = ml_collections.ConfigDict()
  # training  run_lib.py
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 8   # default 64
  training.num_iter_step = 60000  # 总step数 for iter step
  training.snapshot_freq = 250  # 每隔 snapshot_freq 个 step 保存一次checkpoint并且生成batch_size个样本 for update step
  training.log_freq = 25  # 每隔 log_freq 个 step 打印一次train loss日志 for update step
  training.eval_freq = 250  # 每隔 eval_freq 个step 评估一次模型，并打印eval loss日志并plot一次eval_loss曲线 for update step
  ## store additional checkpoints for preemption in cloud computing environments  run_lib.py
  training.snapshot_freq_for_preemption = 250 # 每隔 snapshot_freq_for_preemption 个 step 保存一次 checkpoint_meta for update step
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.grad_accum_steps = 4  # 梯度累积

  # settings for loss computation  losses.py
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling.py
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1  # correcter的步数
  sampling.noise_removal = True
  sampling.probability_flow = False  # 是否用ODE
  sampling.snr = 0.075  # 目前不太懂snr

  # evaluation
  config.eval = eval = ml_collections.ConfigDict()
  eval.begin_ckpt = 50
  eval.end_ckpt = 96
  eval.batch_size = 1         # default 512 最好是平方数 方便可视化
  eval.enable_sampling = True
  eval.num_samples = 50000
  eval.enable_loss = True
  eval.enable_bpd = False
  eval.bpd_dataset = 'test'
  eval.eval_folder = './eval/'

  # datasets.py
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CKM_gain_AoA_128'  # 数据集名称  'MNIST' 'CIFAR10' 'CKM_gain_128' 'CKM_AoA_128' 'CKM_gain_AoA_128'
  data.data_dir = "/opt/dpcvol/models/xsx/data/"
  data.image_size = 128       # 测试先用32
  data.random_flip = True  # 是否要随机水平翻转  对gain数据集，应该翻转 对AoA数据不翻转 sin数据可以翻转
  data.uniform_dequantization = False
  data.centered = True  # 是否要用scalar做中心化。感觉用ncsnpp时，有点bug。因为如果这里取false, 在ncsnpp时会一个中心化，但是ncsnpp里没有反中心化的操作。
  data.num_channels = 3  # 1 for grayscale, 3 for gain+AoA+building 

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 2000  # 离散化sde的步数
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-5
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  # device
  config.seed = 42
  config.device = torch.device('cpu')

  # monitoring
  config.monitoring = monitoring = ml_collections.ConfigDict()
  monitoring.freq = 10  # for update step

  # dp_sampling
  config.dp_sampling = dp_sampling = ml_collections.ConfigDict()
  # specify the task file: inpainting_config, super_resolution_config, joint_degradation_recovery_config
  dp_sampling.task_config = "./configs/dps_configs/joint_degradation_recovery_config.yaml" 
  dp_sampling.checkpoint_num = 100
  dp_sampling.record = True  # 是否记录采样过程的图像  test的时候改成False

  return config
