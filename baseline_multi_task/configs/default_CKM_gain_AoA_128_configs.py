import ml_collections
import torch
from accelerate import Accelerator

def get_default_configs():
  config = ml_collections.ConfigDict()
  # training  run_lib.py
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 8   # default 64
  training.num_iter_step = 10000  # 总step数 for iter step
  training.snapshot_freq = 250  # 每隔 snapshot_freq 个 step 保存一次checkpoint并且生成batch_size个样本 for update step
  training.log_freq = 250  # 每隔 log_freq 个 step 打印一次train loss日志 for update step
  training.eval_freq = 250  # 每隔 eval_freq 个step 评估一次模型，并打印eval loss日志并plot一次eval_loss曲线 for update step
  ## store additional checkpoints for preemption in cloud computing environments  run_lib.py
  training.snapshot_freq_for_preemption = 250 # 每隔 snapshot_freq_for_preemption 个 step 保存一次 checkpoint_meta for update step
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.grad_accum_steps = 4  # 梯度累积

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
  data.dataset = 'CKM_gain_AoA_128'  # 数据集名称  'MNIST' 'CIFAR10' 'CKM_gain_128' 'CKM_gain_AoA_128'
  data.data_dir = "/opt/dpcvol/models/xsx/data/"
  data.image_size = 128       # 测试先用32
  data.random_flip = True  # 是否要随机水平翻转  对gain数据集，应该翻转 对于AoA，不应该翻转。对于AoA-sine，应该翻转
  data.uniform_dequantization = False
  data.centered = True  # 是否要用scalar做中心化。感觉用ncsnpp时，有点bug。因为如果这里取false, 在ncsnpp时会一个中心化，但是ncsnpp里没有反中心化的操作。

  # model
  config.model = model = ml_collections.ConfigDict()
  model.embedding_type = 'fourier'
  model.dropout = 0.
  model.num_channels_input = 2  # 增益和角度
  model.num_channels_output = 2
  model.input_size = config.data.image_size

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  # device
  config.seed = 42
  config.device = torch.device('cpu')

  # test
  config.test = test = ml_collections.ConfigDict()
  test.checkpoint_num = 2

  return config
