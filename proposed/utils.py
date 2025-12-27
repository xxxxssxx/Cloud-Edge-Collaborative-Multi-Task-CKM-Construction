import torch
import os
import logging
import matplotlib.pyplot as plt
import yaml
import numpy as np
import sys
import math
import re
from PIL import Image


def restore_checkpoint(ckpt_dir, state, device):
    """
    Restores state (model weights, optimizer states, EMA, step) from a checkpoint file.
    ckpt_dir: str, path to the .pth file
    state: dict with keys ['optimizer', 'model', 'ema', 'step']
    device: torch device
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returning the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        state['epoch_avg_val_losses'] = loaded_state["epoch_avg_val_losses"]
        return state

def save_checkpoint(ckpt_dir, state, accelerator):
    """
    Saves state (model weights, optimizer states, EMA, step) to a checkpoint file.
    ckpt_dir: str, path to the .pth file
    state: dict with keys ['optimizer', 'model', 'ema', 'step']
    """
    wrapped_model = state['model']
    raw_model = accelerator.unwrap_model(wrapped_model)
    
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': raw_model.state_dict(),
        'ema': state['ema'].state_dict(),
        'iter_step' : state['iter_step'],
        'update_step' : state['update_step'],
        'epoch_avg_val_losses': state['epoch_avg_val_losses']
    }
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    torch.save(saved_state, ckpt_dir, _use_new_zipfile_serialization = True)

def plot_loss(epoch_avg_val_losses, save_folder, cut=False):
    """
    Plots the validation loss over epochs and saves the figure.

    Args:
        epoch_avg_val_losses (list): A list of validation loss values per epoch.
        save_folder (str): Path to the folder where the plot will be saved.
    """
    N = len(epoch_avg_val_losses)
    if N > 20 and cut:
        # 取最后一部分的数据
        tail_len = math.ceil(N * 0.25)
        plot_data = epoch_avg_val_losses[-tail_len:]
        x_axis = range(N - tail_len, N)
    else:
        plot_data = epoch_avg_val_losses
        x_axis = range(N)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, plot_data, marker='o', linestyle='-', color='b', label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Over Steps")
    plt.legend()
    plt.grid(True)

    # 如果目标文件夹不存在则创建文件夹，并保存图像
    os.makedirs(save_folder, exist_ok=True)
    loss_plot_path = os.path.join(save_folder, "validation_loss_plot.png")
    plt.savefig(loss_plot_path)
    logging.info(f"Validation loss plot saved at: {loss_plot_path}")
    plt.close()

def colorize_and_mark(img, img_masked, img_inpainted, mask):
    """
    参数:
        img: 原始图像，形状 (batch, 1, H, W)
        img_masked: 掩盖后的图像，形状 (batch, 1, H, W)
        img_inpainted: 修复后的图像，形状 (batch, 1, H, W)
        mask: 掩码，形状 (batch, 1, H, W)，其中 0 表示被掩盖区域 1 表示有效区域
    返回:
        img_color, img_masked_color, img_inpainted_color:
            全部扩展为3通道的彩色图像 其中 img_masked_color 的被掩盖区域(mask==0)全替换为红色
    """
    # 将灰度图复制到3个通道，便于后续上色显示
    img_color = img.repeat(1, 3, 1, 1)
    img_masked_color = img_masked.repeat(1, 3, 1, 1)
    img_inpainted_color = img_inpainted.repeat(1, 3, 1, 1)

    # mask: 形状 (batch, 1, H, W) ，先 squeeze 掉通道维度，得到 (batch, H, W)
    masked_bool = (mask == 0).squeeze(1)  # 被遮盖位置为 True

    # 对每个样本，在被掩盖的位置上全部标记为红色，即 R=1, G=0, B=0
    for i in range(img_masked_color.shape[0]):
        # 使用 boolean 索引直接修改对应像素
        img_masked_color[i, 0, masked_bool[i]] = 0.5  # red channel
        img_masked_color[i, 1, masked_bool[i]] = 0.2  # green channel
        img_masked_color[i, 2, masked_bool[i]] = 0.2  # blue channel

    return img_color, img_masked_color, img_inpainted_color

def generate_downsample_mask(img, downsample_rate):
    """
    根据输入图像和下采样率生成稀疏采样掩码。
    
    参数:
        img: torch.Tensor 形状为 (batch_size, channels, height, width)
        downsample_rate: int 下采样率 例如 4 或 8 表示每隔 downsample_rate 个像素保留一个像素
    返回:
        mask: torch.Tensor 与 img 尺寸相同，被保留的像素位置为 1 其余为 0
    """
    # 创建与 img 相同尺寸的全 0 掩码
    mask = torch.zeros_like(img)
    # 利用 ellipsis 对最后两个维度进行切片，每隔 downsample_rate 个像素置为 1
    mask[..., ::downsample_rate, ::downsample_rate] = 1.0
    return mask

def generate_block_mask(img, block_size):
    mask_size = block_size  # 方形掩码的大小
    # 创建和图像尺寸相同的全1掩码
    mask = torch.ones_like(img)
    # 对 batch 中的每一张图像单独生成一个随机方形掩码
    batch_size, _, height, width = img.shape
    for i in range(batch_size):
        # 随机选择方形掩码左上角的位置，确保掩码不会超出图像边界
        top = torch.randint(0, height - mask_size + 1, (1,)).item()
        left = torch.randint(0, width - mask_size + 1, (1,)).item()
        # 将对应区域置 0，即为掩码区域
        mask[i, :, top:top+mask_size, left:left+mask_size] = 0.

    return mask

def load_yaml(file_path: str) -> dict:                                             ##!!!!!!!
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def image_grid(img):
    size = img.shape[1]
    channels = img.shape[-1]
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
    return img

# 生成单个 grid 图像的函数，传入 torch tensor，内部自动完成 permute 与 numpy 转换
def get_grid_img(x_tensor):
    # 转换成 numpy 数组，并转换排列成 (batch, h, w, channels)
    x_np = x_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    return image_grid(x_np)

def setup_logging(rank: int, log_path: str = "trainlog.log"):
    root = logging.getLogger()
    # 1) clear old handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    # 2) formatter with rank tag
    fmt = logging.Formatter(
        "%(asctime)s [Rank {:>2}] %(levelname)s: %(message)s".format(rank)
    )

    # 3) console handler (optional)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 4) single shared file handler
    file_h = logging.FileHandler(log_path, mode="a")
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    root.setLevel(logging.INFO)

def img2gif(img_dir):
    files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    files.sort(key=lambda x: int(re.findall(r'x_(\d+)\.png', x)[0]))

    imgs = []
    for f in files:
        img = Image.open(os.path.join(img_dir, f))
        imgs.append(img)

    # 保存为GIF
    output_path = os.path.join(img_dir, 'progress.gif')
    imgs[0].save(
        output_path,
        save_all=True,
        append_images=imgs[1:],
        duration=100,  # 每帧100ms
        loop=0
    )

def single_BS_mask(img_1chw: torch.Tensor, BS_region_size: int) -> torch.BoolTensor:
    """
    img_1chw : (1, H, W)      单通道单张图像
    return    : (1, H, W)      对应掩码
    """
    _, H, W = img_1chw.shape
    flat_idx = img_1chw.view(-1).argmax()
    row, col = int(flat_idx // W), int(flat_idx % W)

    radius = (BS_region_size - 1) // 2
    r0, r1 = max(row - radius, 0), min(row + radius, H - 1)
    c0, c1 = max(col - radius, 0), min(col + radius, W - 1)

    mask = torch.zeros_like(img_1chw, dtype=torch.bool)
    mask[:, r0 : r1 + 1, c0 : c1 + 1] = True  # True表示在基站附近，保留
    return mask

def BS_building_detection(ref_img, BS_region_size):

    B, C, H, W = ref_img.shape       # C=1
    assert C == 1, "仅适用于单通道"

    BS_mask = torch.stack(
        [single_BS_mask(ref_img[b], BS_region_size) for b in range(B)],
        dim=0               # (B,1,H,W)
    )
    building_mask = ref_img < -1 + 1e-3          # True 表示是建筑物

    return BS_mask, building_mask