'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
import torch
from motionblur.motionblur import Kernel

from .dps_utils.resizer import Resizer
from .dps_utils.img_utils import Blurkernel, fft2_m


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.scale_factor = scale_factor
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        x = self.down_sample(data)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')  # 将下采样后的图像上采样回原始大小，仅复制像素
        return x

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)

@register_operator(name="ray_tracing")
class RayTracingOperator(LinearOperator):
    # 这里把 ray tracing 视为一种特殊的 inpainting 任务。没有使用。
    def __init__(self,
                 device,
                 **kwargs):
        super().__init__()
        self.device = device
        self.default_value = float(kwargs.get("default_value", 0.4)) * 2 - 1  # 应用于-1~1的数据，放缩一下

    # forward : (B,1,H,W) → (B,1,H,W)       —— 处处可导
    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        # BS_mask: True为在基站附近 building_mask: True为建筑物区域
        BS_mask = kwargs.get("BS_mask").to(dtype=data.dtype, device=data.device)
        building_mask = kwargs.get("building_mask").to(dtype=data.dtype, device=data.device)
        assert BS_mask.shape == data.shape == building_mask.shape, "数据形状不一致"

        no_BS_or_building = (BS_mask + building_mask) == 0  # True为非基站且非建筑的区域
        no_BS_or_building = no_BS_or_building.to(dtype=data.dtype, device=data.device)
        # 非基站非建筑区域掩盖，置缺省值。建筑以及基站区域保留梯度传播
        BS_and_building = no_BS_or_building *self.default_value + (1-no_BS_or_building) * data

        return BS_and_building
        
    def transpose(self, data):
        return data

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

@register_operator(name='joint_degradation_recovery')
class JointDegradationOperator(NonLinearOperator):
    """
    通道约定:
      data[:, 0:1]  -> 增益 (已在 [-1, 1]，建筑像素为 -1)
      data[:, 1:2]  -> 角度通道的 sin 映射值 (已在 [-1, 1]，非建筑位于 [-0.4, 1]，建筑为 -1)

    参数:
      lower_bound, upper_bound: 以 [0,1] 标尺给出的增益剪裁上下限；内部会映射到 [-1,1]
      num_levels: 角度量化等级数 (在 [-90°, 90°] 上均分)
      eps: 数值稳定用的小常数
    """
    def __init__(self, device, lower_bound=0.0, upper_bound=1.0, num_levels=10, eps=1e-6):
        super().__init__()
        self.device = device
        self.lb01 = float(lower_bound)
        self.ub01 = float(upper_bound)
        self.num_levels = int(num_levels)
        self.eps = float(eps)

        # [0,1] -> [-1,1] 的增益阈
        self.lb = 2.0 * self.lb01 - 1.0
        self.ub = 2.0 * self.ub01 - 1.0

        # 仅有 sin 时，自洽的角度域是 [-90°, 90°]
        self.ang_min = -90.0
        self.ang_max =  90.0
        self.span = self.ang_max - self.ang_min
        self.delta = self.span / self.num_levels  # 每个量化区间宽度(度)

    def forward(self, data, **kwargs):
        # data: (B, 2, H, W)
        gain = data[:, 0:1, :, :]
        angv = data[:, 1:2, :, :]   # 当前已在 [-1,1]；非建筑∈[-0.4,1]，建筑为 -1
        data_clamped = torch.zeros_like(data, device=data.device)

        # -------------------------
        # (1) 增益：上下限剪裁 + 保留建筑
        # -------------------------
        building_mask = (gain <= -1.0 + self.eps)  # 建筑判定
        gain_clamped = torch.clamp(gain, self.lb, self.ub)
        gain_clamped[building_mask] = -1.0
        data_clamped[:, 0:1, :, :] = gain_clamped

        # -------------------------
        # (2) 角度：从映射值 -> sin -> 角度 -> 量化 -> 回写映射值
        # 映射关系：v = 0.7*sin + 0.3  =>  sin = (v - 0.3)/0.7
        # -------------------------
        # 仅对“非建筑”位置做角度量化，建筑保持 -1
        non_building = ~building_mask

        # 还原 sin ∈ [-1,1]
        sin_val = torch.empty_like(angv)
        sin_val[non_building] = (angv[non_building] - 0.3) / 0.7
        sin_val = torch.clamp(sin_val, -1.0, 1.0)

        # sin -> angle(deg) ∈ [-90,90]
        angle_deg = torch.rad2deg(torch.asin(sin_val))

        # 区间编号 & 中线量化
        idx = torch.floor((angle_deg - self.ang_min) / self.delta)
        idx = torch.clamp(idx, 0, self.num_levels - 1)
        angle_q = self.ang_min + self.delta * (idx + 0.5)

        # 回到 sin，再映射回 [-1,1] 中的角度通道值：v_q = 0.7*sin(angle_q) + 0.3
        sin_q = torch.sin(torch.deg2rad(angle_q))
        v_q = 0.7 * sin_q + 0.3

        # 写回（仅非建筑）；建筑置为 -1
        out_ang = torch.empty_like(angv)
        out_ang[non_building] = v_q[non_building]
        out_ang[building_mask] = -1.0

        data_clamped[:, 1:2, :, :] = out_ang
        return data_clamped
    
@register_operator(name='joint_degradation_recovery_ste')
class JointDegradationOperatorSTE(NonLinearOperator):
    # 使用 STE 实现。最终使用了这个。
    """
    Joint degradation with hard quantization + STE on the angle channel.

    通道约定:
      data[:, 0:1] -> 增益 ([-1, 1]，建筑像素为 -1)
      data[:, 1:2] -> 角度通道映射值 v ∈ [-1, 1]
                       非建筑像素约在 [-0.4, 1]，建筑像素为 -1

    参数:
      lower_bound, upper_bound: [0,1] 标尺下的增益裁剪上下限（内部映射到 [-1,1]）
      num_levels              : 角度量化等级数 (在 [-90°, 90°] 上均分)
      eps                     : 建筑判定 & 数值稳定小常数
    """
    def __init__(self, device,
                 lower_bound=0.0,
                 upper_bound=1.0,
                 num_levels=10,
                 eps=1e-6):
        super().__init__()
        self.device = device
        self.lb01 = float(lower_bound)
        self.ub01 = float(upper_bound)
        self.num_levels = int(num_levels)
        self.eps = float(eps)

        # [0,1] -> [-1,1] 的增益裁剪阈值
        self.lb = 2.0 * self.lb01 - 1.0
        self.ub = 2.0 * self.ub01 - 1.0

        # 角度域 [-90°, 90°] 均匀量化
        self.ang_min = -90.0
        self.ang_max =  90.0
        self.span = self.ang_max - self.ang_min
        self.delta = self.span / self.num_levels  # 每个区间宽度 (度)

        # 预先在“角度→像素值”空间中计算：K+1 个边界 + K 个中心
        # 边界角度: θ_k = ang_min + k * delta, k=0,...,K
        k_edges = torch.arange(self.num_levels + 1,
                               dtype=torch.float32,
                               device=self.device)
        angles_edges_deg = self.ang_min + self.delta * k_edges
        angles_edges_rad = angles_edges_deg * torch.pi / 180.0
        sin_edges = torch.sin(angles_edges_rad)
        # v = 0.7 * sinθ + 0.3
        boundaries_v = 0.7 * sin_edges + 0.3   # shape (K+1,)
        self.boundaries_v = boundaries_v

        # 中心角度: θ_c = ang_min + delta*(k+0.5), k=0,...,K-1
        k_centers = torch.arange(self.num_levels,
                                 dtype=torch.float32,
                                 device=self.device)
        centers_deg = self.ang_min + self.delta * (k_centers + 0.5)
        centers_rad = centers_deg * torch.pi / 180.0
        sin_centers = torch.sin(centers_rad)
        centers_v = 0.7 * sin_centers + 0.3    # shape (K,)
        self.centers_v = centers_v

    def _quantize_v_hard(self, v_flat: torch.Tensor) -> torch.Tensor:
        """
        输入: v_flat, shape (N,)
        输出: v_hard_flat, shape (N,)
        规则: v 位于 [boundaries_v[k], boundaries_v[k+1]) -> centers_v[k]
        """
        # bucketize: 返回 idx，使 boundaries[idx-1] <= v < boundaries[idx]
        idx = torch.bucketize(v_flat, self.boundaries_v) - 1  # 令 idx ∈ {0,...,K-1}
        idx = idx.clamp(min=0, max=self.num_levels - 1)
        v_hard_flat = self.centers_v[idx]
        return v_hard_flat

    def forward(self, data, **kwargs):
        # data: (B, 2, H, W)
        gain = data[:, 0:1, :, :]
        angv = data[:, 1:2, :, :]  # ∈ [-1,1]，非建筑 ≈ [-0.4,1]，建筑 = -1

        # -------------------------
        # (1) 增益通道：裁剪 + 建筑像素固定为 -1
        # -------------------------
        building_mask = (gain <= -1.0 + self.eps)  # bool, shape (B,1,H,W)

        gain_clamped = torch.clamp(gain, self.lb, self.ub)
        gain_clamped = torch.where(
            building_mask,
            torch.full_like(gain_clamped, -1.0),
            gain_clamped,
        )

        # -------------------------
        # (2) 角度通道：硬量化 + STE
        # -------------------------
        non_building = ~building_mask           # 只在非建筑位置做角度操作
        out_ang = torch.full_like(angv, -1.0)   # 先全部设为建筑值 -1

        if non_building.any():
            v = angv[non_building]             # shape (N,)
            v_flat = v.view(-1)                # (N,)

            # 前向: 真正硬量化
            v_hard_flat = self._quantize_v_hard(v_flat)
            v_hard = v_hard_flat.view_as(v)

            # STE: v_out = v + (v_hard - v).detach()
            #  - forward: v_out = v_hard
            #  - backward: d v_out / d v = 1
            v_out = v + (v_hard - v).detach()

            out_ang[non_building] = v_out

        # -------------------------
        # (3) 组装输出
        # -------------------------
        out = data.clone()
        out[:, 0:1, :, :] = gain_clamped
        out[:, 1:2, :, :] = out_ang
        return out



__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma