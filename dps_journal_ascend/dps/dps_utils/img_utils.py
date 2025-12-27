import numpy as np
import torch
import scipy
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from motionblur.motionblur import Kernel
from .fastmri_utils import fft2c_new, ifft2c_new

"""
Helper functions for new types of inverse problems
"""

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
  """ IFFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(x)


def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img


def prepare_im(load_dir, image_size, device):
    ref_img = torch.from_numpy(normalize_np(plt.imread(load_dir)[:, :, :3].astype(np.float32))).to(device)
    ref_img = ref_img.permute(2, 0, 1)
    ref_img = ref_img.view(1, 3, image_size, image_size)
    ref_img = ref_img * 2 - 1
    return ref_img


def fold_unfold(img_t, kernel, stride):
    img_shape = img_t.shape
    B, C, H, W = img_shape
    print("\n----- input shape: ", img_shape)

    patches = img_t.unfold(3, kernel, stride).unfold(2, kernel, stride).permute(0, 1, 2, 3, 5, 4)

    print("\n----- patches shape:", patches.shape)
    # reshape output to match F.fold input
    patches = patches.contiguous().view(B, C, -1, kernel*kernel)
    print("\n", patches.shape) # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches = patches.permute(0, 1, 3, 2)
    print("\n", patches.shape) # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches = patches.contiguous().view(B, C*kernel*kernel, -1)
    print("\n", patches.shape) # [B, C*prod(kernel_size), L] as expected by Fold

    output = F.fold(patches, output_size=(H, W),
                    kernel_size=kernel, stride=stride)
    # mask that mimics the original folding:
    recovery_mask = F.fold(torch.ones_like(patches), output_size=(
        H, W), kernel_size=kernel, stride=stride)
    output = output/recovery_mask

    return patches, output


def reshape_patch(x, crop_size=128, dim_size=3):
    x = x.transpose(0, 2).squeeze()  # [9, 3*(128**2)]
    x = x.view(dim_size**2, 3, crop_size, crop_size)
    return x

def reshape_patch_back(x, crop_size=128, dim_size=3):
    x = x.view(dim_size**2, 3*(crop_size**2)).unsqueeze(dim=-1)
    x = x.transpose(0, 2)
    return x


class Unfolder:
    def __init__(self, img_size=256, crop_size=128, stride=64):
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride

        self.unfold = nn.Unfold(crop_size, stride=stride)
        self.dim_size = (img_size - crop_size) // stride + 1

    def __call__(self, x):
        patch1D = self.unfold(x)
        patch2D = reshape_patch(patch1D, crop_size=self.crop_size, dim_size=self.dim_size)
        return patch2D


def center_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

class Folder:
    def __init__(self, img_size=256, crop_size=128, stride=64):
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride

        self.fold = nn.Fold(img_size, crop_size, stride=stride)
        self.dim_size = (img_size - crop_size) // stride + 1

    def __call__(self, patch2D):
        patch1D = reshape_patch_back(patch2D, crop_size=self.crop_size, dim_size=self.dim_size)
        return self.fold(patch1D)

# --------------------------------------------------------------------- #
#  Mask generator                                                       #
# --------------------------------------------------------------------- #
class mask_generator:
    """
    Binary mask generator.
    New CKM family (ckm_box / ckm_random / ckm_both) never masks buildings (≈ -1).

    Returns
    -------
    mask    : Tensor [B, C, H, W]   (1 = keep, 0 = drop)
    success : bool                  (False ⇔ 至少有一个样本未找到合法 box)
    """
    def __init__(self,
                 mask_type: str,
                 mask_len_range=None,
                 mask_prob_range=None,
                 margin=None,
                 max_trials: int = 1,
                 np_rng=None,
                 torch_gen=None):
        legal = {'box', 'random', 'both', 'extreme',
                 'ckm_box', 'ckm_random', 'ckm_both'}
        assert mask_type in legal, f"unknown mask_type {mask_type}"
        self.type       = mask_type
        self.len_rng    = mask_len_range or (32, 128)
        self.prob_rng   = mask_prob_range or (0.30, 0.70)
        self.margin_h, self.margin_w = margin or (16, 16)
        self.max_trials = max_trials
        self.np_rng = np_rng or np.random.default_rng()
        self.torch_gen = torch_gen

    # ---------------------------- helpers ----------------------------- #
    def _try_box_once(self, allowed: torch.Tensor,
                      H: int, W: int) -> tuple[torch.Tensor, bool]:
        """Sample one box wholly inside allowed. Return mask & success."""
        min_len, max_len = self.len_rng
        h = self.np_rng.integers(min_len, max_len)
        w = self.np_rng.integers(min_len, max_len)

        # ── 找合法 top 行 ─────────────────────────────────────
        cand = allowed[:, :, :H - h + 1, :W - w + 1]          # 裁边避免越界
        cand = cand.unfold(2, h, 1).unfold(3, w, 1)           # [B,C,Hy,Wx,h,w]
        ok   = cand.all(-1).all(-1)                           # → [B,C,Hy,Wx]
        ok   = ok.all(0).any(0)                               # 合并 batch / channel
        valid_y = np.where(ok.any(1).cpu().numpy())[0]
        if len(valid_y) == 0:
            return None, False
        top = int(self.np_rng.choice(valid_y))

        # ── 找合法 left 列 ────────────────────────────────────
        row_allowed = allowed[0, 0, top:top + h].all(0)       # [W]
        win_ok = torch.all(row_allowed.unfold(0, w, 1), dim=-1)  # [W-w+1]
        valid_x = np.where((row_allowed[:W - w + 1] & win_ok).cpu().numpy())[0]
        if len(valid_x) == 0:
            return None, False
        left = int(self.np_rng.choice(valid_x))

        mask = torch.ones_like(allowed)
        mask[:, :, top:top + h, left:left + w] = 0
        return mask, True

    def _ckm_box_mask(self, img: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Batched CKM rectangular mask."""
        B, C, H, W = img.shape
        allowed = (img > -0.999).float()  # 1 = signal, 0 = building
        masks   = torch.ones_like(img)
        success_all = True
        for b in range(B):
            ok = False
            for _ in range(self.max_trials):
                m, ok = self._try_box_once(allowed[b:b + 1], H, W)
                if ok:
                    masks[b] = m
                    break
            success_all &= ok
        return masks, success_all

    def _ckm_random_mask(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        allowed = (img > -0.999).float()
        prob = self.np_rng.uniform(*self.prob_rng)
        m_np = (self.np_rng.random((B, 1, H, W)) > prob).astype(np.float32)  # 1=keep
        rand = torch.from_numpy(m_np).to(img.device)  # 移到与图像相同设备
        return 1 - allowed + allowed * rand.repeat(1, C, 1, 1)

    # ------------------- original box / random ------------------------ #
    def _box_mask(self, B, C, H, W):
        masks = torch.ones(B, C, H, W)
        min_len, max_len = self.len_rng
        for b in range(B):
            h = self.np_rng.integers(min_len, max_len)
            w = self.np_rng.integers(min_len, max_len)
            top  = self.np_rng.integers(self.margin_h, H - self.margin_h - h)
            left = self.np_rng.integers(self.margin_w, W - self.margin_w - w)
            masks[b, :, top:top + h, left:left + w] = 0
        return masks

    def _random_mask(self, B, C, H, W):
        prob = self.np_rng.uniform(*self.prob_rng)  # drop 概率
        m_np = (self.np_rng.random((B, 1, H, W)) > prob).astype(np.float32)  # 1=keep
        mask = torch.from_numpy(m_np)  # CPU tensor, shares memory with NumPy
        return mask.repeat(1, C, 1, 1)

    # ------------------------- public API ----------------------------- #
    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, bool]:
        B, C, H, W = img.shape
        success = True

        if self.type in {'box', 'random', 'both', 'extreme'}:
            if self.type == 'box':
                mask = self._box_mask(B, C, H, W)
            elif self.type == 'random':
                mask = self._random_mask(B, C, H, W)
            elif self.type == 'both':
                mask = self._box_mask(B, C, H, W) * self._random_mask(B, C, H, W)
            else:  # extreme
                mask = 1.0 - self._box_mask(B, C, H, W)

        else:  # CKM 系列
            if self.type == 'ckm_box':
                mask, success = self._ckm_box_mask(img)
            elif self.type == 'ckm_random':
                mask = self._ckm_random_mask(img)
            else:  # ckm_both
                box_mask, success = self._ckm_box_mask(img)
                rand_mask         = self._ckm_random_mask(img)
                mask = box_mask * rand_mask

        return mask, success

def unnormalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img / scaling


def normalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img * scaling


def dynamic_thresholding(img, s=0.95):
    img = normalize(img, s=s)
    return torch.clip(img, -1., 1.)


def get_gaussian_kernel(kernel_size=31, std=0.5):
    n = np.zeros([kernel_size, kernel_size])
    n[kernel_size//2, kernel_size//2] = 1
    k = scipy.ndimage.gaussian_filter(n, sigma=std)
    k = k.astype(np.float32)
    return k


def init_kernel_torch(kernel, device="cuda:0"):
    h, w = kernel.shape
    kernel = Variable(torch.from_numpy(kernel).to(device), requires_grad=True)
    kernel = kernel.view(1, 1, h, w)
    kernel = kernel.repeat(1, 3, 1, 1)
    return kernel


class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


class exact_posterior():
    def __init__(self, betas, sigma_0, label_dim, input_dim):
        self.betas = betas
        self.sigma_0 = sigma_0
        self.label_dim = label_dim
        self.input_dim = input_dim

    def py_given_x0(self, x0, y, A, verbose=False):
        norm_const = 1/((2 * np.pi)**self.input_dim * self.sigma_0**2)
        exp_in = -1/(2 * self.sigma_0**2) * torch.linalg.norm(y - A(x0))**2
        if not verbose:
            return norm_const * torch.exp(exp_in)
        else:
            return norm_const * torch.exp(exp_in), norm_const, exp_in

    def pxt_given_x0(self, x0, xt, t, verbose=False):
        beta_t = self.betas[t]
        norm_const = 1/((2 * np.pi)**self.label_dim * beta_t)
        exp_in = -1/(2 * beta_t) * torch.linalg.norm(xt - np.sqrt(1 - beta_t)*x0)**2
        if not verbose:
            return norm_const * torch.exp(exp_in)
        else:
            return norm_const * torch.exp(exp_in), norm_const, exp_in

    def prod_logsumexp(self, x0, xt, y, A, t):
        py_given_x0_density, pyx0_nc, pyx0_ei = self.py_given_x0(x0, y, A, verbose=True)
        pxt_given_x0_density, pxtx0_nc, pxtx0_ei = self.pxt_given_x0(x0, xt, t, verbose=True)
        summand = (pyx0_nc * pxtx0_nc) * torch.exp(-pxtx0_ei - pxtx0_ei)
        return torch.logsumexp(summand, dim=0)



def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def total_variation_loss(img, weight):
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)).mean()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2)).mean()
    return weight * (tv_h + tv_w)


if __name__ == '__main__':

    device = 'cuda:0'
    load_path = 'D:/controllable_generation/DPS_demo/data/samples/00015.png'
    img = torch.tensor(plt.imread(load_path)[:, :, :3])  #rgb
    img = torch.permute(img, (2, 0, 1)).view(1, 3, 256, 256).to(device)
    
    # img = torch.randn(1, 1, 256, 256).to(device)
    mask_type = 'box'
    mask_len_range = (32, 128)
    mask_prob_range = (0.3, 0.7)
    margin=(16, 16)
    # mask
    mask_gen = mask_generator(
        mask_type=mask_type,
        mask_len_range=mask_len_range,
        mask_prob_range=mask_prob_range,
        margin=margin
    )
    mask = mask_gen(img)  # (batch_size, channel, img_size, img_size)
    mask = mask.to(device)
    print(mask.shape)
    mo = mask * img
    mo = np.transpose(mo.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    plt.imshow(mo)
    plt.show()
    print(mo.shape)
