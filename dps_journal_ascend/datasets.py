import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os, re, random
from torchvision.transforms import functional as TF

def _canonical_key(bn: str) -> str:
    # 规范化 key：忽略大小写、空格，并移除 'gain'/'aoa' 词片，便于鲁棒匹配
    s = bn.lower().replace(' ', '')
    s = re.sub(r'(gain|aoa)', '', s)
    s = s.replace('__', '_')
    return s

def make_pair_loader(aoa_split_root: str,
                     image_size: int,
                     do_random_flip: bool,
                     uniform_dequantization: bool,
                     strict: bool = True):
    rng = random.Random()
    size = (image_size, image_size)

    def _pair_loader(gain_path: str):
        # gain_path: .../CKM_gain_128/<split>/<class>/<filename>.png
        cls_dir = os.path.basename(os.path.dirname(gain_path))     # 类别目录
        aoa_dir = os.path.join(aoa_split_root, cls_dir)

        gbn = os.path.basename(gain_path)
        # basename中加上AoA
        name, ext = os.path.splitext(gbn)
        abn_guess = f"{name}_AoA{ext}"
        apath = os.path.join(aoa_dir, abn_guess)
        apath = re.sub(r'image_128', 'image_128_AoA', apath, flags=re.IGNORECASE)


        if not os.path.isfile(apath):
            # 规范化键匹配
            key = _canonical_key(gbn)
            found = None
            try:
                for cand in os.listdir(aoa_dir):
                    if _canonical_key(cand) == key:
                        found = os.path.join(aoa_dir, cand)
                        break
            except FileNotFoundError:
                found = None
            if found is None:
                if strict:
                    raise FileNotFoundError(f"AOA pair not found for {gain_path}")
            else:
                apath = found

        # 读灰度
        g = Image.open(gain_path).convert('L')
        a = Image.open(apath).convert('L')

        # resize
        g = TF.resize(g, size, antialias=True)
        a = TF.resize(a, size, antialias=True)
        
        # 转为张量
        gt = TF.to_tensor(g)   # 1xHxW
        at = TF.to_tensor(a)   # 1xHxW
            
        # 根据增益地图生成建筑地图  建筑为1，非建筑为0
        bt = at < 1e-3
        bt = bt.float()  # 1xHxW

        # 拼成 3 通道
        x = torch.cat([gt, at, bt], dim=0)  # 3xHxW

        if uniform_dequantization:
            x = torch.clamp(x + torch.rand_like(x) / 256.0, 0.0, 1.0)

        return x

    return _pair_loader


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x

def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x
    
class UniformDequantization(object):
    """Apply random uniform dequantization as a torchvision transform."""
    def __call__(self, img):
        # img is a tensor in [0,1]
        return (img * 255. + torch.rand_like(img)) / 256.

class DictWrapperDataset(torch.utils.data.Dataset):
    """
    Wrap torchvision dataset so that __getitem__ returns a dict with
      'image'      : Tensor
      'label'      : int
      'filename'   : basename of the file (e.g.  'SF_128_BS 1_0_5.png')
      'index'      : original numeric index  (handy for distributed gather)
    """
    # 用于CKM图像，因为要返回索引和名字
    def __init__(self, dataset):
        self.dataset = dataset                      # e.g. ImageFolder

    def __getitem__(self, idx):
        img, label = self.dataset[idx]              # ImageFolder still returns (img, label)
        if hasattr(self.dataset, "samples"):        # ImageFolder/SVHN/etc.
            path = self.dataset.samples[idx][0]     # absolute path
        else:                                       # fallback
            path = getattr(self.dataset, "data", [])[idx]
        return {
            "image":    img,
            "label":    label,
            "filename": os.path.basename(path),
            "index":    torch.tensor(idx, dtype=torch.long)  # tensor → easy to all_gather
        }

    def __len__(self):
        return len(self.dataset)

# class DictWrapperDataset(torch.utils.data.Dataset):
#     """
#     A small wrapper to yield a dict with 'image' and 'label'
#     so that the rest of the code can consume it in the same way
#     as the original TF code (which returned {'image':..., 'label':...}).
#     """
#     def __init__(self, dataset):
#         self.dataset = dataset
#     def __getitem__(self, idx):
#         img, label = self.dataset[idx]
#         # Return exactly the same structure that original code expects
#         return {"image": img, "label": label}
#     def __len__(self):
#         return len(self.dataset)

def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """
    Create data loaders for training and evaluation.

    Returns:
      train_ds, eval_ds, dataset_builder (the latter is None for now).
    """
    train_batch_size = config.training.batch_size
    eval_batch_size = config.eval.batch_size

    # We'll define basic transforms for each dataset

    transform_list = []
    # Resize
    transform_list.append(transforms.Resize((config.data.image_size, config.data.image_size)))
    # Random horizontal flip
    if not evaluation and config.data.random_flip:
      transform_list.append(transforms.RandomHorizontalFlip())
    # For many of these dataset settings, images are in [0,1] after transforms.ToTensor().
    transform_list.append(transforms.ToTensor())
    # Add noise for generative models
    if uniform_dequantization:
        transform_list.append(UniformDequantization())

    transform = transforms.Compose(transform_list)

    # For demonstration, we show CIFAR10, SVHN, CELEBA, LSUN, etc.
    # In practice you can add further logic to handle more datasets as needed.

    if config.data.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root=os.path.join(config.data.data_dir, 'cifar10'),
            train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=os.path.join(config.data.data_dir, 'cifar10'),
            train=False, download=True, transform=transform
        )
    elif config.data.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(
            root=os.path.join(config.data.data_dir, 'mnist'),
            train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=os.path.join(config.data.data_dir, 'mnist'),
            train=False, download=True, transform=transform
        )
    elif config.data.dataset == 'CKM_gain_128':
        # 假设数据目录结构如下：
        # data_dir/CKM_gain_128/train/<class_name>/*.png
        # data_dir/CKM_gain_128/eval/ <class_name>/*.png
        base_dir   = os.path.join(config.data.data_dir, 'CKM_gain_128')
        train_dir  = os.path.join(base_dir, 'train')
        eval_dir   = os.path.join(base_dir, 'eval')
        trainset = torchvision.datasets.ImageFolder(
            root=train_dir,
            loader=lambda path: Image.open(path).convert('L'),
            transform=transform
        )
        testset = torchvision.datasets.ImageFolder(
            root=eval_dir,
            loader=lambda path: Image.open(path).convert('L'),
            transform=transform
        )
    elif config.data.dataset == 'CKM_AoA_128':
        # 假设数据目录结构如下：
        # data_dir/CKM_AoA_128/train/<class_name>/*.png
        # data_dir/CKM_AoA_128/eval/ <class_name>/*.png
        base_dir   = os.path.join(config.data.data_dir, 'CKM_AoA_128')
        train_dir  = os.path.join(base_dir, 'train')
        eval_dir   = os.path.join(base_dir, 'eval')
        trainset = torchvision.datasets.ImageFolder(
            root=train_dir,
            loader=lambda path: Image.open(path).convert('L'),
            transform=transform
        )
        testset = torchvision.datasets.ImageFolder(
            root=eval_dir,
            loader=lambda path: Image.open(path).convert('L'),
            transform=transform
        )
    elif config.data.dataset == 'CKM_gain_AoA_128':
        # 目录结构要求与单通道版本一致：
        # CKM_gain_128/{train,eval}/<class>/*.png
        # CKM_AoA_128/{train,eval}/<class>/*.png
        gain_base = os.path.join(config.data.data_dir, 'CKM_gain_128')
        aoa_base  = os.path.join(config.data.data_dir, 'CKM_AoA_128')

        # 训练：允许随机翻转；评测：禁止随机翻转
        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(gain_base, 'train'),
            loader=make_pair_loader(
                aoa_split_root=os.path.join(aoa_base, 'train'),
                image_size=config.data.image_size,
                do_random_flip=False,
                uniform_dequantization=False,
                strict=True
            ),
            transform=None  # 变换已在 loader 里完成
        )
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(gain_base, 'eval'),
            loader=make_pair_loader(
                aoa_split_root=os.path.join(aoa_base, 'eval'),
                image_size=config.data.image_size,
                do_random_flip=False,
                uniform_dequantization=False,
                strict=True
            ),
            transform=None
        )
    else:
        # Fallback or raise an error if dataset not implemented
        raise NotImplementedError(f"Dataset {config.data.dataset} not yet supported with PyTorch loaders.")

    # Wrap them so that we yield the same kind of dict: {'image': tensor, 'label': label}
    train_dataset = DictWrapperDataset(trainset)
    eval_dataset  = DictWrapperDataset(testset)

    # Now create the DataLoader for each
    train_ds = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=(not evaluation), drop_last=True, num_workers=0, pin_memory=True
    )
    eval_ds = torch.utils.data.DataLoader(
        eval_dataset, batch_size=eval_batch_size, shuffle=evaluation, drop_last=False, num_workers=0, pin_memory=True
    )

    return train_ds, eval_ds, None