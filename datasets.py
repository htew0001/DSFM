from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F
import cv2
from absl import logging
from scipy.fftpack import dct

class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


# CIFAR10
class CIFAR10(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, low2high_order=None, reverse_order=None, use_YCbCr=False,  # use YCbCr or RGB
                 Y_bound=None, **kwargs):
        super().__init__()

        self.channels = 3  # RGB
        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        # self.train = dct_rgb(
        #     root_dir=path, resolution=resolution, tokens=tokens,
        #     low_freqs=low_freqs, block_sz=block_sz, low2high_order=low2high_order, reverse_order=reverse_order,
        #     Y_bound=Y_bound, use_YCbCr=use_YCbCr
        # )
        
        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, low2high_order=low2high_order, reverse_order=reverse_order,
            Y_bound=Y_bound
        )

        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        # return self.channels, self.resolution*self.resolution  # (C, H*W)
        return self.tokens, self.low_freqs*6  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'assets/fid_stats/fid_stats_cifar10_train.npz'

    @property
    def has_label(self):
        return False


class FMRI(DatasetFactory):
    def __init__(self, path, dwt_init="outer1", dct_norm_mode="channel_freq", resolution=0, tokens=0, low_freqs=0, block_sz=0, low2high_order=None, reverse_order=None, use_YCbCr=False,  # use YCbCr or RGB
                 Y_bound=None, **kwargs):
        super().__init__()

        self.channels = 3  # RGB
        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        # self.train = dct_rgb(
        #     root_dir=path, resolution=resolution, tokens=tokens,
        #     low_freqs=low_freqs, block_sz=block_sz, low2high_order=low2high_order, reverse_order=reverse_order,
        #     Y_bound=Y_bound, use_YCbCr=use_YCbCr
        # )
        
        # self.train = DCT_4YCbCr(
        #     root_dir=path, img_sz=resolution, tokens=tokens,
        #     low_freqs=low_freqs, block_sz=block_sz, low2high_order=low2high_order, reverse_order=reverse_order,
        #     Y_bound=Y_bound
        # )
        self.train = DCT_Dataset_fmri(
            root_dir=path, img_sz=resolution, 
            block_sz=block_sz, low2high_order=low2high_order, low_freqs=low_freqs, reverse_order=reverse_order,dct_norm_mode=dct_norm_mode,dwt_init=dwt_init)

        # self.train = UnlabeledDataset(self.train)
        labels = self.train.labels  # list of ints, one per file
        self.labels = labels
        self.N = len(self.labels)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        self.num_classes = int(label_tensor.max().item()) + 1
        counts = torch.bincount(label_tensor, minlength=self.num_classes)
        self.class_probs = (counts.float() / counts.sum()).tolist()

        logging.info(f"FMRI dataset initialized with {len(labels)} samples")
        logging.info(f" → Number of classes: {self.num_classes}")
        logging.info(f" → Samples per class: {counts.tolist()}")
        logging.info(f" → Empirical sampling probabilities: {self.class_probs}")

    @property
    def data_shape(self):
        # return self.channels, self.resolution*self.resolution  # (C, H*W)
        return self.tokens, self.low_freqs*12  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'assets/fid_stats/fid_stats_cifar10_train.npz'

    @property
    def has_label(self):
        return True
    
    def sample_label(self, n_samples, device):
        """
        Draw n_samples labels according to the empirical class_probs learned above.
        Returns a (n_samples,) LongTensor on `device`.
        """
        if n_samples == len(self.labels):
            logging.info("exact class distribution sampling")
            all_labels = torch.tensor(self.labels, device=device)
            perm = torch.randperm(self.N, device=device)
            return all_labels[perm]
        else:
            # logging.info("frequency class distribution sampling")
            probs = torch.tensor(self.class_probs, device=device)
            return torch.multinomial(probs, n_samples, replacement=True)

    def label_prob(self, k):
        return self.class_probs[k]


# ImageNet
class FeatureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]

    def __len__(self):
        return 1_281_167 * 2  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f'{idx}.npy')
        z, label = np.load(path, allow_pickle=True)
        return z, label


class ImageNet256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet512Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True):
        super().__init__()

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, 'train'))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, labels=train_labels, random_crop=random_crop, random_flip=random_flip)
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        label = np.array(self.labels[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), label


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# CelebA


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def split_into_blocks(image, block_sz):
    blocks = []
    for i in range(0, image.shape[0], block_sz):
        for j in range(0, image.shape[1], block_sz):
            blocks.append(image[i:i + block_sz, j:j + block_sz])  # first row, then column 
    return np.array(blocks)

def combine_blocks(blocks, height, width, block_sz):
    image = np.zeros((height, width), np.float32)
    index = 0
    for i in range(0, height, block_sz):
        for j in range(0, width, block_sz):
            image[i:i + block_sz, j:j + block_sz] = blocks[index]
            index += 1
    return image

def dct_transform(blocks):
    dct_blocks = []
    for block in blocks:
        dct_block = np.float32(block) - 128  # Shift to center around 0
        dct_block = cv2.dct(dct_block)
        dct_blocks.append(dct_block)
    return np.array(dct_blocks)

def cont_dct_transform(block):
    """
    2D DCT (Type-II) with orthonormal normalization, using SciPy.
    Equivalent to cv2.dct with better theoretical alignment.
    """
    block = np.float32(block)  # Shift to zero-mean
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')  # DCT-II 2D
    return dct_block

def zigzag_orders(img_sz: int):
    """
    Compute JPEG–style zig-zag ordering (low-→ high frequency) and its inverse
    for a square DCT block of size (img_sz × img_sz).

    Returns
    -------
    low2high_order : list[int]
        Flat indices 0 … B²−1 in zig-zag order.
    reverse_order  : list[int]
        Inverse permutation s.t.  reverse_order[low2high_order[i]] == i
        and  low2high_order[reverse_order[j]] == j.
    """
    # --- build zig-zag ------------------------------------------------------
    low2high = []
    for diag in range(2 * img_sz - 1):
        if diag % 2 == 0:          # even diagonal  →  top-to-bottom
            for j in range(diag + 1):
                i = diag - j
                if i < img_sz and j < img_sz:
                    low2high.append(i * img_sz + j)
        else:                      # odd diagonal   →  bottom-to-top
            for i in range(diag + 1):
                j = diag - i
                if i < img_sz and j < img_sz:
                    low2high.append(i * img_sz + j)

    # --- build inverse permutation -----------------------------------------
    reverse = [0] * len(low2high)
    for new_pos, flat_idx in enumerate(low2high):
        reverse[flat_idx] = new_pos

    return np.asarray(low2high), np.asarray(reverse)

class dct_rgb(Dataset):
    def __init__(self, root_dir, resolution=64, tokens=0, low_freqs=0, block_sz=8, low2high_order=None, reverse_order=None,
                 Y_bound=None, use_YCbCr=False):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.img_paths = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.img_paths.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

        # parameters of DCT design
        self.resolution = resolution
        self.low2high_order = low2high_order
        self.reverse_order = reverse_order
        self.low_freqs = low_freqs
        self.use_YCbCr = use_YCbCr

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # img.save('original_img.jpg')
        img = transforms.RandomHorizontalFlip()(img)  # do data augmentation by PIL
        img = np.array(img) / 255.0

        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        if self.use_YCbCr:
            # --- Step 2: RGB to YCbCr (float-difference method) ---
            Y  = 0.299 * R + 0.587 * G + 0.114 * B
            Cb = (B - Y) / 1.772
            Cr = (R - Y) / 1.402
            # --- Step 3: Scale Y, Cb, Cr to [-1, 1] ---
            Y  = (Y  - 0.5) / 0.5
            Cb =  Cb / 0.5
            Cr =  Cr / 0.5
            R = Y; G = Cb; B = Cr
        else:
            # --- Step 2: Scale R, G, B to [-1, 1] ---
            R = (R - 0.5) / 0.5
            G = (G - 0.5) / 0.5
            B = (B - 0.5) / 0.5

        # if self.img_sz != self.re_img_sz:
        #     R = cv2.resize(R, (self.self.re_img_sz, self.self.re_img_sz), interpolation=cv2.INTER_LANCZOS4)
        #     G = cv2.resize(G, (self.self.re_img_sz, self.self.re_img_sz), interpolation=cv2.INTER_LANCZOS4)
        #     B = cv2.resize(B, (self.self.re_img_sz, self.self.re_img_sz), interpolation=cv2.INTER_LANCZOS4)

        dct_r = cont_dct_transform(R)
        dct_g = cont_dct_transform(G)
        dct_b = cont_dct_transform(B)
        dct_rgb = np.stack([dct_r, dct_g, dct_b], axis=0)  # shape: (3, H, W)
        dct_rgb = dct_rgb.reshape(-1, self.resolution * self.resolution)  # shape: (3, H*W)

        sample = dct_rgb[:, self.low2high_order]
        sample = sample[:, :self.low_freqs]
        sample = torch.from_numpy(sample).float()

        return sample

class DCT_4YCbCr(Dataset):
    def __init__(self, root_dir, img_sz=64, tokens=0, low_freqs=0, block_sz=8, low2high_order=None, reverse_order=None,
                 Y_bound=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.img_paths = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.img_paths.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

        # parameters of DCT design
        self.Y_bound = np.array(Y_bound)
        print(f"using Y_bound {self.Y_bound} for training")
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row = int(img_sz / block_sz)
        self.index = []  # index of Y if merging 2*2 Y-block area
        for row in range(0, Y, int(2 * self.Y_blocks_per_row)):  # 0, 32, 64...
            for col in range(0, self.Y_blocks_per_row, 2):  # 0, 2, 4...
                self.index.append(row + col)
        assert len(self.index) == int(Y / 4)

        self.low2high_order = low2high_order
        self.reverse_order = reverse_order

        # token sequence: 4Y-Cb-Cr-4Y-Cb-Cr...
        self.cb_index = [i for i in range(4, tokens, 6)]
        self.cr_index = [i for i in range(5, tokens, 6)]
        self.y_index = [i for i in range(0, tokens) if i not in self.cb_index and i not in self.cr_index]
        assert len(self.y_index) + len(self.cb_index) + len(self.cr_index) == tokens

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]

        #Change to np.load
        img = Image.open(img_path).convert('RGB')
        # img.save('original_img.jpg')
        img = transforms.RandomHorizontalFlip()(img)  # do data augmentation by PIL
        y_blocks = split_into_blocks(img_y, self.block_sz)  # Y component, (64, 64) --> (256, 4, 4)
        dct_y_blocks = dct_transform(y_blocks)  # (256, 4, 4)

        DCT_blocks = np.array(DCT_blocks).reshape(-1, 6, self.block_sz*self.block_sz)  # (64, 6, 4, 4) --> (64, 6, 16)


        DCT_blocks = DCT_blocks[:, :, self.low2high_order]  # (64, 6, 16) --> (64, 6, 16)
        DCT_blocks = DCT_blocks[:, :, :self.low_freqs]  # (64, 6, 16) --> (64, 6, low_freq_coe)

        # numpy to torch
        DCT_blocks = torch.from_numpy(DCT_blocks).reshape(self.tokens, -1)  # (64, 6*low_freq_coe)
        DCT_blocks = DCT_blocks.float()  # float64 --> float32
        # print("DCT_blocks shape after truncation:", DCT_blocks.shape)

        return DCT_blocks


class DCT_4YCbCr_cond(Dataset):
    def __init__(self, img_sz=64, tokens=0, low_freqs=0, block_sz=8, low2high_order=None, reverse_order=None,
                 train_files=None, labels=None, Y_bound=None):

        self.image_paths = train_files
        self.labels = labels

        # parameters of DCT design
        self.Y_bound = np.array(Y_bound)
        print(f"using Y_bound {self.Y_bound} for training")
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row = int(img_sz / block_sz)
        self.index = []  # index of Y if merging 2*2 Y-block area
        for row in range(0, Y, int(2 * self.Y_blocks_per_row)):  # 0, 32, 64...
            for col in range(0, self.Y_blocks_per_row, 2):  # 0, 2, 4...
                self.index.append(row + col)
        assert len(self.index) == int(Y / 4)

        self.low2high_order = low2high_order
        self.reverse_order = reverse_order

        # token sequence: 4Y-Cb-Cr-4Y-Cb-Cr...
        self.cb_index = [i for i in range(4, tokens, 6)]
        self.cr_index = [i for i in range(5, tokens, 6)]
        self.y_index = [i for i in range(0, tokens) if i not in self.cb_index and i not in self.cr_index]
        assert len(self.y_index) + len(self.cb_index) + len(self.cr_index) == tokens

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # img.save('original_img.jpg')
        img = transforms.RandomHorizontalFlip()(img)  # do data augmentation by PIL
        img = np.array(img)

        # Step 1: Convert RGB to YCbCr
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img_y = 0.299 * R + 0.587 * G + 0.114 * B
        img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

        cb_downsampled = cv2.resize(img_cb, (img_cb.shape[1] // 2, img_cb.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)
        cr_downsampled = cv2.resize(img_cr, (img_cr.shape[1] // 2, img_cr.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)

        # Step 2: Split the Y, Cb, and Cr components into 4x4 blocks
        y_blocks = split_into_blocks(img_y, self.block_sz)  # Y component, (64, 64) --> (256, 4, 4)
        cb_blocks = split_into_blocks(cb_downsampled, self.block_sz)  # Cb component, (32, 32) --> (64, 4, 4)
        cr_blocks = split_into_blocks(cr_downsampled, self.block_sz)  # Cr component, (32, 32) --> (64, 4, 4)

        # Step 3: Apply DCT on each block
        dct_y_blocks = dct_transform(y_blocks)  # (256, 4, 4)
        dct_cb_blocks = dct_transform(cb_blocks)  # (64, 4, 4)
        dct_cr_blocks = dct_transform(cr_blocks)  # (64, 4, 4)

        # Step 4: organize the token order by Y-Y-Y-Y-Cb-Cr (2_blocks*2_blocks pixel region)
        DCT_blocks = []
        for i in range(dct_cr_blocks.shape[0]):
            DCT_blocks.append([
                dct_y_blocks[self.index[i]],  # Y
                dct_y_blocks[self.index[i] + 1],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row + 1],  # Y
                dct_cb_blocks[i],  # Cb
                dct_cr_blocks[i],  # Cr
            ])
        DCT_blocks = np.array(DCT_blocks).reshape(-1, 6, self.block_sz * self.block_sz)  # (64, 6, 4, 4) --> (64, 6, 16)

        # Step 5: scale into [-1, 1]
        assert DCT_blocks.shape == (self.tokens, 6, self.block_sz * self.block_sz)
        DCT_blocks[:, :4:] = DCT_blocks[:, :4:] / self.Y_bound
        DCT_blocks[:, 4, :] = DCT_blocks[:, 4, :] / self.Y_bound
        DCT_blocks[:, 5, :] = DCT_blocks[:, 5, :] / self.Y_bound

        # Step 6: reorder coe from low to high freq, then mask out high-freq signals
        DCT_blocks = DCT_blocks[:, :, self.low2high_order]  # (64, 6, 16) --> (64, 6, 16)
        DCT_blocks = DCT_blocks[:, :, :self.low_freqs]  # (64, 6, 16) --> (64, 6, low_freq_coe)

        # numpy to torch
        DCT_blocks = torch.from_numpy(DCT_blocks).reshape(self.tokens, -1)  # (64, 6*low_freq_coe)
        DCT_blocks = DCT_blocks.float()  # float64 --> float32

        label = np.array(self.labels[idx], dtype=np.int64)

        return DCT_blocks, label


def split_into_blocks_fmri(image, block_sz):
    # image: (H, W, C)
    H, W, C = image.shape
    all_blocks = []
    for ch in range(C):
        blocks = []
        for i in range(0, H, block_sz):
            for j in range(0, W, block_sz):
                blocks.append(image[i:i + block_sz, j:j + block_sz, ch])
        all_blocks.append(np.array(blocks))  # shape: (num_blocks, block_sz, block_sz)
    # Return shape: (C, num_blocks, block_sz, block_sz)
    return np.stack(all_blocks, axis=0)

def dct_transform_fmri(blocks):
    # blocks: (C, num_blocks, block_sz, block_sz)
    C, num_blocks, H, W = blocks.shape
    dct_blocks = []
    for ch in range(C):
        ch_dct = []
        for block in blocks[ch]:
            dct_block = np.float32(block) 
            # dct_block = np.float32(block) 
            dct_block = cv2.dct(dct_block)
            ch_dct.append(dct_block)
        dct_blocks.append(np.array(ch_dct))
    return np.stack(dct_blocks, axis=0)  # shape: (C, num_blocks, block_sz, block_sz)


class DCT_Dataset_fmri(Dataset):
    def __init__(self, root_dir, img_sz=116, block_sz=8, low2high_order=None, low_freqs=16, reverse_order=None,dct_norm_mode="channel_freq",dwt_init="outer1"):
        self.root_dir = root_dir
        self.img_paths = []
        self.labels = []
        for cls in os.listdir(root_dir):
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.img_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(int(cls))  # assumes folder names are numeric

        self.block_sz = block_sz
        self.low2high_order = low2high_order if low2high_order is not None else np.arange(block_sz*block_sz)

        self.reverse_order = (
            reverse_order if reverse_order is not None
            else np.arange(block_sz * block_sz)
        )
        self.low_freqs = low_freqs

        H, W = img_sz, img_sz
        self.n_blocks = (H // block_sz) * (W // block_sz)

        self.channels   = 12
        self.tokens     = self.n_blocks
        self.data_shape = (self.tokens, self.channels * self.low_freqs)


        if  dwt_init.startswith("outer"):
            outer_number = dwt_init.replace("outer", "")
            Y_bound_path = YOUR_PATH
            logging.info(f"training: initialize ybound of path {Y_bound_path}")
        else:
            logging.info(f"training: no initialize ybound")

        mat = np.load(Y_bound_path).astype(np.float32)  # (12,16)

        if dct_norm_mode == "channel_freq":
            print("inside datasetnorm: ", dct_norm_mode)
            logging.info("inside datasetnorm: channel_freq")
            # per channel-frequency norm
            self.Y_bound = torch.tensor(mat).view(1, 12, 16)
        elif dct_norm_mode == "channel":
            print("inside datasetnorm: ", dct_norm_mode)
            logging.info("inside datasetnorm: channel")
            # global norm 
            # per channel norm 
            mat = np.abs(mat).max(axis=0)
            self.Y_bound = torch.tensor(mat).view(1, 1, -1)
        else:
            print("no init dct norm mode")
            logging.info("no init dct norm mode")

        # vec = np.load(Y_bound_path).astype(np.float32)     # (8,)
        # self.Y_bound = torch.tensor(vec).view(1, 1, 8)      # shape (1,1,8)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = np.load(img_path)  # shape: (116, 116, 6)
        # min_val = img.min()
        # max_val = img.max()
        # print(f"DCT min: {min_val}, max: {max_val}")
        
        # Data Augmentation
        # if np.random.rand() > 0.5:
        #     img = np.flip(img, axis=1)
        
        blocks = split_into_blocks_fmri(img, self.block_sz)  # (6, n_blocks, 8, 8)
        dct_blocks = dct_transform_fmri(blocks)              # (6, n_blocks, 8, 8)
        n_blocks = dct_blocks.shape[1]

        # Pack as (n_blocks, 6, 64)
        DCT_blocks = dct_blocks.transpose(1,0,2,3).reshape(n_blocks, 12, -1)
        
        # Keep only low frequency coefficients
        DCT_blocks = DCT_blocks[:,:,self.low2high_order][:,:,:self.low_freqs]
        # print("thechaannneel size: ",DCT_blocks.shape[1])

        DCT_blocks    = torch.as_tensor(DCT_blocks, dtype=torch.float32)        # to Tensor
        Yb     = self.Y_bound.to(DCT_blocks.device)                      # (1,12,8) or (1,1,8)
        DCT_blocks    = DCT_blocks / (Yb + 1e-6) 
        # min_val = DCT_blocks.min()
        # max_val = DCT_blocks.max()
        # print(f"DCT min: {min_val}, max: {max_val}")
        
        if isinstance(DCT_blocks, np.ndarray):
            DCT_blocks = torch.from_numpy(DCT_blocks)

        DCT_blocks = DCT_blocks.float().reshape(n_blocks, -1)                               # avoid /0
        
        # Flatten for model input if needed
        # DCT_blocks = torch.from_numpy(DCT_blocks).float().reshape(n_blocks, -1)  # (n_blocks, 6*low_freqs)
        # DCT_blocks = DCT_blocks / self.Y_bound
        # return DCT_blocks, label
        return DCT_blocks, label

# class DCT_Dataset_fmri(Dataset):
#     def __init__(self, root_dir, img_sz=116, block_sz=8, low2high_order=None, low_freqs=16, reverse_order=None):
#         self.root_dir = root_dir
#         self.img_paths = []
#         self.labels = []
#         for cls in os.listdir(root_dir):
#             cls_dir = os.path.join(root_dir, cls)
#             for img_name in os.listdir(cls_dir):
#                 self.img_paths.append(os.path.join(cls_dir, img_name))
#                 self.labels.append(int(cls))  # assumes folder names are numeric

#         self.block_sz = block_sz
#         self.low2high_order = low2high_order if low2high_order is not None else np.arange(block_sz*block_sz)

#         self.reverse_order = (
#             reverse_order if reverse_order is not None
#             else np.arange(block_sz * block_sz)
#         )
#         self.low_freqs = low_freqs

#         H, W = img_sz, img_sz
#         self.n_blocks = (H // block_sz) * (W // block_sz)

#         self.channels  = 12
#         self.tokens     = self.n_blocks
#         self.data_shape = (self.tokens, self.channels * self.low_freqs)

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         label = self.labels[idx]
#         img = np.load(img_path)  # shape: (116, 116, 6)
        
#         # Data Augmentation
#         # if np.random.rand() > 0.5:
#         #     img = np.flip(img, axis=1)
        
#         blocks = split_into_blocks_fmri(img, self.block_sz)  # (6, n_blocks, 8, 8)
#         dct_blocks = dct_transform_fmri(blocks)              # (6, n_blocks, 8, 8)
#         n_blocks = dct_blocks.shape[1]

#         # Pack as (n_blocks, 6, 64)
#         DCT_blocks = dct_blocks.transpose(1,0,2,3).reshape(n_blocks, 12, -1)
        
#         # Keep only low frequency coefficients
#         DCT_blocks = DCT_blocks[:,:,self.low2high_order][:,:,:self.low_freqs]
#         # print("thechaannneel size: ",DCT_blocks.shape[1])
        
#         # Flatten for model input if needed
#         DCT_blocks = torch.from_numpy(DCT_blocks).float().reshape(n_blocks, -1)  # (n_blocks, 6*low_freqs)
#         # return DCT_blocks, label
#         min_val = dct_blocks.min()
#         max_val = dct_blocks.max()
#         print(f"DCT min: {min_val}, max: {max_val}")
#         return DCT_blocks


class CelebA(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, low2high_order=None, reverse_order=None,
                 Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        """
        manually download dataset: https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&authuser=0
        then do center crop to 64x64 and set the image folder as the following 'path'
        """
        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, low2high_order=low2high_order, reverse_order=reverse_order,
            Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'assets/fid_stats/fid_stats_celeba64_all.npz'

    @property
    def has_label(self):
        return False


class FFHQ128(DatasetFactory):
    def __init__(self, path, resolution=128, tokens=0, low_freqs=0, block_sz=0, low2high_order=None, reverse_order=None,
                 Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, low2high_order=low2high_order, reverse_order=reverse_order,
            Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'assets/fid_stats/fid_stats_ffhq128_jpg.npz'

    @property
    def has_label(self):
        return False


class FFHQ256(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, low2high_order=None, reverse_order=None,
                 Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        self.train = DCT_4YCbCr(
            root_dir=path, img_sz=resolution, tokens=tokens,
            low_freqs=low_freqs, block_sz=block_sz, low2high_order=low2high_order, reverse_order=reverse_order,
            Y_bound=Y_bound
        )
        # self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return 'assets/fid_stats/fid_stats_ffhq256_jpg.npz'

    @property
    def has_label(self):
        return False


class ImageNet64(DatasetFactory):
    def __init__(self, path, resolution=0, tokens=0, low_freqs=0, block_sz=0, low2high_order=None, reverse_order=None,
                Y_bound=None, **kwargs):
        super().__init__()

        self.resolution = resolution
        self.tokens = tokens
        self.low_freqs = low_freqs

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(path)
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = DCT_4YCbCr_cond(
            img_sz=resolution, tokens=tokens, train_files=train_files, labels=train_labels,
            low_freqs=low_freqs, block_sz=block_sz, low2high_order=low2high_order, reverse_order=reverse_order,
            Y_bound=Y_bound,
        )

        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return self.tokens, self.low_freqs*6  # (96, 43)

    @property
    def fid_stat(self):
        # specify the fid_stats file that will be used for FID computation during the training
        return f'assets/fid_stats/fid_stats_imgnet64_jpg.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# MS COCO
def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
        return z, c


class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
        self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
        assert len(self.train) == 82783
        assert len(self.test) == 40504
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            prompt, context = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'


def get_dataset(name, **kwargs):
    if name == 'cifar10':
        return CIFAR10(**kwargs)
    # elif name == 'imagenet':
    #     return ImageNet(**kwargs)
    # elif name == 'imagenet256_features':
    #     return ImageNet256Features(**kwargs)
    # elif name == 'imagenet512_features':
    #     return ImageNet512Features(**kwargs)
    elif name == 'celeba':
        return CelebA(**kwargs)
    elif name == 'ffhq128':
        return FFHQ128(**kwargs)
    elif name == 'ffhq256':
        return FFHQ256(**kwargs)
    elif name == 'imgnet64':
        return ImageNet64(**kwargs)
    # elif name == 'mscoco256_features':
    #     return MSCOCO256Features(**kwargs)
    elif name == 'fmri':
        return FMRI(**kwargs)
    else:
        raise NotImplementedError(name)
