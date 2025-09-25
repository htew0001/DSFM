import torch
import torch.nn as nn
import numpy as np
import math
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging
import cv2
from PIL import Image
from scipy.fftpack import idct

def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit':
        from libs.uvit import UViT
        return UViT(**kwargs)
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        return UViT(**kwargs)
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1

def combine_blocks(blocks, height, width, block_sz):
    image = np.zeros((height, width), np.float32)
    index = 0
    for i in range(0, height, block_sz):
        for j in range(0, width, block_sz):
            image[i:i + block_sz, j:j + block_sz] = blocks[index]
            index += 1
    return image

def idct_transform(blocks):
    idct_blocks = []
    for block in blocks:
        idct_block = cv2.idct(block)
        idct_block = idct_block # Shift back
        idct_blocks.append(idct_block)
    return np.array(idct_blocks)

def cont_idct_transform(dct_block):
    """
    2D inverse DCT (Type-III) with orthonormal normalization, using SciPy.
    """
    block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')  # IDCT 2D
    block = block  # Shift back
    return block

def DCT_to_RGB(sample, tokens=0, low_freqs=0, block_sz=0, reverse_order=None, resolution=0, Y_bound=None):
    num_y_blocks = tokens * 4
    num_cb_blocks = tokens
    cb_blocks_per_row = int((resolution / block_sz) / 2)
    Y_blocks_per_row = int(resolution / block_sz)

    assert sample.shape == (tokens, low_freqs*6)
    sample = np.clip(sample, -2, 2)  # clamp into [-1, 1]
    sample = sample.reshape(tokens, 6, low_freqs)  # (tokens, 6, low_freqs)

    # fill up DCT coes
    DCT = np.zeros((tokens, 6, block_sz * block_sz))  # (tokens, 6, 16)
    DCT[:, :, :low_freqs] = sample
    DCT = DCT[..., reverse_order]  # convert the low to high freq order back to 8*8 order

    Y_bound = np.array(Y_bound)
    DCT_Y = DCT[:, :4, :] * Y_bound  # (64, 4, 16)
    DCT_Cb = DCT[:, 4, :] * Y_bound  # (64, 16)
    DCT_Cr = DCT[:, 5, :] * Y_bound  # (64, 16)

    DCT_Cb = DCT_Cb.reshape(num_cb_blocks, block_sz, block_sz)  # (64, 16) --> (64, 4, 4)
    DCT_Cr = DCT_Cr.reshape(num_cb_blocks, block_sz, block_sz)  # (64, 16) --> (64, 4, 4)

    y_blocks = []
    for row in range(cb_blocks_per_row):  # 16 cb/cr blocks, so 4*4 spatial blocks
        tem_ls = []
        for col in range(cb_blocks_per_row):
            ind = row * cb_blocks_per_row + col
            y_blocks.append(DCT_Y[ind, 0, :])
            y_blocks.append(DCT_Y[ind, 1, :])
            tem_ls.append(DCT_Y[ind, 2, :])
            tem_ls.append(DCT_Y[ind, 3, :])
        for ele in tem_ls:
            y_blocks.append(ele)
    DCT_Y = np.array(y_blocks).reshape(num_y_blocks, block_sz, block_sz)  # (256, 4, 4)

    # Apply Inverse DCT on each block
    idct_y_blocks = idct_transform(DCT_Y)
    idct_cb_blocks = idct_transform(DCT_Cb)
    idct_cr_blocks = idct_transform(DCT_Cr)

    # Combine blocks back into images
    height, width = resolution, resolution
    y_reconstructed = combine_blocks(idct_y_blocks, height, width, block_sz)
    cb_reconstructed = combine_blocks(idct_cb_blocks, int(height / 2), int(width / 2), block_sz)
    cr_reconstructed = combine_blocks(idct_cr_blocks, int(height / 2), int(width / 2), block_sz)

    # Upsample Cb and Cr to original size
    cb_upsampled = cv2.resize(cb_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)
    cr_upsampled = cv2.resize(cr_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)

    # Step 5: Convert YCbCr back to RGB
    R = y_reconstructed + 1.402 * (cr_upsampled - 128)
    G = y_reconstructed - 0.344136 * (cb_upsampled - 128) - 0.714136 * (cr_upsampled - 128)
    B = y_reconstructed + 1.772 * (cb_upsampled - 128)

    rgb_reconstructed = np.zeros((height, width, 3))
    rgb_reconstructed[:, :, 0] = np.clip(R, 0, 255)
    rgb_reconstructed[:, :, 1] = np.clip(G, 0, 255)
    rgb_reconstructed[:, :, 2] = np.clip(B, 0, 255)

    # Convert to uint8
    rgb_reconstructed = np.uint8(rgb_reconstructed)  # (h, w, 3), RGB channels

    return rgb_reconstructed

def dct_to_rgb(sample, tokens=0, low_freqs=0, block_sz=0, reverse_order=None, resolution=0, Y_bound=None, use_YCbCr=False):

    dct_rgb = np.zeros([3, resolution * resolution])  # (C, H, W)
    dct_rgb[:, :low_freqs] = sample
    dct_rgb = dct_rgb[:, reverse_order]  # convert the low to high freq order back to 8*8 order

    dct_r = dct_rgb[0].reshape(resolution, resolution)  # (H*W) --> (H, W)
    dct_g = dct_rgb[1].reshape(resolution, resolution)  # (H*W) --> (H, W)
    dct_b = dct_rgb[2].reshape(resolution, resolution)  # (H*W) --> (H, W)

    # Apply Inverse DCT on each block
    R = cont_idct_transform(dct_r)
    G = cont_idct_transform(dct_g)
    B = cont_idct_transform(dct_b)

    # if resolution != downsampled_resolution:
    #     R = cv2.resize(R, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    #     G = cv2.resize(G, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    #     B = cv2.resize(B, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)

    if use_YCbCr:
        Y = R; Cb = G; Cr = B  # Use Y, Cb, Cr directly for RGB reconstruction
        # --- Step 4: Inverse-scale each channel from [-1, 1] back to float-difference YCbCr ---
        Y = Y * 0.5 + 0.5   #  →  [0, 1]
        Cb = Cb * 0.5         #  →  [-0.5, 0.5]
        Cr = Cr * 0.5         #  →  [-0.5, 0.5]
        # --- Step 5: Convert float-difference YCbCr → RGB (still in [0, 1]) ---
        R = Y + 1.402   * Cr
        G = Y - 0.344136 * Cb - 0.714136 * Cr
        B = Y + 1.772   * Cb
    else:
        # --- Step 4: Inverse-scale each channel from [-1, 1] back to [0, 1] ---
        R = R * 0.5 + 0.5
        G = G * 0.5 + 0.5
        B = B * 0.5 + 0.5

    # --- Step 6: Stack and convert back to uint8 [0, 255] ---
    # R = Y; G = Cb; B = Cr  # Use Y, Cb, Cr directly for RGB reconstruction
    img_rgb = np.stack([R, G, B], axis=-1)
    img_rgb = np.clip(img_rgb, 0.0, 1.0)
    img_rgb = (img_rgb * 255.0).astype(np.uint8)  # (H, W, 3), RGB channels

    return img_rgb

def DCTsamples_to_grid_image(samples, tokens=0, low_freqs=0, block_sz=0,
                             reverse_order=None, resolution=0, grid_sz=0, path=None, Y_bound=None):
    samples = samples.detach().cpu().numpy()
    rgb_imgs = []
    for sample in samples:
        rgb_reconstructed = DCT_to_RGB(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)
        # rgb_reconstructed = dct_to_rgb(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)
        rgb_imgs.append(rgb_reconstructed)
    rgb_imgs = np.array(rgb_imgs)
    img_sz = rgb_imgs.shape[1]

    # Fill the grid image with the 36 smaller images
    grid_image = np.zeros((grid_sz * img_sz, grid_sz * img_sz, 3), dtype=np.uint8)
    for i in range(grid_sz):
        for j in range(grid_sz):
            idx = i * grid_sz + j
            if idx < rgb_imgs.shape[0]:
                grid_image[i * img_sz:(i + 1) * img_sz, j * img_sz:(j + 1) * img_sz, :] = rgb_imgs[idx]

    # Convert the NumPy array to an image and save or show it
    final_image = Image.fromarray(grid_image)
    final_image.save(path)


def DCTsample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn,
                  tokens=0, low_freqs=0, reverse_order=None, resolution=0, block_sz=8, Y_bound=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    print(f'using Y_bound {Y_bound} for sampling')

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = sample_fn(mini_batch_size)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        samples = samples.detach().cpu().numpy()
        if accelerator.is_main_process:
            for sample in samples:
                rgb_reconstructed = DCT_to_RGB(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)
                # rgb_reconstructed = dct_to_rgb(sample, tokens, low_freqs, block_sz, reverse_order, resolution, Y_bound)

                cv2.imwrite(os.path.join(path, f"{idx}.jpg"), cv2.cvtColor(rgb_reconstructed, cv2.COLOR_RGB2BGR))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


import matplotlib.pyplot as plt
from pathlib import Path
import einops                                                # already in your deps

@torch.no_grad()
def save_fmri_samples_npy(
    samples: torch.Tensor,           # (N, tokens, C*low_freqs)
    save_dir: str,
    *,
    tokens: int,
    low_freqs: int,
    channels: int,
    block_sz: int,
    reverse_order: np.ndarray,
    resolution: int,
):
    """
    Reconstructs each channel with inverse‑DCT and writes
        save_dir/sample_{n:04d}_ch{c}.npy
    The stored array is float32, **no scaling applied**.
    """
    from pathlib import Path
    import einops, cv2, numpy as np

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    samples = samples.cpu().float().numpy()                   # (N, T, C*lf)

    H_blocks = W_blocks = resolution // block_sz

    for n, sample in enumerate(samples):
        for c in range(channels):
            coeffs = sample[:, c * low_freqs : (c + 1) * low_freqs]

            full16 = np.zeros((tokens, 16), np.float32)
            full16[:, reverse_order[:low_freqs]] = coeffs

            # inverse‑DCT per block
            blocks = full16.reshape(tokens, block_sz, block_sz)
            blocks = einops.rearrange(blocks, "(h w) b1 b2 -> (h w) b1 b2",
                                      h=H_blocks, w=W_blocks)
            img = np.stack([cv2.idct(b) for b in blocks], 0)
            img = einops.rearrange(img,
                                   "(h w) b1 b2 -> (h b1) (w b2)",
                                   h=H_blocks, w=W_blocks).astype(np.float32)

            np.save(f"{save_dir}/sample_{n:04d}_ch{c}.npy", img)


def dct_to_fmri_image(sample,
                 tokens,
                 low_freqs,
                 block_sz,
                 reverse_order,
                 resolution,
                 bound=None,
                 shift=0):
    """
    Reconstruct an n_ch-channel image from its block-DCT coefficients.

    Args:
      sample        : np.ndarray, shape (tokens, low_freqs * n_ch)
      tokens        : int, number of blocks (e.g. 29*29 = 841)
      low_freqs     : int, number of kept DCT coeffs per block (e.g. 16)
      block_sz      : int, block height/width (e.g. 4)
      reverse_order : array-like of length block_sz*block_sz, zigzag → 2D mapping
      resolution    : int, full image height/width (e.g. 116)
      bound         : None or np.ndarray of shape (n_ch, low_freqs) or (n_ch, block_sz*block_sz)
                      per-frequency scale factors
      shift         : float, amount to add after IDCT (e.g. 128 if you subtracted 128 on forward DCT)

    Returns:
      img_rec       : np.ndarray, shape (resolution, resolution, n_ch), dtype=float32
    """
    # infer channel count
    n_ch = sample.shape[1] // low_freqs
    assert sample.shape == (tokens, low_freqs * n_ch)

    # clamp & reshape into (tokens, n_ch, low_freqs)
    sample = np.clip(sample, -2, 2).reshape(tokens, n_ch, low_freqs)

    # build full-size DCT blocks (zeros for the dropped high freqs)
    DCT = np.zeros((tokens, n_ch, block_sz * block_sz), dtype=np.float32)
    DCT[:, :, :low_freqs] = sample
    DCT = DCT[..., reverse_order]  # zigzag → natural 2D order

    # apply per-frequency scaling if provided
    if bound is not None:
        b = np.array(bound, dtype=np.float32)
        DCT = DCT * b[np.newaxis, ...]  # broadcast over tokens

    # IDCT each 2D block
    N = tokens * n_ch
    blocks = DCT.reshape(N, block_sz, block_sz)
    rec_blocks = np.empty_like(blocks, dtype=np.float32)
    for i in range(N):
        tmp = cv2.idct(blocks[i])
        if shift:
            tmp = tmp + shift
        rec_blocks[i] = tmp

    # reshape back → (tokens, n_ch, block_sz, block_sz)
    rec_blocks = rec_blocks.reshape(tokens, n_ch, block_sz, block_sz)

    # re-tile blocks into each channel’s full image
    imgs = []
    for c in range(n_ch):
        img = np.zeros((resolution, resolution), dtype=np.float32)
        idx = 0
        for y in range(0, resolution, block_sz):
            for x in range(0, resolution, block_sz):
                img[y:y+block_sz, x:x+block_sz] = rec_blocks[idx, c]
                idx += 1
        imgs.append(img)

    # stack → (H, W, n_ch)
    return np.stack(imgs, axis=-1)


from pathlib import Path
import numpy as np
from pathlib import Path
from PIL import Image

import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))
from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss

def minmax_normalize(ori_data, fake_data):
    """
    Normalizes a 3D array (Subjects, Timesteps, ROI) using min-max scaling
    computed across subjects and timesteps for each ROI.
    """
    # data_min = np.min(data, axis=(0,1), keepdims=True)
    # data_max = np.max(data, axis=(0,1), keepdims=True)

    data_min = ori_data.min(axis=(0,1), keepdims=True)
    data_max = ori_data.max(axis=(0,1), keepdims=True)
    # reuse on both
    ori_norm  = (ori_data - data_min) / (data_max - data_min + 1e-8)
    fake_norm = (fake_data - data_min) / (data_max - data_min + 1e-8)
    # Add a small constant to avoid division by zero.
    # return (data - data_min) / (data_max - data_min + 1e-8)
    return ori_norm, fake_norm

def DCTfmrisamples_to_grid_image(
        samples,
        save_dir,
        tokens,
        low_freqs,
        block_sz,
        reverse_order,
        resolution,
        embedder,
        device,
        X_outer_train,
        Y_bound=None,
        shift=0,
    ):
    # 1) to NumPy
    if hasattr(samples, 'detach'):
        arr = samples.detach().cpu().numpy()
    else:
        arr = np.array(samples)

    # 2) reconstruct all samples
    recon_list = []
    for sample in arr:
        img = dct_to_fmri_image(
            sample,
            tokens=tokens,
            low_freqs=low_freqs,
            block_sz=block_sz,
            reverse_order=reverse_order,
            resolution=resolution,
            bound=Y_bound,
            shift=shift
        )
        recon_list.append(img)
    all_imgs = np.stack(recon_list, axis=0)  # shape: (N, H, W, n_ch)

    # 3) ensure save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4) save the full batch as .npy
    np.save(save_dir / "all_samples_spectogram.npy", all_imgs)
    print(f"Saved reconstructed batch to {save_dir/'all_samples_spectogram.npy'}")
    # samples shape ((381, 116, 116, 12))

    left_img = all_imgs[:,:,:,:6]
    right_img = all_imgs[:,:,:,6:]

    channel_size = all_imgs.shape[3]//2

    merged = np.concatenate([left_img, right_img], axis=2) #(B, 116, 232, 6)

    merged_out = merged.transpose(0, 3, 1, 2)
    print("sample and reconcat image shape: ",merged_out.shape)

    tensor_out = torch.from_numpy(merged_out).float().to(device)
    rec   = embedder.img_to_ts(tensor_out) #input shape (B, 6,116,232)
    print("reconstructed time series size:", rec.shape)
    rec = rec.detach().cpu().numpy()
    np.save(save_dir / "unnnorm_real_timeseries_all.npy", X_outer_train)
    np.save(save_dir / "unnorm_fake_timeseries_all.npy", rec)

    ori_data, fake_data = minmax_normalize(X_outer_train[:rec.shape[0]], rec)
    np.save(save_dir / "norm_real_timeseries_all.npy", ori_data)
    np.save(save_dir / "norm_fake_timeseries_all.npy", fake_data)

    context_fid_score = []
    iterations = 5

    for i in range(iterations):
        context_fid = Context_FID(ori_data[:], fake_data[:ori_data.shape[0]])
        context_fid_score.append(context_fid)
        logging.info(f"Iter {i}: context-fid = {context_fid:.4f}")

    display_scores(context_fid_score)

    print("compute correlational score")

    def random_choice(size, num_select=100):
        select_idx = np.random.randint(low=0, high=size, size=(num_select,))
        return select_idx

    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(fake_data)

    correlational_score = []
    size = int(x_real.shape[0] / iterations)

    for i in range(iterations):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
        loss = corr.compute(x_fake[fake_idx, :, :])
        correlational_score.append(loss.item())
        logging.info(f"Iter {i}: cross-correlation = {loss.item():.6f}")

    display_scores(correlational_score)

    # ori_data, fake_data: (n_subjects, T, R)
    n, T, R = ori_data.shape

    # Flatten each subject’s time×ROI matrix into a vector of length T*R
    X = ori_data.reshape(n, T*R)
    Y = fake_data.reshape(n, T*R)

    # Now compute RBF-MMD (as before)
    from sklearn.metrics.pairwise import rbf_kernel
    def mmd_rbf(X, Y, gamma):
        XX = rbf_kernel(X, X, gamma=gamma)
        YY = rbf_kernel(Y, Y, gamma=gamma)
        XY = rbf_kernel(X, Y, gamma=gamma)
        return XX.mean() + YY.mean() - 2*XY.mean()

    # Choose gamma via the median heuristic:
    from scipy.spatial.distance import pdist
    sq_dists = pdist(X, 'sqeuclidean')
    sigma2 = np.median(sq_dists)
    gamma = 1.0 / (2 * sigma2)

    mmd_value = mmd_rbf(X, Y, gamma)
    # print(f"Subject-wise RBF-MMD: {mmd_value:.6f}")
    noise = np.random.normal(size=X.shape)
    noise_mmd = mmd_rbf(X, noise, gamma=gamma)
    logging.info(f"RBF-MMD-realvsfake = {mmd_value:.6f}")
    logging.info(f"RBF-MMD-realvsnoise = {noise_mmd:.6f}")

    # 5) save first subject’s channel-0 as PNG
    # first = all_imgs[0]            # shape (H, W, n_ch)
    # ch0 = first[..., 0]            # pick channel 0
    # img0 = Image.fromarray(np.clip(ch0, 0, 255).astype(np.uint8))
    # img0.save(save_dir / "subject0_ch0.png")
    # print(f"Saved first subject channel-0 to {save_dir/'subject0_ch0.png'}")