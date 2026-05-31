import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse

import numpy as np 
from abc import ABC, abstractmethod
# from utils.utils_data import MinMaxArgs

def MinMaxScaler(data, axis=(0,2,3), return_scalers=False):
    """Min Max normalizer.

    Args:|
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, axis=axis, keepdims=True)
    max = np.max(data, axis=axis, keepdims=True)
    numerator = data - np.min(data, axis=axis, keepdims=True)
    denominator = np.max(data, axis=axis, keepdims=True) - np.min(data, axis=axis, keepdims=True)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data

def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass

# ------------------------------------------------------------------
# Embedder
# ------------------------------------------------------------------
class DWTEmbedder(TsImgEmbedder):
    """
    1-D Discrete Wavelet Transform (DWT) embedder with two output
    layouts:
        • layout="scale_feature"  → (B, S, F, T)  (scale, time)
        • layout="feature_scale" → (B, F, S, T)  (feature, time)
    """
    def __init__(
        self,
        device: torch.device,
        seq_len: int,
        J: int = 2,
        wave: str = "db1",
        mode: str = "symmetric",
        layout: str = "scale_feature",   # or "feature_scale"
        norm_mode: str = "channel"
    ):
        super().__init__(device, seq_len)
        assert layout in {"scale_feature", "feature_scale"}
        self.J = J
        self.wave = wave
        self.mode = mode
        self.layout = layout
        # self._axis = axis

        # cached normalisation bounds (single pair – real-valued only)
        self.min_coeff, self.max_coeff = None, None

        self.dwt  = DWT1DForward(J=J, wave=wave, mode=mode).to(device)
        self.idwt = DWT1DInverse(wave=wave, mode=mode).to(device)

        if norm_mode == "channel":
            print("channel")
            self._axis = (0,2,3)
        elif norm_mode == "pixel":
            print("pixel")
            self._axis = (0,1)
        elif norm_mode == "channel_pixel":
            print("channel and pixel")
            self._axis = 0
        elif norm_mode == "subject":
            print("subject")
            self._axis = (1,2,3)
        else:
            raise ValueError("norm_mode must be 'channel' or 'pixel'")

    # ------------------------------------------------------------------
    # caching
    # ------------------------------------------------------------------
    def cache_min_max_params(self, train_data: torch.Tensor):
        """
        Compute and cache global min / max of DWT coefficients
        (call once before training).
        Args
        ----
        train_data : (B, T, F)  raw time-series batch
        """
        coeff_img = self.dwt_transform(train_data).cpu()
        coeff_img, xmin, xmax = MinMaxScaler(coeff_img.numpy(), self._axis
        , return_scalers=True)
        print("intialize dwt embedder")
        # print("minimum value coefficient: ", xmin)
        # print("maximum value coefficient: ", xmax)
        self.min_coeff = torch.tensor(xmin, dtype=torch.float32)
        self.max_coeff = torch.tensor(xmax, dtype=torch.float32)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward transform + normalise to [-1,1].
        signal : (B, T, F)
        returns: (B, C, H, W)  where (C,H,W) = layout-chosen (S,F,T) or (F,S,T)
        """
        assert self.min_coeff is not None, "Run cache_min_max_params() first"
        coeff = self.dwt_transform(signal)

        # Min-Max scaling to [-1, 1]
        coeff = (MinMaxArgs(
            coeff,
            self.min_coeff.to(coeff.device),
            self.max_coeff.to(coeff.device)) - 0.5
        ) * 2.0
        return coeff

    @torch.no_grad()
    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        """
        Inverse path: denormalise + inverse DWT → (B, T, F)
        """
        min_c = self.min_coeff.to(img.device)
        max_c = self.max_coeff.to(img.device)

        coeff = ((img / 2.0) + 0.5) * (max_c - min_c) + min_c
        ts = self.inv_dwt_transform(coeff)
        return ts  # already (B, T, F)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def dwt_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw DWT → wavelet image (no scaling).
        x : (B, T, F)
        returns:
            (B, S, F, T)  if layout=="scale_feature"
            (B, F, S, T)  if layout=="feature_scale"
        """
        B, T, F = x.shape
        # DWT expects (B, C, T) → treat features as channels
        x = x.permute(0, 2, 1)  # (B, F, T)

        # dwt = DWT1DForward(J=self.J, wave=self.wave, mode=self.mode)
        yl, yh = self.dwt(x)                          # yl: LL_J, yh: [LH1..LH_J]

        def upsample(band, factor):
            return band.repeat_interleave(factor, dim=-1)[..., :T]

        rows = []
        for k, band in enumerate(yh, 1):
            rows.append(upsample(band, 2 ** k))  # detail
        rows.append(upsample(yl, 2 ** self.J))   # approximation

        wimg = torch.stack(rows, dim=1)          # (B, S, F, T)
        if self.layout == "feature_scale":
            wimg = wimg.permute(0, 2, 1, 3)      # (B, F, S, T)
        return wimg

    def inv_dwt_transform(self, wimg: torch.Tensor) -> torch.Tensor:
        """
        Invert wavelet image back to (B, T, F)
        """
        if self.layout == "feature_scale":
            wimg = wimg.permute(0, 2, 1, 3)      # → (B, S, F, T)

        B, S, F, T = wimg.shape
        J = S - 1

        # approximation (LP)
        yl = wimg[:, -1, :, :: 2 ** J]

        # detail bands
        yh = []
        for k in range(1, J + 1):
            band_up = wimg[:, k - 1, :, :]
            band_ds = band_up[:, :, :: 2 ** k]
            yh.append(band_ds)

        # idwt = DWT1DInverse(wave=self.wave, mode=self.mode)
        x_rec = self.idwt((yl, yh))                   # (B, F, T)
        return x_rec.permute(0, 2, 1)            # (B, T, F)