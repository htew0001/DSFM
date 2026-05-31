import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
import numpy as np
import math
from tqdm import tqdm
from typing import Tuple, Optional

def get_sde(name, **kwargs):
    if name == 'vpsde':
        return VPSDE(**kwargs)
    elif name == 'vpsde_cosine':
        return VPSDECosine(**kwargs)
    else:
        raise NotImplementedError


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)


class SDE(object):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def drift(self, x, t):
        raise NotImplementedError

    def diffusion(self, t):
        raise NotImplementedError

    def cum_beta(self, t):  # the variance of xt|x0
        raise NotImplementedError

    def cum_alpha(self, t):
        raise NotImplementedError

    def snr(self, t):  # signal noise ratio
        raise NotImplementedError

    def nsr(self, t):  # noise signal ratio
        raise NotImplementedError

    def marginal_prob(self, x0, t):  # the mean and std of q(xt|x0)
        alpha = self.cum_alpha(t)
        beta = self.cum_beta(t)
        mean = stp(alpha ** 0.5, x0)  # E[xt|x0]
        std = beta ** 0.5  # Cov[xt|x0] ** 0.5
        return mean, std

    def sample(self, x0, t_init=0):  # sample from q(xn|x0), where n is uniform
        t = torch.rand(x0.shape[0], device=x0.device) * (1. - t_init) + t_init
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn_like(x0)
        xt = mean + stp(std, eps)
        return t, eps, xt


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, SNR_scale=1.0):
        # 0 <= t <= 1
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.SNR_scale = SNR_scale
        print(f"using SNR_scale: {self.SNR_scale} for training")

    def drift(self, x, t):
        return -0.5 * stp(self.squared_diffusion(t), x)

    def diffusion(self, t):
        return self.squared_diffusion(t) ** 0.5

    def squared_diffusion(self, t):  # beta(t)
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)

        # SNR adjustment
        a = self.beta_0
        b = self.beta_1 - self.beta_0
        exp_term = torch.exp(-1 * (a * t + 0.5 * b * (t**2)))
        SNR_term_numerator = (self.SNR_scale - 1) * exp_term * (-1 * a - (b * t))
        SNR_term_denominator = 1 + (self.SNR_scale - 1) * exp_term

        return beta_t + (SNR_term_numerator / SNR_term_denominator)

    def squared_diffusion_integral(self, s, t):  # \int_s^t beta(tau) d tau
        integral_beta = self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5

        # SNR adjustment
        a = self.beta_0
        b = self.beta_1 - self.beta_0
        exp_term = torch.exp(-1 * (a * (t-s) + 0.5 * b * (t**2 - s**2)))
        SNR_term = torch.log(1 + (self.SNR_scale - 1) * exp_term) - math.log(self.SNR_scale)

        # # verify SNR term
        # intgral_beta_t = self.beta_0 * t + (self.beta_1 - self.beta_0) * (t ** 2) * 0.5
        # cum_alpha = (- intgral_beta_t).exp()  # alpha_bar
        # cum_beta = 1 - cum_alpha
        # org_SNR = cum_alpha / cum_beta
        #
        # exp_term = torch.exp(-1 * (a * t + 0.5 * b * (t ** 2)))
        # SNR_term = torch.log(1 + (self.SNR_scaleup - 1) * exp_term) - math.log(self.SNR_scaleup)
        # intgral_beta_t = intgral_beta_t + SNR_term
        # cum_alpha = (- intgral_beta_t).exp()  # alpha_bar
        # cum_beta = 1 - cum_alpha
        # new_SNR = cum_alpha / cum_beta
        # print(f"SNR scale up by {new_SNR / org_SNR}")

        return integral_beta + SNR_term

    def skip_beta(self, s, t):  # beta_{t|s}, Cov[xt|xs]=beta_{t|s} I
        return 1. - self.skip_alpha(s, t)

    def skip_alpha(self, s, t):  # alpha_{t|s}, E[xt|xs]=alpha_{t|s}**0.5 xs
        x = -self.squared_diffusion_integral(s, t)
        return x.exp()

    def cum_beta(self, t):
        return self.skip_beta(0, t)

    def cum_alpha(self, t):
        return self.skip_alpha(0, t)

    def nsr(self, t):
        return self.squared_diffusion_integral(0, t).expm1()

    def snr(self, t):
        return 1. / self.nsr(t)

    def __str__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'

    def __repr__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'


class VPSDECosine(SDE):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def __init__(self, s=0.008):
        self.s = s
        self.F = lambda t: torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        self.F0 = math.cos(s / (1 + s) * math.pi / 2) ** 2

    def drift(self, x, t):
        ft = - torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi / 2
        return stp(ft, x)

    def diffusion(self, t):
        return (torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi) ** 0.5

    def cum_beta(self, t):  # the variance of xt|x0
        return 1 - self.cum_alpha(t)

    def cum_alpha(self, t):
        return self.F(t) / self.F0

    def snr(self, t):  # signal noise ratio
        Ft = self.F(t)
        return Ft / (self.F0 - Ft)

    def nsr(self, t):  # noise signal ratio
        Ft = self.F(t)
        return self.F0 / Ft - 1

    def __str__(self):
        return 'vpsde_cosine'

    def __repr__(self):
        return 'vpsde_cosine'

class VPSDECosineDCT(nn.Module):
    """Heat‑blurred cosine forward SDE in DCT space (global β).

    Each DCT coefficient *k* follows the linear SDE
        d z_{t,k} = f_{t,k}(t) z_{t,k} dt + g_{t,k}(t) dW_{t,k},
    whose marginal solution is assumed to be
        z_{t,k} = α(t) ω_{t,k} x_{0,k} + β(t) ε_k ,
    with global VP cosine pair α(t)=cos(½πt), β(t)=sin(½πt) **without**
    including the blur factor in β.  All analytic schedules and their
    derivatives are provided; numerical differentiation is never used.
    """

    # ------------------------------------------------------------------
    # ctor
    # ------------------------------------------------------------------
    def __init__(
        self,
        resolution: int,
        patch_sz: int,
        sigma_blur_max: float = 20.0,
        d_min: float = 1e-3,
        blur_beta: bool = False,
        low2high_order: Optional[np.ndarray] = None,
        low_freqs: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.resolution = resolution
        self.patch_sz = patch_sz
        self.sigma_blur_max = float(sigma_blur_max)
        self.d_min = float(d_min)
        self.blur_beta = bool(blur_beta)
        self.low2high_order = low2high_order
        self.low_freqs = low_freqs

        lamb = self._compute_dct_eigenvalues()  # (K,)
        self.register_buffer("lamb", lamb, persistent=False)

    # ------------------------------------------------------------------
    # helpers – DCT Laplacian eigenvalues λ_k (flattened zig‑zag)
    # ------------------------------------------------------------------
    def _compute_dct_eigenvalues(self) -> torch.Tensor:
        # freqs = (np.pi / self.resolution) * np.arange(self.resolution, dtype=np.float32)
        freqs = (np.pi / self.patch_sz) * np.arange(self.patch_sz, dtype=np.float32)
        lamb_2d = freqs[:, None] ** 2 + freqs[None, :] ** 2  # (L,L)
        lamb_flat = lamb_2d.flatten()
        if self.low2high_order is not None:
            lamb_flat = lamb_flat[self.low2high_order]
        if self.low_freqs is not None:
            lamb_flat = lamb_flat[: self.low_freqs]
        return torch.from_numpy(lamb_flat)  # (K,)

    # ------------------------------------------------------------------
    # global cosine pair α, β and derivatives (shape B,)
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_and_dot(t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        phi = 0.5 * math.pi * t
        alpha = torch.cos(phi)
        beta = torch.sin(phi)
        alpha_dot = -0.5 * math.pi * beta
        beta_dot = 0.5 * math.pi * alpha
        return alpha, beta, alpha_dot, beta_dot

    # ------------------------------------------------------------------
    # blur factor ω_{t,k} and derivative (shape B×K)
    # ------------------------------------------------------------------
    def _blur_and_dot(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = 0.5 * math.pi * t
        sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
        sigmaB = self.sigma_blur_max * sin_phi ** 2
        sigmaB_dot = self.sigma_blur_max * math.pi * sin_phi * cos_phi
        tau, tau_dot = 0.5 * sigmaB ** 2, sigmaB * sigmaB_dot

        lamb = self.lamb.to(t.device)
        exp_term = torch.exp(-tau[:, None] * lamb)  # (B,K)
        omega = (1.0 - self.d_min) * exp_term + self.d_min
        omega_dot = -(1.0 - self.d_min) * lamb * tau_dot[:, None] * exp_term
        return omega, omega_dot

    # ------------------------------------------------------------------
    # per‑mode schedules α_{t,k}, β_{t,k}  and derivatives
    # ------------------------------------------------------------------
    def _alpha_beta_and_dot(self, t: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # global cosine pair (B,)
        alpha_g, beta_g, alpha_g_dot, beta_g_dot = self._cosine_and_dot(t)
        # blur (B,K)
        omega, omega_dot = self._blur_and_dot(t)

        # α and α̇  include blur
        alpha = alpha_g[:, None] * omega                    # (B,K)
        alpha_dot = alpha_g_dot[:, None] * omega + alpha_g[:, None] * omega_dot

        if self.blur_beta:
            # β, β̇ include blur (variance-preserving per-mode)
            beta = torch.sqrt(1.0 - alpha ** 2)                 # (B,K)
            beta_dot = -(alpha * alpha_dot) / beta              # (B,K)
        else:
            # β, β̇ stay global (cosine-only or heat-blurred cosine schedule)
            beta = beta_g[:, None].expand_as(alpha)             # (B,K)
            beta_dot = beta_g_dot[:, None].expand_as(alpha)     # (B,K)

        return alpha, beta, alpha_dot, beta_dot

    # ------------------------------------------------------------------
    # cumulative α², β² and derivatives
    # ------------------------------------------------------------------
    def cum_alpha(self, t: torch.Tensor) -> torch.Tensor:  # (B,K)
        return self._alpha_beta_and_dot(t)[0] ** 2

    def cum_beta(self, t: torch.Tensor) -> torch.Tensor:   # (B,K)
        return self._alpha_beta_and_dot(t)[1] ** 2

    def cum_alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        alpha, _, alpha_dot, _ = self._alpha_beta_and_dot(t)
        return 2.0 * alpha * alpha_dot

    def cum_beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        _, beta, _, beta_dot = self._alpha_beta_and_dot(t)
        return 2.0 * beta * beta_dot

    # ------------------------------------------------------------------
    # marginal mean / std and derivatives
    # ------------------------------------------------------------------
    def marginal_prob(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha, beta, _, _ = self._alpha_beta_and_dot(t)
        return alpha, beta

    def marginal_prob_dot(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, alpha_dot, beta_dot = self._alpha_beta_and_dot(t)
        return alpha_dot, beta_dot

    # ------------------------------------------------------------------
    # drift f and diffusion g  (B,K)
    # ------------------------------------------------------------------
    def drift(self, t: torch.Tensor) -> torch.Tensor:
        alpha, _, alpha_dot, _ = self._alpha_beta_and_dot(t)
        return alpha_dot / alpha

    def diffusion(self, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        alpha, beta, alpha_dot, beta_dot = self._alpha_beta_and_dot(t)
        f = alpha_dot / alpha.clamp(min=eps)
        g2 = 2.0 * beta * (beta_dot - f * beta)
        return torch.sqrt(torch.clamp(g2, min=eps))

    # ------------------------------------------------------------------
    # scalar / tensor product helper (broadcast-safe)
    # ------------------------------------------------------------------
    @staticmethod
    def _stp(s: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Scalar tensor product, broadcast-safe."""
        #HH: change from 6 to 12
        s = s.repeat(1, 12)
        while s.dim() < ts.dim():
            s = s.unsqueeze(1)
        return s * ts

    # ------------------------------------------------------------------
    # sampling utilities
    # ------------------------------------------------------------------
    def sample(self, x0: torch.Tensor, *, t_min: float = 1e-6,
               seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        t_max = 1.0 - t_min          # ensure t_init < 0.5
        t = torch.rand(x0.shape[0], device=x0.device) * (t_max - t_min) + t_min
        mean, std = self.marginal_prob(t)
        mean_dot, std_dot = self.marginal_prob_dot(t)
        eps = torch.randn_like(x0)
        # print("mean.shape", mean.shape, "x0.shape", x0.shape, "std.shape", std.shape, "eps.shape", eps.shape)
        xt = self._stp(mean, x0) + self._stp(std, eps)
        dxdt = self._stp(mean_dot, x0) + self._stp(std_dot, eps)
        return t, eps, xt, dxdt

    def sample_seq(self, x0: torch.Tensor, *, num_steps: int = 100, t_min: float = 1e-6,
                   seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        t_max = 1.0 - t_min
        t = torch.rand(x0.shape[0], device=x0.device) * (t_max - t_min) + t_min
        mean, std = self.marginal_prob(t)                      # (L,K)
        eps = torch.randn((num_steps, *x0.shape[1:]), device=x0.device)
        xt = self._stp(mean, x0) + self._stp(std, eps)
        return t, eps, xt

    # ------------------------------------------------------------------
    def __str__(self) -> str:  # noqa: DunderStr
        return "vpsde_cosine_dct"

    __repr__ = __str__

class ScoreModel(object):
    r"""
        The forward process is q(x_[0,T])
    """

    def __init__(self, nnet: nn.Module, pred: str, sde: SDE, T=1, patch_sz=4):
        assert T == 1
        self.nnet = nnet
        self.pred = pred
        self.sde = sde
        self.T = T
        self.patch_sz = patch_sz
        print(f'ScoreModel with pred={pred}, sde={sde}, T={T}, patch_sz={patch_sz}')

    def predict(self, xt, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = duplicate(t, xt.size(0))
        return self.nnet(xt, t * 999, **kwargs)  # follow SDE

    def noise_pred(self, xt, t, **kwargs):
        pred = self.predict(xt, t, **kwargs)
        if self.pred == 'noise_pred':
            noise_pred = pred
        elif self.pred == 'x0_pred':
            noise_pred = - stp(self.sde.snr(t).sqrt(), pred) + stp(self.sde.cum_beta(t).rsqrt(), xt)
        else:
            raise NotImplementedError
        return noise_pred

    def x0_pred(self, xt, t, **kwargs):
        pred = self.predict(xt, t, **kwargs)
        if self.pred == 'noise_pred':
            x0_pred = stp(self.sde.cum_alpha(t).rsqrt(), xt) - stp(self.sde.nsr(t).sqrt(), pred)
        elif self.pred == 'x0_pred':
            x0_pred = pred
        else:
            raise NotImplementedError
        return x0_pred
    
    def v_pred(self, xt, t, **kwargs):
        if self.pred == 'v_pred':
            # xt = self.patchify(xt)
            v_pred = self.predict(xt, t, **kwargs)
            # v_pred = self.unpatchify(v_pred)
        else:
            raise NotImplementedError
        return v_pred

    def patchify(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, 3, H*W] DCT image to patch tokens [B, N, 3*P*P].
        """
        B, C, HW = sample.shape
        assert C == 3, "Expected 3 channels (RGB)."
        H = W = int(math.isqrt(HW))
        assert H * W == HW, "H*W must be a perfect square."
        P = self.patch_sz
        assert H % P == 0 and W % P == 0, "Patch size must divide H and W."

        sample = sample.view(B, C, H, W)  # restore spatial
        patches = F.unfold(sample, kernel_size=P, stride=P)  # [B, 3*P*P, N]
        patches = patches.permute(0, 2, 1).contiguous()  # [B, N, 3*P*P]
        return patches

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct [B, 3, H*W] from patch tokens [B, N, 3*P*P].
        """
        B, N, patch_dim = patches.shape
        C = 3
        P = self.patch_sz
        assert patch_dim == C * P * P, f"Patch dim mismatch: got {patch_dim}, expected {3*P*P}"
        grid_size = int(math.isqrt(N))
        assert grid_size * grid_size == N, "N must be a perfect square."
        H = W = grid_size * P

        patches = patches.permute(0, 2, 1).contiguous()  # [B, 3*P*P, N]
        sample = F.fold(patches, output_size=(H, W), kernel_size=P, stride=P)  # [B, 3, H, W]
        sample = sample.view(B, C, H * W)  # return to flattened [B, 3, H*W]
        return sample

    def score(self, xt, t, **kwargs):
        cum_beta = self.sde.cum_beta(t)
        noise_pred = self.noise_pred(xt, t, **kwargs)
        return stp(-cum_beta.rsqrt(), noise_pred)

class ReverseSDE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt + g(t) dw
    """
    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, t, **kwargs)
        return drift - stp(diffusion ** 2, score)

    def diffusion(self, t):
        return self.sde.diffusion(t)


class ODE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt
    """

    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, t, **kwargs)
        return drift - 0.5 * stp(diffusion ** 2, score)

    def diffusion(self, t):
        return 0


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})

@ torch.no_grad()
def euler_maruyama(rsde, x_init, sample_steps, eps=1e-3, T=1, trace=None, verbose=False, **kwargs):
    r"""
    The Euler Maruyama sampler for reverse SDE / ODE
    See `Score-Based Generative Modeling through Stochastic Differential Equations`
    """
    assert isinstance(rsde, ReverseSDE) or isinstance(rsde, ODE)
    print(f"euler_maruyama with sample_steps={sample_steps}")
    timesteps = np.append(0., np.linspace(eps, T, sample_steps))
    timesteps = torch.tensor(timesteps).to(x_init)
    x = x_init
    if trace is not None:
        trace.append(x)
    for s, t in tqdm(list(zip(timesteps, timesteps[1:]))[::-1], disable=not verbose, desc='euler_maruyama'):
        drift = rsde.drift(x, t, **kwargs)
        diffusion = rsde.diffusion(t)
        dt = s - t
        mean = x + drift * dt
        sigma = diffusion * (-dt).sqrt()
        x = mean + stp(sigma, torch.randn_like(x)) if s != 0 else mean
        if trace is not None:
            trace.append(x)
        statistics = dict(s=s, t=t, sigma=sigma.item())
        logging.debug(dct2str(statistics))
    return x


def LSimple(score_model: ScoreModel, x0, pred='noise_pred', reweight=None, **kwargs):
    if pred == 'noise_pred':
        t, noise, xt = score_model.sde.sample(x0)
        noise_pred = score_model.noise_pred(xt, t, **kwargs)
        loss = (noise - noise_pred).pow(2)  # (batch, tokens, dim)
        loss = loss * reweight  # loss re-weighting
        return loss.flatten(start_dim=1).mean(dim=-1)

    elif pred == 'x0_pred':
        t, noise, xt = score_model.sde.sample(x0)
        x0_pred = score_model.x0_pred(xt, t, **kwargs)
        loss = (x0 - x0_pred).pow(2)
        loss = loss * reweight  # loss re-weighting
        return loss.flatten(start_dim=1).mean(dim=-1)

    elif pred == 'v_pred':
        t, noise, xt, dxdt = score_model.sde.sample(x0)
        v_pred = score_model.v_pred(xt, t, **kwargs)
        loss = (dxdt - v_pred).pow(2)  # (B, C, K)
        loss = loss * reweight  # loss re-weighting
        return loss.flatten(start_dim=1).mean(dim=-1)

    else:
        raise NotImplementedError(pred)
