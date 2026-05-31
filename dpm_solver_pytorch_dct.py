import torch
import torch.nn.functional as F
import math
import numpy as np


class NoiseScheduleVP_DCT:
    def __init__(self, eigvals, schedule='linear', norm_factor=0.5, SNR_scale=1.0):
        if schedule != 'linear':
            raise ValueError(f"Unsupported noise schedule {schedule}. Only 'linear' is supported.")

        print(f"Using DCT mode for DPM-Solver sampling")
        self.norm_factor = norm_factor
        print(f"Using norm factor: {self.norm_factor} for DPM-Solver sampling")

        if eigvals is None:
            raise ValueError("eigvals (Laplacian spectrum) must be provided for DCT-mode VPSDE.")

        self.eigvals = eigvals
        self.normed_eigvals = norm_factor * eigvals  # shape: [D]
        self.schedule = schedule
        self.T = 0.9946  # consistent with cosine schedule default in DPM-Solver
        self.SNR_scale = SNR_scale
        print(f"Using SNR scale: {self.SNR_scale} for DPM-Solver sampling")

    def marginal_log_mean_coeff(self, t):
        """
        Computes log(alpha_t) for each DCT mode using frequency-aware decay and optional SNR correction.
        """
        # Compute raw decay: shape [B, D] or [D]
        if t.dim() == 0:
            decay = t * self.normed_eigvals  # [D]
        else:
            decay = t[:, None] * self.normed_eigvals[None, :]  # [B, D]
        # exp(-decay) term
        exp_decay = torch.exp(-decay)
        # SNR correction term (per-mode)
        snr_term = torch.log1p((self.SNR_scale - 1) * exp_decay) - math.log(self.SNR_scale)
        # Apply full formula
        log_alpha_t = -decay - snr_term
        # Clamp to prevent instability (especially at t=0)
        log_alpha_t = torch.clamp(log_alpha_t, max=-1e-5)
        print("log_alpha_t values:", log_alpha_t)
        return log_alpha_t

    def marginal_std(self, t):
        log_alpha_t = self.marginal_log_mean_coeff(t)
        exp_term = torch.exp(2. * log_alpha_t)
        var_term = torch.clamp(1.0 - exp_term, min=1e-12)
        sigma_t = torch.sqrt(var_term)
        return sigma_t

    def marginal_lambda(self, t):
        log_alpha_t = self.marginal_log_mean_coeff(t)
        exp_term = torch.exp(2. * log_alpha_t)
        exp_term = torch.clamp(exp_term, max=1. - 1e-6)  # prevent log(<=0)
        log_sigma_t = 0.5 * torch.log(1. - exp_term)
        lambda_t = log_alpha_t - log_sigma_t
        return lambda_t

    def inverse_lambda(self, lamb, method='logexp'):
        """
        Invert lambda_t to time t for SNR-corrected DCT-mode VPSDE.
        Returns a single t per batch by reducing across frequency modes.
        """
        normed_eigvals = self.normed_eigvals  # shape: [D]
        SNR_scale = self.SNR_scale
        eps = 1e-5

        if method == 'approx':
            # Linear approximation: lambda_t ≈ -t * eigval - SNR_term(t)
            # Ignore SNR term for this crude approximation
            t = -lamb / (normed_eigvals[None, :] + eps)

        elif method == 'logexp':
            # SNR-corrected inverse of:
            # lambda_t = -t * eigval - log(1 + (SNR - 1) * exp(-t * eigval)) + log(SNR)
            #
            # Forward formula:
            # log_alpha = -t * eigval - log(1 + (SNR - 1) * exp(-t * eigval)) + log(SNR)
            # ⇒ exp(2 * lambda_t) = (exp(-2 * t * eigval) * SNR^2) / (1 + (SNR - 1) * exp(-t * eigval))^2
            #
            # We now solve:
            # log1p(exp(2 * lambda)) / (2 * eigval)  = t   (approximate)
            #
            # lamb = torch.clamp(lamb, max=20)
            exp2lamb = torch.exp(2.0 * lamb)
            numerator = SNR_scale * (1. + exp2lamb)
            denominator = exp2lamb
            inside_log = numerator / denominator + 1.0 - SNR_scale
            log_term = torch.log(inside_log + eps)  # small eps to prevent log(0)
            t = log_term / (normed_eigvals[None, :] + eps)

        else:
            raise ValueError(f"Unknown inverse_lambda method: {method}")

        # Postprocessing: clamp nan/inf, and return a single t per sample (batch)
        # t = torch.nan_to_num(t, nan=1.0, posinf=1.0, neginf=0.0)
        # t = t.max(dim=-1).values  # choose max t across DCT modes to guarantee safety
        t = t.mean(dim=-1) # average across modes for a single time step per sample
        print("t values:", t)
        return t

    

class DPM_Solver_DCT:
    def __init__(self, model_fn, noise_schedule):
        """Construct a DPM-Solver.

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input
                (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        """
        self.model_fn = model_fn
        self.noise_schedule = noise_schedule

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            # print("lambda_T values:", lambda_T)
            # print("lambda_0 values:", lambda_0)
            # logSNR_steps = torch.linspace(lambda_T, lambda_0, N + 1).to(device)
            # lambda_T and lambda_0 are shape [K]
            logSNR_steps = torch.linspace(0, 1, N + 1).to(device)[:, None] * (lambda_0 - lambda_T)[None, :] + lambda_T[None, :]  # [N+1, K]
            # print("logSNR_steps shape:", logSNR_steps.shape)
            time_steps = self.noise_schedule.inverse_lambda(logSNR_steps)
            # print("time_steps shape:", time_steps.shape)
            return time_steps
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t = torch.linspace(t_0, t_T, 10000000).to(device)
            quadratic_t = torch.sqrt(t)
            quadratic_steps = torch.linspace(quadratic_t[0], quadratic_t[-1], N + 1).to(device)
            return torch.flip(
                torch.cat([t[torch.searchsorted(quadratic_t, quadratic_steps)[:-1]], t_T * torch.ones((1,)).to(device)],
                          dim=0), dims=[0])
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_time_steps_for_dpm_solver_fast(self, t_T, t_0, steps, device):
        """
        Compute the intermediate time steps and the order of each step for sampling by DPM-Solver-fast.

        We recommend DPM-Solver-fast for fast sampling of DPMs. Given a fixed number of function evaluations by `steps`,
        the sampling procedure by DPM-Solver-fast is:
            - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
            - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
            - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            steps: A `int`. The total number of function evaluations (NFE).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
            timesteps: A pytorch tensor of the time steps, with the shape of (K + 1,).
        """
        K = steps // 3 + 1
        if steps % 3 == 0:
            orders = [3, ] * (K - 2) + [2, 1]
        elif steps % 3 == 1:
            orders = [3, ] * (K - 1) + [1]
        else:
            orders = [3, ] * (K - 1) + [2]
        timesteps = self.get_time_steps('logSNR', t_T, t_0, K, device)
        return orders, timesteps

    def dpm_solver_first_update(self, x, s, t, return_noise=False):
        """
        A single step for DPM-Solver-1.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            return_noise: A `bool`. If true, also return the predicted noise at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)

        phi_1 = torch.expm1(h)

        noise_s = self.model_fn(x, s)
        x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[:, None, :] * x
                - (sigma_t * phi_1)[:, None, :] * noise_s
        )
        if return_noise:
            return x_t, {'noise_s': noise_s}
        else:
            return x_t

    def dpm_solver_second_update(self, x, s, t, r1=0.5, noise_s=None, return_noise=False):
        """
        A single step for DPM-Solver-2.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the second-order solver. We recommend the default setting `0.5`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            return_noise: A `bool`. If true, also return the predicted noise at time `s` and `s1` (the intermediate time).
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(
            s1), ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_t = ns.marginal_std(s1), ns.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)

        if noise_s is None:
            noise_s = self.model_fn(x, s)
        x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s)[:, None, :] * x
                - (sigma_s1 * phi_11)[:, None, :] * noise_s
        )
        noise_s1 = self.model_fn(x_s1, s1)
        x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[:, None, :] * x
                - (sigma_t * phi_1)[:, None, :] * noise_s
                - (0.5 / r1) * (sigma_t * phi_1)[:, None, :] * (noise_s1 - noise_s)
        )
        if return_noise:
            return x_t, {'noise_s': noise_s, 'noise_s1': noise_s1}
        else:
            return x_t

    def dpm_solver_third_update(self, x, s, t, r1=1. / 3., r2=2. / 3., noise_s=None, noise_s1=None, noise_s2=None):
        """
        A single step for DPM-Solver-3.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `1 / 3`.
            r2: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `2 / 3`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            noise_s1: A pytorch tensor. The predicted noise at time `s1` (the intermediate time given by `r1`).
                If `noise_s1` is None, we compute the predicted noise by `s1`; otherwise we directly use it.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(
            s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
        phi_2 = torch.expm1(h) / h - 1.

        if noise_s is None:
            noise_s = self.model_fn(x, s)
        if noise_s1 is None:
            # print("x shape:", x.shape)
            # print("log_alpha_s1 shape:", log_alpha_s1.shape)
            # print("log_alpha_s shape:", log_alpha_s.shape)
            # print("sigma_s1 shape:", sigma_s1.shape)
            # print("phi_11 shape:", phi_11.shape)
            # print("noise_s shape:", noise_s.shape)
            x_s1 = (
                    torch.exp(log_alpha_s1 - log_alpha_s)[:, None, :] * x
                    - (sigma_s1 * phi_11)[:, None, :] * noise_s
            )
            noise_s1 = self.model_fn(x_s1, s1)
        if noise_s2 is None:
            x_s2 = (
                    torch.exp(log_alpha_s2 - log_alpha_s)[:, None, :] * x
                    - (sigma_s2 * phi_12)[:, None, :] * noise_s
                    - r2 / r1 * (sigma_s2 * phi_22)[:, None, :] * (noise_s1 - noise_s)
            )
            noise_s2 = self.model_fn(x_s2, s2)
        x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[:, None, :] * x
                - (sigma_t * phi_1)[:, None, :] * noise_s
                - (1. / r2) * (sigma_t * phi_2)[:, None, :] * (noise_s2 - noise_s)
        )
        return x_t

    def dpm_solver_update(self, x, s, t, order):
        """
        A single step for DPM-Solver of the given order `order`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t)
        elif order == 2:
            return self.dpm_solver_second_update(x, s, t)
        elif order == 3:
            return self.dpm_solver_third_update(x, s, t)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5):
        """
        The adaptive step size solver based on DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((x.shape[0],)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_noise=True)
            higher_update = lambda x, s, t, **kwargs: self.dpm_solver_second_update(x, s, t, r1=r1, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.dpm_solver_second_update(x, s, t, r1=r1, return_noise=True)
            higher_update = lambda x, s, t, **kwargs: self.dpm_solver_third_update(x, s, t, r1=r1, r2=r2, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def sample(self, x, steps=10, eps=1e-4, T=None, order=3, skip_type='logSNR',
               adaptive_step_size=False, fast_version=True, atol=0.0078, rtol=0.05,
               ):
        """
        Compute the sample at time `eps` by DPM-Solver, given the initial `x` at time `T`.

        We support the following algorithms:

            - Adaptive step size DPM-Solver (i.e. DPM-Solver-12 and DPM-Solver-23)

            - Fixed order DPM-Solver (i.e. DPM-Solver-1, DPM-Solver-2 and DPM-Solver-3).

            - Fast version of DPM-Solver (i.e. DPM-Solver-fast), which uses uniform logSNR steps and combine
                different orders of DPM-Solver.

        **We recommend DPM-Solver-fast for both fast sampling in few steps (<=20) and fast convergence in many steps (50 to 100).**

        Choosing the algorithms:

            - If `adaptive_step_size` is True:
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                If `order`=2, we use DPM-Solver-12 which combines DPM-Solver-1 and DPM-Solver-2.
                If `order`=3, we use DPM-Solver-23 which combines DPM-Solver-2 and DPM-Solver-3.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.

            - If `adaptive_step_size` is False and `fast_version` is True:
                We ignore `order` and use DPM-Solver-fast with number of function evaluations (NFE) = `steps`.
                We ignore `skip_type` and use uniform logSNR steps for DPM-Solver-fast.
                Given a fixed NFE=`steps`, the sampling procedure by DPM-Solver-fast is:
                    - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                    - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                    - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

            - If `adaptive_step_size` is False and `fast_version` is False:
                We use DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
                We support three types of `skip_type`:
                    - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                    - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                    - 'time_quadratic': quadratic time for the time steps. (Used in DDIM.)

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `T` (a sample from the normal distribution).
            steps: A `int`. The total number of function evaluations (NFE).
            eps: A `float`. The ending time of the sampling.
                We recommend `eps`=1e-3 when `steps` <= 15; and `eps`=1e-4 when `steps` > 15.
            T: A `float`. The starting time of the sampling. Default is `None`.
                If `T` is None, we use self.noise_schedule.T.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. Default is 'logSNR'.
            adaptive_step_size: A `bool`. If true, use the adaptive step size DPM-Solver.
            fast_version: A `bool`. If true, use DPM-Solver-fast (recommended).
            atol: A `float`. The absolute tolerance of the adaptive step size solver.
            rtol: A `float`. The relative tolerance of the adaptive step size solver.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        t_0 = eps
        t_T = self.noise_schedule.T if T is None else T
        device = x.device
        if adaptive_step_size:
            with torch.no_grad():
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol)
        else:
            if fast_version:
                orders, timesteps = self.get_time_steps_for_dpm_solver_fast(t_T=t_T, t_0=t_0, steps=steps,
                                                                            device=device)
            else:
                N_steps = steps // order
                orders = [order, ] * N_steps
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=N_steps, device=device)
            with torch.no_grad():
                for i, order in enumerate(orders):
                    vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(
                        device) * timesteps[i + 1]
                    x = self.dpm_solver_update(x, vec_s, vec_t, order)
        return x