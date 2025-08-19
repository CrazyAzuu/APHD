
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import (cosine_beta_schedule,
                     linear_beta_schedule,
                     vp_beta_schedule,
                     extract,
                     Losses)
from utils.utils import Progress, Silent

def apply_condition(seq, cond):
    for key, value in cond.items():
        seq[:, key] = value.clone()
    return seq

class Diffusion_Lower(nn.Module):
    def __init__(self, state_dim, model, horizon=20, beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=False, predict_epsilon=True, w=1, use_cfg=True):
        super(Diffusion_Lower, self).__init__()

        self.state_dim = state_dim
        self.horizon = horizon
        self.max_step = 100
        self.model = model
        self.use_cfg = use_cfg
        self.w = w
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(self.max_step)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.max_step)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(self.max_step)

        self.gamma = [1 for _ in np.linspace(0, 1, horizon)]
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32) # shape (horizon,)
        # betas shape (n_timesteps, 1)
        betas = betas.unsqueeze(1)
        betas = betas * self.gamma # shape (n_timesteps, horizon)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones((1, self.horizon)), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # math: Var[x_{t-1}] = Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        # return x0; from x_t = \sqrt(\bar{\alpha}_t) x0 + \sqrt(1 - \bar{\alpha}_t) \epsilon_t

        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        # caculate mean:u_q = (beta_t * sqrt(\bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)) * x_start + \\
        #               (1 - \bar{\alpha}_{t-1}) * sqrt(\alpha_t) / (1 - \bar{\alpha}_t) * x_t
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, s, r, t):

        if self.use_cfg: 
            epsilon_cond = self.model(x, s, r, t, force_dropout=False, use_dropout=False)
            epsilon_uncond = self.model(x, s, r, t, force_dropout=True, use_dropout=False)
            epsilon = epsilon_uncond + self.w*(epsilon_cond - epsilon_uncond)
        else: 
            epsilon = self.model(x, s, r, t, force_dropout=False, use_dropout=False)

        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, s, r, t):
        b, l, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, s=s, r=r, t=t) # Return mean and logarithmic variance
        noise = torch.randn_like(x) # Generate Gaussian noise
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # x_{t-1} = mean + std * noise
        # where std = sqrt(Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, x, state, reward, cond=None, shape=None, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = apply_condition(x, cond)

        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            x = self.p_sample(x, state, reward, timesteps)

            x = apply_condition(x, cond)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, x, state, reward, cond=None, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x, state, reward, cond, shape, *args, **kwargs)
        return x

    # ------------------------------------------ training ------------------------------------------#
    # The cond here is used to fix the initial position of the trajectory, and is not a conditional factor in the paper
    def q_sample(self, x_start, t, noise=None, cond=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # math:
        # x_{t-1} = \sqrt{1 - \alpha_t} * x_0 + \sqrt{\alpha_t} * \epsilon_t
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, reward, t, cond, weights=1.0):
        
        noise = torch.randn_like(x_start)
        noise[:, 0] = 0.0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, cond=cond)
        if self.use_cfg:
            x_recon = self.model(x_noisy, state, reward, time=t)
        else:
            x_recon = self.model(x_noisy, state, reward, time=t, use_dropout=False)

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            x_recon = apply_condition(x_recon, cond)
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state=None, reward=None, cond=None, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.max_step, (batch_size, ), device=x.device).long()
        return self.p_losses(x, state, reward, t, cond, weights)

    def forward(self, x, state, reward, cond=None, *args, **kwargs):
        return self.sample(x, state, reward, cond, *args, **kwargs)

class Diffusion_Upper(nn.Module):
    def __init__(self, state_dim, model, horizon=4, beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=False, predict_epsilon=True, w=1, use_cfg=True):
        super(Diffusion_Upper, self).__init__()

        self.state_dim = state_dim
        self.horizon = horizon
        self.max_step = 100
        self.model = model
        self.use_cfg = use_cfg
        self.w = w

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(self.max_step)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.max_step)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(self.max_step)

        self.gamma = [1 for _ in np.linspace(0, 1, horizon)]
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32) # shape (horizon,)
        # betas shape (n_timesteps, 1)
        betas = betas.unsqueeze(1)
        betas = betas * self.gamma # shape (n_timesteps, horizon)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones((1, self.horizon)), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # Register the parameters as a buffer of the model, as part of the model, but not as parameters of the model, which means they will not be updated
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        # Calculate posterior mean and variance
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, r):

        if self.use_cfg: 
            epsilon_cond = self.model(x, r, t, force_dropout=False, use_dropout=False)
            epsilon_uncond = self.model(x, r, t, force_dropout=True, use_dropout=False)
            epsilon = epsilon_uncond + self.w*(epsilon_cond - epsilon_uncond)

 
        else: 
            epsilon = self.model(x, r, t, force_dropout=False, use_dropout=False)

        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        # x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, r, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, r):
        b, l, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, r=r)
        
        noise = torch.randn_like(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, x, reward, shape, cond=None, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = apply_condition(x, cond)

        # Initialize pure noise action
        # x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, reward)
            x = apply_condition(x, cond)
            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, x, reward, cond=None, *args, **kwargs):
        batch_size = x.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x, reward, shape, cond, *args, **kwargs)

        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None, cond=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # It's not just adding noise one at a time, but setting the parameters for each time step in advance and executing them once
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, reward, t, cond=None, weights=1.0):
        # Generate random noise
        noise = torch.randn_like(x_start)
        noise[:, 0] = 0.0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, cond=cond)
        if self.use_cfg:
            x_recon = self.model(x_noisy, reward, time=t)
        else:
            x_recon = self.model(x_noisy, reward, time=t, use_dropout=False)

        assert noise.shape == x_recon.shape

        # Default self. redist_epsilon is true, model outputs noise
        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            x_recon = apply_condition(x_recon, cond)
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss
    
    def loss(self, x, reward=None, cond=None, weights=1.0):
        batch_size = len(x)
        # The diffusion time step t follows a uniform distribution
        t = torch.randint(0, self.max_step, (batch_size,), device=x.device).long()
        return self.p_losses(x, reward, t, cond, weights)

    def forward(self, x, reward, cond=None, *args, **kwargs):
        return self.sample(x, reward, cond, *args, **kwargs)

class Uncond_Diffusion_Upper(nn.Module):
    def __init__(self, state_dim, model, horizon=4, beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=False, predict_epsilon=True, w=1, use_cfg=True):
        super(Uncond_Diffusion_Upper, self).__init__()

        self.state_dim = state_dim
        self.horizon = horizon
        self.max_step = 100
        self.model = model
        self.use_cfg = use_cfg
        self.w = w

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(self.max_step)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.max_step)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(self.max_step)

        self.gamma = [1 for _ in np.linspace(0, 1, horizon)]
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32) # shape (horizon,)
        # betas shape (n_timesteps, 1)
        betas = betas.unsqueeze(1)
        betas = betas * self.gamma # shape (n_timesteps, horizon)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones((1, self.horizon)), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t):

        epsilon = self.model(x, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t):
        b, l, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        
        noise = torch.randn_like(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, x, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]

        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, x, *args, **kwargs):
        batch_size = x.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x, shape, *args, **kwargs)

        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t, weights=1.0):
        noise = torch.randn_like(x_start)
        noise[:, 0] = 0.0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, time=t)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss
    
    def loss(self, x, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.max_step, (batch_size,), device=x.device).long()
        return self.p_losses(x, t, weights)

    def forward(self, x, *args, **kwargs):
        return self.sample(x, *args, **kwargs)

class Uncond_Diffusion_Lower(nn.Module):
    def __init__(self, state_dim, model, horizon=20, beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=False, predict_epsilon=True, w=1, use_cfg=True):
        super(Uncond_Diffusion_Lower, self).__init__()

        self.state_dim = state_dim
        self.horizon = horizon
        self.max_step = 100
        self.model = model
        self.use_cfg = use_cfg
        self.w = w
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(self.max_step)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.max_step)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(self.max_step)

        self.gamma = [1 for _ in np.linspace(0, 1, horizon)]
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32) # shape (horizon,)
        # betas shape (n_timesteps, 1)
        betas = betas.unsqueeze(1)
        betas = betas * self.gamma # shape (n_timesteps, horizon)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones((1, self.horizon)), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # math: Var[x_{t-1}] = Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        # return x0; from x_t = \sqrt(\bar{\alpha}_t) x0 + \sqrt(1 - \bar{\alpha}_t) \epsilon_t

        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        # caculate mean:u_q = (beta_t * sqrt(\bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)) * x_start + \\
        #               (1 - \bar{\alpha}_{t-1}) * sqrt(\alpha_t) / (1 - \bar{\alpha}_t) * x_t
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, s, t):
        epsilon = self.model(x, s, t)

        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, s, t):
        b, l, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, s=s, t=t) 
        noise = torch.randn_like(x) 
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # x_{t-1} = mean + std * noise
        # where std = sqrt(Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, x, state, shape=None, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]

        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            x = self.p_sample(x, state, timesteps)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, x, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x, state, shape, *args, **kwargs)
        return x

    # ------------------------------------------ training ------------------------------------------#
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # math:
        # x_{t-1} = \sqrt{1 - \alpha_t} * x_0 + \sqrt{\alpha_t} * \epsilon_t
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        
        noise = torch.randn_like(x_start)
        noise[:, 0] = 0.0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, state, time=t)

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state=None, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.max_step, (batch_size, ), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, x, state, *args, **kwargs):
        return self.sample(x, state, *args, **kwargs)
