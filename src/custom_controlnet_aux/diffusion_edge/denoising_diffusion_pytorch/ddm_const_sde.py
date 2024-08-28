import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
import math
import torch.nn.functional as F
# import torchvision.transforms.functional as F2
from .utils import default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from tqdm.auto import tqdm
from einops import rearrange, reduce
from functools import partial
from collections import namedtuple
from random import random, randint, sample, choice
from .encoder_decoder import DiagonalGaussianDistribution
import random
from custom_controlnet_aux.diffusion_edge.taming.modules.losses.vqperceptual import *

# gaussian diffusion trainer class
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DDPM(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l2',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        original_elbo_weight=0.,
        ddim_sampling_eta = 1.,
        clip_x_start=True,
        train_sample=-1,
        input_keys=['image'],
        start_dist='normal',
        sample_type='ddim',
        perceptual_weight=1.,
        use_l1=False,
        **kwargs
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        cfg = kwargs.pop("cfg", None)
        super().__init__(**kwargs)
        # assert not (type(self) == DDPM and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.input_keys = input_keys
        self.cfg = cfg
        self.eps = cfg.get('eps', 1e-4) if cfg is not None else 1e-4
        self.weighting_loss = cfg.get("weighting_loss", False) if cfg is not None else False
        if self.weighting_loss:
            print('#### WEIGHTING LOSS ####')

        self.clip_x_start = clip_x_start
        self.image_size = image_size
        self.train_sample = train_sample
        self.objective = objective
        self.start_dist = start_dist
        assert start_dist in ['normal', 'uniform']

        assert objective in {'pred_noise', 'pred_x0', 'pred_v', 'pred_delta', 'pred_KC'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps, s=1e-4)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        # betas[0] = 2e-3 * betas[0]
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.time_range = list(range(self.num_timesteps + 1))
        self.loss_type = loss_type
        self.original_elbo_weight = original_elbo_weight

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        # assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)
        assert not torch.isnan(self.p2_loss_weight).all()
        if self.objective == "pred_noise":
            lvlb_weights = self.betas ** 2 / (
                        2 * (self.posterior_variance+1e-5) * alphas * (1 - self.alphas_cumprod))
        elif self.objective == "pred_x0":
            lvlb_weights = 0.5 * torch.sqrt(alphas_cumprod) / (2. * 1 - alphas_cumprod)
        elif self.objective == "pred_delta":
            lvlb_weights = 0.5 * torch.sqrt(alphas_cumprod) / (2. * 1 - alphas_cumprod)
        elif self.objective == "pred_KC":
            lvlb_weights = 0.5 * torch.sqrt(alphas_cumprod) / (2. * 1 - alphas_cumprod)
        elif self.objective == "pred_v":
            lvlb_weights = 0.5 * torch.sqrt(alphas_cumprod) / (2. * 1 - alphas_cumprod)
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
        self.use_l1 = use_l1

        self.perceptual_weight = perceptual_weight
        if self.perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False, use_ema=False):
        sd = torch.load(path, map_location="cpu")
        if 'ema' in list(sd.keys()) and use_ema:
            sd = sd['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]    # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
        else:
            if "model" in list(sd.keys()):
                sd = sd["model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def p_sample(self, x, mask, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, mask=mask, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, mask, up_scale=1, unnormalize=True):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        img = F.interpolate(img, scale_factor=up_scale, mode='bilinear', align_corners=True)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, mask, t, self_cond)
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, mask, up_scale=1, unnormalize=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        img = F.interpolate(img, scale_factor=up_scale, mode='bilinear', align_corners=True)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total=len(time_pairs)):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, mask, self_cond)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img


    @torch.no_grad()
    def interpolate(self, x1, x2, mask, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, mask, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        assert 'image' in self.input_keys;
        if len(self.input_keys) > len(batch.keys()):
            x, *_ = batch.values()
        else:
            x = batch.values()
        return x

    def training_step(self, batch):
        z, *_ = self.get_input(batch)
        cond = batch['cond'] if 'cond' in batch else None
        loss, loss_dict = self(z, cond)
        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # continuous time, t in [0, 1]
        # t = []
        # for _ in range(x.shape[0]):
        #     if self.train_sample <= 0:
        #         t.append(torch.tensor(sample(self.time_range, 2), device=x.device).long())
        #     else:
        #         sl = choice(self.time_range)
        #         sl_range = list(range(sl - self.train_sample, sl + self.train_sample))
        #         sl_range = list(set(sl_range) & set(self.time_range))
        #         sl_range.pop(sl_range.index(sl))
        #         sl2 = choice(sl_range)
        #         t.append(torch.tensor([sl, sl2], device=x.device).long())
        # t = torch.stack(t, dim=0)
        # t = torch.randint(0, self.num_timesteps+1, (x.shape[0],), device=x.device).long()
        eps = self.eps  # smallest time step
        # t = torch.rand((x.shape[0],), device=x.device) * (self.num_timesteps / eps)
        # t = t.round() * eps
        # t[t < eps] = eps
        t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        return self.p_losses(x, t, *args, **kwargs)

    def q_sample2(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        _, nt = t.shape
        param_x = self.sqrt_alphas_cumprod.repeat(b, 1).gather(-1, t)  # (b, nt)
        x = x_start.expand(nt, b, c, h, w).transpose(1, 0) * param_x.reshape(b, nt, 1, 1, 1).repeat(1, 1, c, h, w)
        param_noise = self.sqrt_one_minus_alphas_cumprod.repeat(b, 1).gather(-1, t)
        n = noise.expand(nt, b, c, h, w).transpose(1, 0) * param_noise.reshape(b, nt, 1, 1, 1).repeat(1, 1, c, h, w)
        return x + n  # (b, nt, c, h, w)

    def q_sample3(self, x_start, t, C):
        b, c, h, w = x_start.shape
        _, nt = t.shape
        # K_ = K.unsqueeze(1).repeat(1, nt, 1, 1, 1)
        C_ = C.unsqueeze(1).repeat(1, nt, 1, 1, 1)
        x_noisy = x_start.expand(nt, b, c, h, w).transpose(1, 0) + \
                  + C_ * t.reshape(b, nt, 1, 1, 1).repeat(1, 1, c, h, w) / self.num_timesteps
        return x_noisy  # (b, nt, c, h, w)

    # def q_sample(self, x_start, t, C):
    #     x_noisy = x_start + C * t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1))) / self.num_timesteps
    #     return x_noisy
    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + torch.sqrt(time) * noise
        return x_noisy

    def q_sample2(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C / 2 * time ** 2 + torch.sqrt(time) * noise
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - torch.sqrt(time) * noise
        return x0

    def pred_x0_from_xt2(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C / 2 * time ** 2 - torch.sqrt(time) * noise
        return x0

    def pred_xtms_from_xt(self, xt, noise, C, t, s):
        # noise = noise / noise.std(dim=[1, 2, 3]).reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt + C * (time-s) - C * time - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time-s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def pred_xtms_from_xt2(self, xt, noise, C, t, s):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt + C / 2 * (time-s) ** 2 - C / 2 * time ** 2 - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time-s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def WCE_loss(self, prediction, labelf, beta=1.1):
        label = labelf.long()
        mask = labelf.clone()

        num_positive = torch.sum(label == 1).float()
        num_negative = torch.sum(label == 0).float()

        mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[label == 0] = beta * num_positive / (num_positive + num_negative)
        mask[label == 2] = 0
        cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='sum')

        return cost

    def Dice_Loss(self, pred, label):
        # pred = torch.sigmoid(pred)
        smooth = 1
        pred_flat = pred.view(-1)
        label_flat = label.view(-1)

        intersecion = pred_flat * label_flat
        unionsection = pred_flat.pow(2).sum() + label_flat.pow(2).sum() + smooth
        loss = unionsection / (2 * intersecion.sum() + smooth)
        loss = loss.sum()
        return loss

    def p_losses(self, x_start, t, *args, **kwargs):
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        # K = -1. * torch.ones_like(x_start)
        # C = noise - x_start  # t = 1000 / 1000
        C = -1 * x_start             # U(t) = Ct, U(1) = -x0
        # C = -2 * x_start               # U(t) = 1/2 * C * t**2, U(1) = 1/2 * C = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, 2, c, h, w)
        C_pred, noise_pred = self.model(x_noisy, t, **kwargs)
        # C_pred = C_pred / torch.sqrt(t)
        # noise_pred = noise_pred / torch.sqrt(1 - t)
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t)       # x_rec:(B, 1, H, W)
        loss_dict = {}
        prefix = 'train'

        # elif self.objective == 'pred_KC':
        #     target1 = C
        #     target2 = noise
        #     target3 = x_start

        target1 = C
        target2 = noise
        target3 = x_start

        loss_simple = 0.
        loss_vlb = 0.
        # use l1 + l2
        if self.weighting_loss:
            simple_weight1 = 2*torch.exp(1-t)
            simple_weight2 = torch.exp(torch.sqrt(t))
            if self.cfg.model_name == 'ncsnpp9':
                simple_weight1 = (t + 1) / t.sqrt()
                simple_weight2 = (2 - t).sqrt() / (1 - t + self.eps).sqrt()
        else:
            simple_weight1 = 1
            simple_weight2 = 1

        loss_simple += simple_weight1 * self.get_loss(C_pred, target1, mean=False).mean([1, 2, 3]) + \
                       simple_weight2 * self.get_loss(noise_pred, target2, mean=False).mean([1, 2, 3])
        if self.use_l1:
            loss_simple += simple_weight1 * (C_pred - target1).abs().mean([1, 2, 3]) + \
                           simple_weight2 * (noise_pred - target2).abs().mean([1, 2, 3])
            loss_simple = loss_simple / 2
        # rec_weight = (1 - t.reshape(C.shape[0], 1)) ** 2
        rec_weight = 1 - t.reshape(C.shape[0], 1)           # (B, 1)
        loss_simple = loss_simple.mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple})

        # loss_vlb += torch.abs(x_rec - target3).mean([1, 2, 3]) * rec_weight: (B, 1)
        loss_vlb += self.Dice_Loss(x_rec, target3)
        loss_vlb = loss_vlb.mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + loss_vlb
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, denoise=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        return self.sample_fn((batch_size, channels, image_size[0], image_size[1]),
                              up_scale=up_scale, unnormalize=True, cond=cond, denoise=denoise)

    @torch.no_grad()
    def sample_fn(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], \
                                                                             self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # times = torch.linspace(-1, total_timesteps, steps=self.sampling_timesteps + 1).int()
        # times = list(reversed(times.int().tolist()))
        # time_pairs = list(zip(times[:-1], times[1:]))
        # time_steps = torch.tensor([0.25, 0.15, 0.1, 0.1, 0.1, 0.09, 0.075, 0.06, 0.045, 0.03])
        step = 1. / self.sampling_timesteps
        # time_steps = torch.tensor([0.1]).repeat(10)
        time_steps = torch.tensor([step]).repeat(self.sampling_timesteps)
        if denoise:
            eps = self.eps
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - eps]), torch.tensor([eps])), dim=0)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        img = F.interpolate(img, scale_factor=up_scale, mode='bilinear', align_corners=True)
        # K = -1 * torch.ones_like(img)
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time)
            # C, noise = pred.chunk(2, dim=1)
            C, noise = pred[:2]
            # correct C
            x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            if self.clip_x_start:
                x0.clamp_(-1., 1.)
                # C.clamp_(-2., 2.)
            C = -1 * x0
            img = self.pred_xtms_from_xt(img, noise, C, cur_time, s)
            # img = self.pred_xtms_from_xt2(img, noise, C, cur_time, s)
            cur_time = cur_time - s
        img.clamp_(-1., 1.)
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img



class LatentDiffusion(DDPM):
    def __init__(self,
                 auto_encoder,
                 scale_factor=1.0,
                 scale_by_std=True,
                 scale_by_softsign=False,
                 input_keys=['image'],
                 sample_type='ddim',
                 num_timesteps_cond=1,
                 train_sample=-1,
                 default_scale=False,
                 *args,
                 **kwargs
                 ):
        self.scale_by_std = scale_by_std
        self.scale_by_softsign = scale_by_softsign
        self.default_scale = default_scale
        self.num_timesteps_cond = num_timesteps_cond
        self.train_sample = train_sample
        self.perceptual_weight = 0
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        super().__init__(*args, **kwargs)
        assert self.num_timesteps_cond <= self.num_timesteps
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        if self.scale_by_softsign:
            self.scale_by_std = False
            print('### USING SOFTSIGN RESCALING')
        assert (self.scale_by_std and self.scale_by_softsign) is False;

        self.init_first_stage(auto_encoder)
        # self.instantiate_cond_stage(cond_stage_config)
        self.input_keys = input_keys
        self.clip_denoised = False
        assert sample_type in ['p_loop', 'ddim', 'dpm', 'transformer'] ###  'dpm' is not availible now, suggestion 'ddim'
        self.sample_type = sample_type

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)

    def init_first_stage(self, first_stage_model):
        self.first_stage_model = first_stage_model.eval()
        # self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    '''
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    '''

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # return self.scale_factor * z.detach() + self.scale_bias
        return z.detach()

    @torch.no_grad()
    def on_train_batch_start(self, batch):
        # only for the first batch
        if self.scale_by_std and (not self.scale_by_softsign):
            if not self.default_scale:
                assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
                # set rescale weight to 1./std of encodings
                print("### USING STD-RESCALING ###")
                x, *_ = batch.values()
                encoder_posterior = self.first_stage_model.encode(x)
                z = self.get_first_stage_encoding(encoder_posterior)
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())
                print(f"setting self.scale_factor to {self.scale_factor}")
                # print("### USING STD-RESCALING ###")
            else:
                print(f'### USING DEFAULT SCALE {self.scale_factor}')
        else:
            print(f'### USING SOFTSIGN SCALE !')

    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        assert 'image' in self.input_keys;
        # if len(self.input_keys) > len(batch.keys()):
        #     x, cond, *_ = batch.values()
        # else:
        #     x, cond = batch.values()
        x = batch['image']
        cond = batch['cond'] if 'cond' in batch else None
        z = self.first_stage_model.encode(x)
        # print('zzzz', z.shape)
        z = self.get_first_stage_encoding(z)
        out = [z, cond, x]
        if return_first_stage_outputs:
            xrec = self.first_stage_model.decode(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(cond)
        return out

    def training_step(self, batch):
        z, c, *_ = self.get_input(batch)
        # print(_[0].shape)
        if self.scale_by_softsign:
            z = F.softsign(z)
        elif self.scale_by_std:
            z = self.scale_factor * z
        # print('grad', self.scale_bias.grad)
        loss, loss_dict = self(z, c, edge=_[0])
        return loss, loss_dict

    def q_sample3(self, x_start, t, K, C):
        b, c, h, w = x_start.shape
        _, nt = t.shape
        K_ = K.unsqueeze(1).repeat(1, nt, 1, 1, 1)
        C_ = C.unsqueeze(1).repeat(1, nt, 1, 1, 1)
        x_noisy = x_start.expand(nt, b, c, h, w).transpose(1, 0) + K_ / 2 * (t.reshape(b, nt, 1, 1, 1).repeat(1, 1, c, h, w) / self.num_timesteps) ** 2 \
            + C_ * t.reshape(b, nt, 1, 1, 1).repeat(1, 1, c, h, w) / self.num_timesteps
        return x_noisy  # (b, nt, c, h, w)

    def pred_xtms_from_xt(self, xt, noise, C, t, s):
        # noise = noise / noise.std(dim=[1, 2, 3]).reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt - C * s - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time-s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def WCE_loss(self, prediction, labelf, beta=1.1):
        label = labelf.long()
        mask = labelf.clone()

        num_positive = torch.sum(label == 1).float()
        num_negative = torch.sum(label == 0).float()

        mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[label == 0] = beta * num_positive / (num_positive + num_negative)
        mask[label == 2] = 0
        cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='sum')

        return cost

    def Dice_Loss(self, pred, label):
        # pred = torch.sigmoid(pred)
        B = pred.shape[0]
        smooth = 1
        pred_flat = pred.view(B, -1)
        label_flat = label.view(B, -1)

        intersecion = pred_flat * label_flat
        unionsection = pred_flat.pow(2).sum(dim=-1) + label_flat.pow(2).sum(dim=-1) + smooth
        loss = unionsection / (2 * intersecion.sum(dim=-1) + smooth)
        loss = loss.reshape(B, 1)
        return loss

    def p_losses(self, x_start, t, *args, **kwargs):
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        # K = -1. * torch.ones_like(x_start)
        # C = noise - x_start  # t = 1000 / 1000
        C = -1 * x_start  # U(t) = Ct, U(1) = -x0
        # C = -2 * x_start               # U(t) = 1/2 * C * t**2, U(1) = 1/2 * C = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, 2, c, h, w)
        if self.cfg.model_name == 'cond_unet8':
            C_pred, noise_pred, (e1, e2) = self.model(x_noisy, t, *args, **kwargs)
        if self.cfg.model_name == 'cond_unet13':
            C_pred, noise_pred, aux_C = self.model(x_noisy, t, *args, **kwargs)
        else:
            C_pred, noise_pred = self.model(x_noisy, t, *args, **kwargs)
        # C_pred = C_pred / torch.sqrt(t)
        # noise_pred = noise_pred / torch.sqrt(1 - t)
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t)  # x_rec:(B, C, H, W)
        loss_dict = {}
        prefix = 'train'

        # elif self.objective == 'pred_KC':
        #     target1 = C
        #     target2 = noise
        #     target3 = x_start

        target1 = C
        target2 = noise
        target3 = x_start

        loss_simple = 0.
        loss_vlb = 0.

        simple_weight1 = (t + 1) / t.sqrt()
        simple_weight2 = (2 - t).sqrt() / (1 - t + self.eps).sqrt()

        # if self.weighting_loss:
        #     simple_weight1 = 2 * torch.exp(1 - t)
        #     simple_weight2 = torch.exp(torch.sqrt(t))
        #     if self.cfg.model_name == 'ncsnpp9':
        #         simple_weight1 = (t + 1) / t.sqrt()
        #         simple_weight2 = (2 - t).sqrt() / (1 - t + self.eps).sqrt()
        # else:
        #     simple_weight1 = 1
        #     simple_weight2 = 1

        loss_simple += simple_weight1 * self.get_loss(C_pred, target1, mean=False).mean([1, 2, 3]) + \
                       simple_weight2 * self.get_loss(noise_pred, target2, mean=False).mean([1, 2, 3])

        # loss_simple += self.Dice_Loss(C_pred, target1) * simple_weight1

        if self.use_l1:
            loss_simple += simple_weight1 * (C_pred - target1).abs().mean([1, 2, 3]) + \
                           simple_weight2 * (noise_pred - target2).abs().mean([1, 2, 3])
            loss_simple = loss_simple / 2

        if self.cfg.model_name == 'cond_unet8':
            loss_simple += 0.05*(self.Dice_Loss(e1, (kwargs['edge'] + 1)/2) + self.Dice_Loss(e2, (kwargs['edge'] + 1)/2))
        elif self.cfg.model_name == 'cond_unet13':
            loss_simple += 0.5 * (simple_weight1 * self.get_loss(aux_C, target1, mean=False).mean([1, 2, 3]) + \
                                  simple_weight1 * (aux_C - target1).abs().mean([1, 2, 3]))

        rec_weight = (1 - t.reshape(C.shape[0], 1)) ** 2
        # rec_weight = 1 - t.reshape(C.shape[0], 1)  # (B, 1)
        loss_simple = loss_simple.mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple})

        loss_vlb += torch.abs(x_rec - target3).mean([1, 2, 3]) * rec_weight # : (B, 1)
        # loss_vlb += self.Dice_Loss(x_rec, target3) * rec_weight

        # loss_vlb = loss_vlb
        loss_vlb = loss_vlb.mean()

        if self.cfg.get('use_disloss', False):
            with torch.no_grad():
                edge_rec = self.first_stage_model.decode(x_rec / self.scale_factor)
                edge_rec = unnormalize_to_zero_to_one(edge_rec)
                edge_rec = torch.clamp(edge_rec, min=0., max=1.) # B, 1, 320, 320
            loss_tmp = self.cross_entropy_loss_RCF(edge_rec, (kwargs['edge'] + 1)/2) * rec_weight  # B, 1
            loss_ce = SpecifyGradient.apply(x_rec, loss_tmp.mean())
            # print(loss_ce.shape)
            # print(loss_vlb.shape)
            loss_vlb += loss_ce.mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + loss_vlb
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def cross_entropy_loss_RCF(self, prediction, labelf, beta=1.1):
        # label = labelf.long()
        label = labelf
        mask = labelf.clone()

        num_positive = torch.sum(label == 1).float()
        num_negative = torch.sum(label == 0).float()

        mask_temp = (label > 0) & (label <= 0.3)
        mask[mask_temp] = 0.

        mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[label == 0] = beta * num_positive / (num_positive + num_negative)

        # mask[label == 2] = 0
        cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='none')
        return cost.mean([1, 2, 3])

    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True):
        # image_size, channels = self.image_size, self.channels
        channels = self.channels
        image_size = cond.shape[-2:]
        if cond is not None:
            batch_size = cond.shape[0]
        down_ratio = self.first_stage_model.down_ratio
        if self.cfg.model_name == 'cond_unet8' or self.cfg.model_name == 'cond_unet13':
            z, aux_out = self.sample_fn((batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio),
                               up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise)
        else:
            z = self.sample_fn((batch_size, channels, image_size[0]//down_ratio, image_size[1]//down_ratio),
                               up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise)
            aux_out = None

        if self.scale_by_std:
            z = 1. / self.scale_factor * z.detach()
            if self.cfg.model_name == 'cond_unet13':
                aux_out = 1. / self.scale_factor * aux_out.detach()
        elif self.scale_by_softsign:
            z = z / (1 - z.abs())
            z = z.detach()
        #print(z.shape)
        x_rec = self.first_stage_model.decode(z)
        x_rec = unnormalize_to_zero_to_one(x_rec)
        x_rec = torch.clamp(x_rec, min=0., max=1.)
        if self.cfg.model_name == 'cond_unet13':
            aux_out = self.first_stage_model.decode(aux_out)
            aux_out = unnormalize_to_zero_to_one(aux_out)
            aux_out = torch.clamp(aux_out, min=0., max=1.)
        if mask is not None:
            x_rec = mask * unnormalize_to_zero_to_one(cond) + (1 - mask) * x_rec
        if aux_out is not None:
            return x_rec, aux_out
        return x_rec

    @torch.no_grad()
    def sample_fn(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], \
            self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # times = torch.linspace(-1, total_timesteps, steps=self.sampling_timesteps + 1).int()
        # times = list(reversed(times.int().tolist()))
        # time_pairs = list(zip(times[:-1], times[1:]))
        # time_steps = torch.tensor([0.25, 0.15, 0.1, 0.1, 0.1, 0.09, 0.075, 0.06, 0.045, 0.03])
        step = 1. / self.sampling_timesteps
        # time_steps = torch.tensor([0.1]).repeat(10)
        time_steps = torch.tensor([step]).repeat(self.sampling_timesteps)
        if denoise:
            eps = self.eps
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - eps]), torch.tensor([eps])), dim=0)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        img = F.interpolate(img, scale_factor=up_scale, mode='bilinear', align_corners=True)
        img_aux = F.interpolate(img.clone(), scale_factor=up_scale, mode='bilinear', align_corners=True)
        # img_aux = img.clone()
        # K = -1 * torch.ones_like(img)
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time)
            # C, noise = pred.chunk(2, dim=1)
            C, noise = pred[:2]
            if self.cfg.model_name == 'cond_unet8' or self.cfg.model_name == 'cond_unet13':
                aux_out = pred[-1]
            else:
                aux_out = None
            # if self.scale_by_softsign:
            #     # correct the C for softsign
            #     x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            #     x0 = torch.clamp(x0, min=-0.987654321, max=0.987654321)
            #     C = -x0
            # correct C
            x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            C = -1 * x0
            img = self.pred_xtms_from_xt(img, noise, C, cur_time, s)
            # if self.cfg.model_name == 'cond_unet13' and i == len(time_steps) - 2:
            #     img_aux = img
            # if self.cfg.model_name == 'cond_unet13' and i in [len(time_steps)-2, len(time_steps)-1]:
            #     x0_aux = self.pred_x0_from_xt(img_aux, noise, aux_out, cur_time)
            #     C_aux = -1 * x0_aux
            #     img_aux = self.pred_xtms_from_xt(img_aux, noise, C_aux, cur_time, s)
            if self.cfg.model_name == 'cond_unet13':
                for _ in range(1):
                    x0_aux = self.pred_x0_from_xt(img_aux, noise, aux_out, cur_time)
                    C_aux = -1 * x0_aux
                    img_aux = self.pred_xtms_from_xt(img_aux, noise, C_aux, cur_time, s)
            cur_time = cur_time - s
        if self.scale_by_softsign:
            img.clamp_(-0.987654321, 0.987654321)
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        if self.cfg.model_name == 'cond_unet13':
            aux_out = img_aux
        if aux_out is not None:
            return img, aux_out
        return img

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

if __name__ == "__main__":
    ddconfig = {'double_z': True,
                'z_channels': 4,
                'resolution': (240, 960),
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 2, 4, 4],  # num_down = len(ch_mult)-1
                'num_res_blocks': 2,
                'attn_resolutions': [],
                'dropout': 0.0}
    lossconfig = {'disc_start': 50001,
                  'kl_weight': 0.000001,
                  'disc_weight': 0.5}
    from encoder_decoder import AutoencoderKL
    auto_encoder = AutoencoderKL(ddconfig, lossconfig, embed_dim=4,
                                 )
    from mask_cond_unet import Unet
    unet = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=4, cond_in_dim=1,)
    ldm = LatentDiffusion(auto_encoder=auto_encoder, model=unet, image_size=ddconfig['resolution'])
    image = torch.rand(1, 3, 128, 128)
    mask = torch.rand(1, 1, 128, 128)
    input = {'image': image, 'cond': mask}
    time = torch.tensor([1])
    with torch.no_grad():
        y = ldm.training_step(input)
    pass