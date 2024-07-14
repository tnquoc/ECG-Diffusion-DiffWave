import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.u_net import SimpleUnet
# from models.diffwave import DiffWave
from models.diffwave_condition import DiffWave

from utils.diffusion_utils import get_index_from_list, forward_diffusion_sample


class DiffusionModel(nn.Module):
    def __init__(self, beta_1=1e-4, beta_2=0.02, T=100):
        super(DiffusionModel, self).__init__()
        # self.decoder = SimpleUnet()
        # self.decoder = DiffWave()
        self.decoder = DiffWave(conditional=True)
        self.T = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.register_buffer(
            "betas", torch.linspace(self.beta_1, self.beta_2, steps=self.T).double()
        )
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def forward(self, x, t, label=None):
        x_noisy, noise = forward_diffusion_sample(x, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, x.device)
        x_noisy = x_noisy.type(torch.FloatTensor).to(x.device)
        noise_pred = self.decoder(x_noisy, t, label)
        return noise, noise_pred

    @torch.no_grad()
    def sample_timestep(self, x, t, label=None):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
                # x - betas_t * self.decoder(x, t) / sqrt_one_minus_alphas_cumprod_t
                x - betas_t * self.decoder(x, t, label) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

