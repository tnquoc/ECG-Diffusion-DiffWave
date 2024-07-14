import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = x * torch.sigmoid(x)
        x = self.projection2(x)
        x = x * torch.sigmoid(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation):
        '''
    :param n_mels: inplanes of conv1x1 for spectrogram condition
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram condition
    '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(512, residual_channels)

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        y = self.dilated_conv(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


# class DiffWave(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.params = params
#         self.input_projection = Conv1d(1, params.residual_channels, 1)
#         self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
#         if self.params.unconditional:  # use unconditional model
#             self.spectrogram_upsampler = None
#         else:
#             self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)
#
#         self.residual_layers = nn.ModuleList([
#             ResidualBlock(params.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
#                           uncond=params.unconditional)
#             for i in range(params.residual_layers)
#         ])
#         self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
#         self.output_projection = Conv1d(params.residual_channels, 1, 1)
#         nn.init.zeros_(self.output_projection.weight)
#
#     def forward(self, audio, diffusion_step, spectrogram=None):
#         assert (spectrogram is None and self.spectrogram_upsampler is None) or \
#                (spectrogram is not None and self.spectrogram_upsampler is not None)
#         x = audio.unsqueeze(1)
#         x = self.input_projection(x)
#         x = F.relu(x)
#
#         diffusion_step = self.diffusion_embedding(diffusion_step)
#         if self.spectrogram_upsampler:  # use condition model
#             spectrogram = self.spectrogram_upsampler(spectrogram)
#
#         skip = None
#         for layer in self.residual_layers:
#             x, skip_connection = layer(x, diffusion_step, spectrogram)
#             skip = skip_connection if skip is None else skip_connection + skip
#
#         x = skip / sqrt(len(self.residual_layers))
#         x = self.skip_projection(x)
#         x = F.relu(x)
#         x = self.output_projection(x)
#         return x


class DiffWave(nn.Module):
    def __init__(self, unconditional=False):
        super().__init__()
        # self.params = params
        self.input_projection = Conv1d(1, 64, 1)
        self.diffusion_embedding = DiffusionEmbedding(200)
        if unconditional:  # use unconditional model
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(80)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(80, 64, 2 ** (i % 10))
            for i in range(30)
        ])
        self.skip_projection = Conv1d(64, 64, 1)
        self.output_projection = Conv1d(64, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step):
        # x = audio.unsqueeze(1)
        x = self.input_projection(audio)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
