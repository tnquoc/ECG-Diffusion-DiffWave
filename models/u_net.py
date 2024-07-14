import torch
from torch import nn
import math


class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=((3 - 1) * 20) // 2,
                      dilation=20)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=((3 - 1) * 40) // 2,
                      dilation=40)
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=((3 - 1) * 80) // 2,
                      dilation=80)
        )
        self.transform = nn.Conv1d(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x_cat = torch.cat([x1, x2, x3, x4], 1)
        x_res = self.transform(x_cat)

        return self.relu(x + x_res)


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.up = up
        if up:
            self.time_mlp = nn.Linear(time_emb_dim, 2 * in_channels)
            self.block = InceptionResidualBlock(2 * in_channels)
            self.transform = nn.ConvTranspose1d(2 * in_channels, out_channels, 4, 2, 1)
        else:
            self.time_mlp = nn.Linear(time_emb_dim, in_channels)
            self.block = InceptionResidualBlock(in_channels)
            self.transform = nn.Conv1d(in_channels, out_channels, 4, 2, 1)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        x = self.block(x)

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 1]
        x = x + time_emb

        x = self.transform(x)

        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, n_channels):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels
        self.LINEAR_SCALE = 5000

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(half_dim)
        exponents = 1e-4 ** exponents
        exponents = self.LINEAR_SCALE * noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 64
        # time_emb_dim = 128

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            # nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # self.time_mlp = nn.Sequential(
        #     PositionalEncoding(time_emb_dim),
        #     nn.Linear(time_emb_dim, time_emb_dim),
        #     nn.Linear(time_emb_dim, time_emb_dim),
        #     nn.ReLU()
        # )

        # Initial projection
        self.conv0 = nn.Conv1d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([UnetBlock(down_channels[i], down_channels[i + 1], time_emb_dim) \
                                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([UnetBlock(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv1d(up_channels[-1], 1, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)


if __name__ == '__main__':
    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    x = torch.rand((1, 1, 2496))
    out = model(x)
    print(out.shape)
