import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=4, num_steps=1000, base_channels=128):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps
        C = base_channels

        self.time_embedding = nn.Embedding(num_steps, C * 8)
        self.time_mlp = nn.Sequential(
            nn.Linear(C * 8, C * 8),
            nn.SiLU()
        )

        self.enc1  = self._conv_block(in_channels, C)
        self.pool1 = nn.Conv2d(C,     C,     kernel_size=2, stride=2)
        self.enc2  = self._conv_block(C,     C * 2)
        self.pool2 = nn.Conv2d(C * 2, C * 2, kernel_size=2, stride=2)
        self.enc3  = self._conv_block(C * 2, C * 4)
        self.pool3 = nn.Conv2d(C * 4, C * 4, kernel_size=2, stride=2)

        self.bottleneck = self._conv_block(C * 4, C * 8)

        self.up3  = nn.ConvTranspose2d(C * 8, C * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(C * 8, C * 4)
        self.up2  = nn.ConvTranspose2d(C * 4, C * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(C * 4, C * 2)
        self.up1  = nn.ConvTranspose2d(C * 2, C,     kernel_size=2, stride=2)
        self.dec1 = self._conv_block(C * 2, C)

        self.final = nn.Conv2d(C, in_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x, t, mask):
        # mask: [B, 1, H, W] bool, True=ocean False=land
        B, _, H, W = x.shape

        # ✅ constant padding — reflect mirrors values into land regions
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x    = F.pad(x,            (0, pad_w, 0, pad_h), mode='constant', value=0)
            mask = F.pad(mask.float(), (0, pad_w, 0, pad_h), mode='constant', value=0).bool()

        # ✅ Precompute mask at each pooled scale
        m1 = mask.float()                                          # full res
        m2 = F.interpolate(m1, scale_factor=0.5,  mode='nearest') # after pool1
        m3 = F.interpolate(m1, scale_factor=0.25, mode='nearest') # after pool2/3
        m4 = F.interpolate(m1, scale_factor=0.125, mode='nearest')

        # Time embedding
        t_emb = self.time_mlp(self.time_embedding(t)).view(B, -1, 1, 1)

        # ✅ Encoder — mask after every block so land never bleeds into ocean
        e1 = self.enc1(x * m1) * m1
        e2 = self.enc2(self.pool1(e1) * m2) * m2
        e3 = self.enc3(self.pool2(e2) * m3) * m3

        # ✅ Bottleneck
        bn = (self.bottleneck(self.pool3(e3) * m4) + t_emb) * m4

        # ✅ Decoder — mask after every block
        x = self.dec3(torch.cat([self.up3(bn), e3], dim=1)) * m3
        x = self.dec2(torch.cat([self.up2(x),  e2], dim=1)) * m2
        x = self.dec1(torch.cat([self.up1(x),  e1], dim=1)) * m1

        x = self.final(x)

        # Crop padding back to original size
        if pad_h > 0 or pad_w > 0:
            x    = x[:, :, :H, :W]
            mask = mask[:, :, :H, :W]

        # ✅ Final guarantee — land is always zero in output
        return x * mask.float()


class Diffusion:
    def __init__(self, model, num_steps, beta_0, beta_f, device):
        self.model          = model.to(device)
        self.num_steps      = num_steps
        self.device         = device
        self.betas          = torch.linspace(beta_0, beta_f, num_steps, device=device)
        self.alphas         = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)

    def set_device(self, device):
        self.device         = device
        self.model.to(device)
        self.betas          = self.betas.to(device)
        self.alphas         = self.alphas.to(device)
        self.alpha_cum_prod = self.alpha_cum_prod.to(device)

    def forward_diffusion(self, x_0, t, mask):
        # mask: True=ocean, False=land
        noise = torch.randn_like(x_0) * mask.float()   # noise only on ocean

        t = t.to(dtype=torch.long)
        alpha_cum_prod = self.alpha_cum_prod[t].view(-1, 1, 1, 1)
        sqrt_alpha_cum_prod = torch.sqrt(alpha_cum_prod)
        sqrt_1_minus_alpha_cum_prod = torch.sqrt(1.0 - alpha_cum_prod)

        x_t = sqrt_alpha_cum_prod * x_0 + sqrt_1_minus_alpha_cum_prod * noise
        x_t = torch.where(mask, x_t, x_0)

        return x_t, noise

    def sample(self, shape, mask, device=None):
        if device is not None:
            self.set_device(device)

        x_t = torch.randn(shape, device=self.device)
        x_t = torch.where(mask, x_t, torch.zeros_like(x_t))

        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x_t, t_tensor, mask)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_cum_prod[t]
            beta_t = self.betas[t]

            x_t = (1 / alpha_t.sqrt()) * (
                x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * noise_pred
            )

            if t > 0:
                x_t = x_t + beta_t.sqrt() * (torch.randn_like(x_t) * mask.float())

            x_t = torch.where(mask, x_t, torch.zeros_like(x_t))

        x_t = torch.where(mask, x_t, torch.full_like(x_t, float('nan')))

        return x_t
    