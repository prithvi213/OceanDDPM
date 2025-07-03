import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=4, num_steps=1000, base_channels=64):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps

        # Sinusoidal embeddings for time conditioning
        def get_sinusoidal_embeddings(num_steps, embedding_dim):
            position = torch.arange(num_steps).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
            embeddings = torch.zeros(num_steps, embedding_dim)
            embeddings[:, 0::2] = torch.sin(position * div_term)
            embeddings[:, 1::2] = torch.cos(position * div_term)
            return embeddings

        self.register_buffer('time_embeddings', get_sinusoidal_embeddings(num_steps, base_channels * 4))

        # Layer-specific time MLPs
        self.time_mlp_enc1 = nn.Sequential(
            nn.Linear(base_channels * 4, 128),
            nn.SiLU(),
            nn.Linear(128, base_channels * 2)  # 128 channels
        )
        self.time_mlp_enc2 = nn.Sequential(
            nn.Linear(base_channels * 4, 128),
            nn.SiLU(),
            nn.Linear(128, base_channels * 4)  # 256 channels
        )
        self.time_mlp_enc3 = nn.Sequential(
            nn.Linear(base_channels * 4, 128),
            nn.SiLU(),
            nn.Linear(128, base_channels * 8)  # 512 channels
        )
        self.time_mlp_bottleneck = nn.Sequential(
            nn.Linear(base_channels * 4, 128),
            nn.SiLU(),
            nn.Linear(128, base_channels * 8)  # 512 channels
        )

        # Double base channels for increased capacity
        base_channels = base_channels * 2  # Now 128

        # Encoder with downsampling
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck with self-attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=base_channels * 4, num_heads=8)

        # Decoder with upsampling
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 6, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, mask):
        batch_size, _, h, w = x.shape
        x = x * ~mask

        # Pad input to be divisible by 8
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Time embeddings
        t = t.long().clamp(0, self.num_steps - 1)
        t_emb_enc1 = self.time_mlp_enc1(self.time_embeddings[t]).view(batch_size, -1, 1, 1)
        t_emb_enc2 = self.time_mlp_enc2(self.time_embeddings[t]).view(batch_size, -1, 1, 1)
        t_emb_enc3 = self.time_mlp_enc3(self.time_embeddings[t]).view(batch_size, -1, 1, 1)
        t_emb_bottleneck = self.time_mlp_bottleneck(self.time_embeddings[t]).view(batch_size, -1, 1, 1)

        # Encoder
        x1 = self.enc1(x) + t_emb_enc1
        p1 = self.pool1(x1)
        x2 = self.enc2(p1) + t_emb_enc2
        p2 = self.pool2(x2)
        x3 = self.enc3(p2) + t_emb_enc3
        p3 = self.pool3(x3)

        # Bottleneck
        xb = self.bottleneck(p3) + t_emb_bottleneck
        xb_flat = xb.permute(2, 3, 0, 1).reshape(-1, batch_size, xb.shape[1])
        xb_attn, _ = self.attention(xb_flat, xb_flat, xb_flat)
        xb = xb + xb_attn.reshape(xb.shape[2], xb.shape[3], batch_size, xb.shape[1]).permute(2, 3, 0, 1)

        # Decoder
        x = self.up3(xb)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]

        return x
    
class Diffusion:
    def __init__(self, model, num_steps, beta_0, beta_f, device):
        self.model = model.to(device)
        self.num_steps = num_steps
        self.device = device

        self.betas = torch.linspace(beta_0, beta_f, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0).to(device)

    def set_device(self, device):
        self.device = device
        self.model.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cum_prod = self.alpha_cum_prod.to(device)

    def forward_diffusion(self, x_0, t, mask):
        noise = torch.randn_like(x_0, device=self.device, dtype=torch.float32)
        masked_noise = noise * ~mask
        #noise = torch.full(x_0.shape, 1e-6, device=self.device, dtype=torch.float32)
        t = t.to(dtype=torch.long)
        alpha_cum_prod = self.alpha_cum_prod[t]
        sqrt_alpha_cum_prod = torch.sqrt(alpha_cum_prod).view(-1, 1, 1, 1)
        sqrt_1_minus_alpha_cum_prod = torch.sqrt(1.0 - alpha_cum_prod).view(-1, 1, 1, 1)

        x_t = sqrt_alpha_cum_prod * x_0 + sqrt_1_minus_alpha_cum_prod * masked_noise
        x_t = x_t * (~mask) + x_0 * mask
        return x_t, noise
    
    def sample(self, shape, mask, device=None):
        if device is not None:
            self.set_device(device)

        x_t = torch.randn(shape, device=self.device)

        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x_t, t_tensor, mask)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_cum_prod[t]
            beta_t = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise
            else:
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            
            x_t = torch.where(mask == 1, x_t, torch.full_like(x_t, float('nan')))

        x_t = torch.where(mask == 1, x_t, torch.full_like(x_t, float('nan')))
        return x_t
    