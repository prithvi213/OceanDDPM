import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=4, num_steps=1000, base_channels=64):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps

        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_channels * 4),
            nn.ReLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Conv2d(base_channels * 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        batch_size = x.shape[0]
        t = t.view(batch_size, 1).float() / self.num_steps
        t_emb = self.time_mlp(t).view(batch_size, -1, 1, 1)

        # Down-sampling with ReLU activations
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck with ReLU
        xb = self.bottleneck(x3) + t_emb

        # Up-sampling with ReLU and skip connections
        x = self.dec3(torch.cat([xb, x3], dim=1))
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.dec1(torch.cat([x, x1], dim=1))

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
        noise = torch.randn_like(x_0, device=self.device, dtype=torch.float8_e4m3fn)
        alpha_cum_prod = self.alpha_cum_prod[t]
        sqrt_alpha_cum_prod = torch.sqrt(alpha_cum_prod).view(-1, 1, 1, 1)
        sqrt_1_minus_alpha_cum_prod = torch.sqrt(1.0 - alpha_cum_prod).view(-1, 1, 1, 1)

        x_t = sqrt_alpha_cum_prod * x_0 + sqrt_1_minus_alpha_cum_prod * noise
        return x_t, noise
    
    def sample(self, shape, device=None):
        if device is not None:
            self.set_device(device)

        x_t = torch.randn(shape, device=self.device)

        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x_t, t_tensor)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_cum_prod[t]
            beta_t = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise
            else:
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)

        return x_t
    