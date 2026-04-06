import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ocean_dataset import OceanDataset
from diffusion_model import Diffusion, DiffusionModel
import numpy as np
import os
import time
#import optparse

def masked_mse(noise_pred, noise, mask):
    diff = (noise_pred - noise) ** 2
    n_ocean = mask.float().sum() * noise_pred.shape[1]
    return (diff * mask).sum() / n_ocean

# Initialize device and the masked locations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mask = torch.load('mask.pth', map_location=device, weights_only=True)
mask = mask.unsqueeze(0)
dataset = OceanDataset(data_dir='./preprocessed_data/')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = DiffusionModel().to(device)
diffusion = Diffusion(model, num_steps=1000, beta_0=1e-4, beta_f=0.02, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 400
epoch_losses = []
cumulative_mses = []
cumulative_mse_loss = 0.0
start_epoch = 0
checkpoint_path = '/checkpoint_100_april2026.pth'

# Training Loop Starts Here (100 Epochs)
for epoch in range(100, 100 + num_epochs):
    # Start Training and Keep Track of Loss
    model.train()
    curr_epoch_loss = 0.0
    every_10_batch_loss = 0.0

    # Go through each of the 548 batches of size 16
    for batch_idx, data in enumerate(dataloader):
        #with record_function("get_data"):
        # Get the 16 samples from the dataloader
        batch_start_time = time.perf_counter()

        x_0 = data.to(device).float()
        #print(f"before nan_to_num: {torch.isnan(x_0).any()}")
        x_0 = torch.nan_to_num(x_0, nan=0.0)
        #print(f"after nan_to_num: {torch.isnan(x_0).any()}")   # must be False
        batch_size = x_0.size(0)
        
        mask_expanded = mask.expand(batch_size, -1, -1, -1).to(device)

        #with record_function("random_ts"):
        # Take a time step from random and start the zero-gradient for optimizer
        t = torch.randint(0, diffusion.num_steps, (batch_size,), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        
        #with record_function("forward_diffusion"):
        # Apply forward diffusion and ensure that the mask is applied to the predicted noise
        x_t, noise = diffusion.forward_diffusion(x_0, t, mask_expanded)
        #print(f"x_0 nan: {torch.isnan(x_0).any()}")
        #print(f"x_t nan: {torch.isnan(x_t).any()}")
        #print(f"noise nan: {torch.isnan(noise).any()}")


        noise_pred = model(x_t, t, mask_expanded)
        #print(f"noise_pred nan: {torch.isnan(noise_pred).any()}")

        loss = masked_mse(noise_pred, noise, mask_expanded)
        #print(f"loss nan: {torch.isnan(loss).any()}")
        
        # Apply backward propagation
        loss.backward()
        optimizer.step()

        every_10_batch_loss += loss.item()
        curr_epoch_loss += loss.item()

        # If at the end of the batch, print step and calculated loss
        if batch_idx % 54 == 53:
            average_loss = every_10_batch_loss / 54
            print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}], Step [{batch_idx+1}], Loss: {average_loss:.10f}")
            every_10_batch_loss = 0.0
        
        batch_time = time.perf_counter() - batch_start_time
        print(f"Batch: {batch_idx}, Batch Time: {batch_time}, Batch Loss: {loss.item():.10f}")

    avg_epoch_loss = curr_epoch_loss / len(dataloader)
    epoch_losses.append(avg_epoch_loss)
    cumulative_mse_loss += avg_epoch_loss
    cumulative_mses.append(cumulative_mse_loss)
    print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}] - Avg MSE: {avg_epoch_loss:.10f}, Cumulative MSE: {cumulative_mse_loss:.10f}")

    if epoch % 10 == 9:
        checkpoint_path = f'checkpoint_{epoch + 1}_april2026.pth'
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_losses': epoch_losses,
            'cumulative_mse_loss': cumulative_mse_loss,
            'cumulative_mses': cumulative_mses,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
