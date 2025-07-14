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

"""
def process(args):
    print(f"Positional arguments: {args}")
"""

# Initialize device and the masked locations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#mask = torch.zeros((169, 300), dtype=torch.bool)

"""
parser = optparse.OptionParser()
opts, args = parser.parse_args()
process(args)


with open(args[0], 'r') as file:
    for line in file:
        x, y = map(float, line.strip().split(','))
        mask[int(x), int(y)] = True

mask = mask.unsqueeze(0).expand(4, -1, -1)
"""

land_mask = torch.load('mask.pth', map_location=device, weights_only=True)
sparse_mask = torch.load('sparse.pth', map_location=device, weights_only=True)

# Initialize OceanDataset, DataLoader, Diffusion Model, Sampler, and Optimizer
dataset = OceanDataset(data_dir='./preprocessed_data_half/')

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = DiffusionModel().to(device)
diffusion = Diffusion(model, num_steps=1000, beta_0=1e-4, beta_f=0.02, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100
epoch_losses = []
cumulative_mses = []
cumulative_mse_loss = 0.0
training_file = torch.load('checkpoint90_newmodel.pth', map_location=device, weights_only=False)
model.load_state_dict(training_file['model_state_dict'])
criterion = nn.MSELoss()
#checkpoint_path = './checkpoint.pth'

# Load checkpoint if exists
"""
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    epoch_losses = checkpoint['epoch_losses']
    cumulative_mse_loss = checkpoint['cumulative_mse_loss']
    cumulative_mses = checkpoint['cumulative_mses']
    print(f"Resuming training from epoch {start_epoch} (loaded from {checkpoint_path})")
else:
    print("No checkpoint found, starting training from epoch 1")
    start_epoch = 0
"""

start_epoch = 0

# Training Loop Starts Here (100 Epochs)
for epoch in range(start_epoch, start_epoch + num_epochs):
    # Start Training and Keep Track of Loss
    model.train()
    curr_epoch_loss = 0.0
    every_10_batch_loss = 0.0

    # Go through each of the 548 batches of size 16
    for batch_idx, data in enumerate(dataloader):
        #with record_function("get_data"):
        # Get the 16 samples from the dataloader
        batch_start_time = time.perf_counter()
        batch_loss = 0.0
        
        x_0 = data.to(device).float()
        batch_size = x_0.size(0)

        #with record_function("expand_mask"):
        # Expand the mask for batch use (Filtering out masked values from noise and mse loss)
        landmask_expanded = land_mask.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        sparsemask_expanded = sparse_mask.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

        #with record_function("random_ts"):
        # Take a time step from random and start the zero-gradient for optimizer
        t = torch.randint(0, diffusion.num_steps, (batch_size,), dtype=torch.float32, device=device)
        optimizer.zero_grad()
        
        #with record_function("forward_diffusion"):
        # Apply forward diffusion and ensure that the mask is applied to the predicted noise
        x_t, noise = diffusion.forward_diffusion(x_0, t, sparsemask_expanded)
        
        #with record_function("apply_to_model"):
        noise_pred = model(x_t, t, sparsemask_expanded, landmask_expanded)

        # Set the noise land values to 0
        #noise_pred[mask_expanded] = 0
        #noise[mask_expanded] = 0

        #with record_function("calculate_MSE_loss"):
        # Calculate the loss
        mse_loss = (noise_pred - noise) ** 2
        masked_mse_loss = mse_loss * (sparsemask_expanded)
        ocean_elements = (sparsemask_expanded).sum()
        loss = masked_mse_loss.sum() / ocean_elements
        curr_epoch_loss += loss.item()
        batch_loss += loss.item()
        every_10_batch_loss += loss.item()

        #with record_function("backward_propagation"):
        # Apply backward propagation
        loss.backward()
        optimizer.step()

        # If at the end of the batch, print step and calculated loss
        if batch_idx % 54 == 53:
            average_loss = every_10_batch_loss / 54
            print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}], Step [{batch_idx+1}], Loss: {average_loss:.10f}")
            total_loss = 0.0
            every_10_batch_loss = 0.0
        
        batch_time = time.perf_counter() - batch_start_time
        print(f"Batch: {batch_idx}, Batch Time: {batch_time}, Batch Loss: {batch_loss}")

    avg_epoch_loss = curr_epoch_loss / len(dataloader)
    epoch_losses.append(avg_epoch_loss)
    cumulative_mse_loss += avg_epoch_loss
    cumulative_mses.append(cumulative_mse_loss)
    print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}] - Avg MSE: {avg_epoch_loss:.10f}, Cumulative MSE: {cumulative_mse_loss:.10f}")

    if epoch % 10 == 9:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_losses': epoch_losses,
            'cumulative_mse_loss': cumulative_mse_loss,
            'cumulative_mses': cumulative_mses,
        }

        checkpoint_path = f'checkpoint{epoch + 1}_withsparsity.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
