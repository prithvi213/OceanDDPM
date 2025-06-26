import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ocean_dataset import OceanDataset
from diffusion_model import Diffusion, DiffusionModel
import numpy as np
import os
import time
import sys
#from torch.profiler import profile, record_function

# Initialize device and the masked locations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
mask = torch.zeros((169, 300), dtype=torch.bool)

with open('coordinates.txt', 'r') as file:
    for line in file:
        x, y = map(float, line.strip().split(','))
        mask[int(x), int(y)] = True

mask = mask.unsqueeze(0).expand(4, -1, -1)

# Initialize OceanDataset, DataLoader, Diffusion Model, Sampler, and Optimizer
dataset = OceanDataset(data_dir="./preprocessed_data")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
model = DiffusionModel().to(device)
print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
diffusion = Diffusion(model, num_steps=1000, beta_0=1e-4, beta_f=0.02, device=device)
print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 20
epoch_losses = []
cumulative_mses = []
cumulative_mse_loss = 0.0
#checkpoint_path = 'checkpoint_epoch_10.pth'

"""
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('/log/profiler_output'),  # Updated path
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as profiler:
"""
# Training Loop Starts Here (50 Epochs)
for epoch in range(num_epochs):
    # Start Training and Keep Track of Loss
    model.train()
    print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    curr_epoch_loss = 0.0
    every_10_batch_loss = 0.0

    # Go through each of the 548 batches of size 16
    for batch_idx, data in enumerate(dataloader):
        print(f"\n Batch {batch_idx}:")
        #with record_function("get_data"):
        # Get the 16 samples from the dataloader
        batch_start_time = time.perf_counter()
        batch_loss = 0.0
        #print(f"batch_start_time = {sys.getsizeof(batch_start_time) / 1024**2:.2f} MB")
        #print(f"batch_loss = {sys.getsizeof(batch_loss) / 1024**2:.2f} MB")

        x_0 = data.to(device).float()
        batch_size = x_0.size(0)
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #print(f"x_0 shape = {x_0.shape}, size = {x_0.nelement() * x_0.element_size() / 1024**2:.2f} MB")

        #with record_function("expand_mask"):
        # Expand the mask for batch use (Filtering out masked values from noise and mse loss)
        mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #print(f"mask_expanded shape = {mask_expanded.shape}, size = {mask_expanded.nelement() * mask_expanded.element_size() / 1024**2:.2f} MB")

        #with record_function("random_ts"):
        # Take a time step from random and start the zero-gradient for optimizer
        t = torch.randint(0, diffusion.num_steps, (batch_size,), dtype=torch.float32, device=device)
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        mem_before_zero = torch.cuda.memory_allocated()
        optimizer.zero_grad()
        mem_after_zero = torch.cuda.memory_allocated()

        mem_zero_grad = mem_after_zero - mem_before_zero
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #print(f"[Zero Grad] Memory Used: {mem_zero_grad / 1024**2:.2f} MB")
        #print(f"t = {t.element_size() / 1024**2:.2f} MB")
        
        #with record_function("forward_diffusion"):
        # Apply forward diffusion and ensure that the mask is applied to the predicted noise
        x_t, noise = diffusion.forward_diffusion(x_0, t, mask_expanded)
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #print(f"x_t size: {x_t.nelement() * x_t.element_size() / 1024**2:.2f} MB")
        #print(f"noise size: {noise.nelement() * noise.element_size() / 1024**2:.2f} MB")
        
        #with record_function("apply_to_model"):
        noise_pred = model(x_t, t)
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #print(f"noise_pred size: {noise_pred.nelement() * noise_pred.element_size() / 1024**2:.2f} MB")

        # Set the noise land values to 0
        noise_pred[mask_expanded] = 0
        noise[mask_expanded] = 0

        #with record_function("calculate_MSE_loss"):
        # Calculate the loss
        mse_loss = (noise_pred - noise) ** 2
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #print(f"mse_loss size: {mse_loss.nelement() * mse_loss.element_size() / 1024**2:.2f} MB")

        ocean_elements = (~mask_expanded).sum()
        loss = mse_loss.sum() / ocean_elements
        curr_epoch_loss += loss.item()
        batch_loss += loss.item()
        every_10_batch_loss += loss.item()
        #print(f"ocean_elements = {sys.getsizeof(ocean_elements) / 1024**2:.2f} MB")
        #print(f"loss = {sys.getsizeof(loss) / 1024**2:.2f} MB")
        #print(f"curr_epoch_loss = {sys.getsizeof(curr_epoch_loss) / 1024**2:.2f} MB")
        #print(f"batch_loss = {sys.getsizeof(batch_loss) / 1024**2:.2f} MB")
        #print(f"every_10_batch_loss = {sys.getsizeof(every_10_batch_loss) / 1024**2:.2f} MB")

        #with record_function("backward_propagation"):
        # Apply backward propagation
        mem_before_backward = torch.cuda.memory_allocated()
        loss.backward()
        mem_after_backward = torch.cuda.memory_allocated()
        mem_backward = mem_after_backward - mem_before_backward
        #print(f"[Backward] Memory Used: {mem_backward / 1024**2:.2f} MB")
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        mem_before_step = torch.cuda.memory_allocated()
        optimizer.step()
        mem_after_step = torch.cuda.memory_allocated()
        mem_step = mem_after_step - mem_before_step
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #print(f"[Optimizer Step] Memory Used: {mem_step / 1024**2:.2f} MB")

        # If at the end of the batch, print step and calculated loss
        if batch_idx % 54 == 53:
            average_loss = every_10_batch_loss / 54
            #print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {average_loss:.10f}")
            total_loss = 0.0
            every_10_batch_loss = 0.0
        
        batch_time = time.perf_counter() - batch_start_time
        #print(f"batch_time = {sys.getsizeof(batch_time) / 1024**2:.2f} MB")
        #print(f"Batch: {batch_idx}, Batch Time: {batch_time}, Batch Loss: {batch_loss}")

        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #print(f"[GPU] Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"[GPU] Peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    avg_epoch_loss = curr_epoch_loss / len(dataloader)
    epoch_losses.append(avg_epoch_loss)
    cumulative_mse_loss += avg_epoch_loss
    cumulative_mses.append(cumulative_mse_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg MSE: {avg_epoch_loss:.10f}, Cumulative MSE: {cumulative_mse_loss:.10f}")
