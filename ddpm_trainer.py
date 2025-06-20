import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ocean_dataset import OceanDataset
from diffusion_model import Diffusion, DiffusionModel
import numpy as np
import os
import torch.profiler
from torch.cuda.amp import autocast, GradScaler

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize mask on GPU
mask = torch.zeros((169, 300), dtype=torch.bool)
with open('coordinates.txt', 'r') as file:
    for line in file:
        x, y = map(float, line.strip().split(','))
        mask[int(x), int(y)] = True
mask = mask.unsqueeze(0).expand(4, -1, -1).to(device)

# Initialize dataset, dataloader, model, diffusion, and optimizer
dataset = OceanDataset(data_dir="/scratch/your_username/preprocessed_data")  # Updated path
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = DiffusionModel().to(device)
diffusion = Diffusion(model, num_steps=1000, beta_0=1e-4, beta_f=0.02, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler('cuda')  # Updated for deprecation
num_epochs = 50
epoch_losses = []
cumulative_mses = []
cumulative_mse_loss = 0.0
checkpoint_path = '/scratch/your_username/checkpoint_epoch_10.pth'  # Updated path

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    epoch_losses = checkpoint['epoch_losses']
    cumulative_mse_loss = checkpoint['cumulative_mse_loss']
    cumulative_mses = checkpoint['cumulative_mses']
    print(f"Resuming training from epoch {start_epoch} (loaded from {checkpoint_path})")
else:
    print("No checkpoint found, starting training from epoch 1")

# Timing setup
timings = {
    "data_loading": [],
    "mask_expansion": [],
    "time_sampling": [],
    "forward_diffusion": [],
    "model_inference": [],
    "loss_computation": [],
    "backward_pass": [],
    "checkpoint_saving": []
}

# Profiler setup
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('/scratch/your_username/log/profiler_output'),  # Updated path
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as profiler:
    for epoch in range(num_epochs):
        model.train()
        curr_epoch_loss = 0.0
        torch.cuda.synchronize()

        for batch_idx, data in enumerate(dataloader):
            # Timing for data_loading
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.profiler.record_function("data_loading"):
                x_0 = data.to(device, non_blocking=True).float()
                batch_size = x_0.size(0)
            end.record()
            torch.cuda.synchronize()
            timings["data_loading"].append(start.elapsed_time(end))

            # Timing for mask_expansion
            start.record()
            with torch.profiler.record_function("mask_expansion"):
                mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
            end.record()
            torch.cuda.synchronize()
            timings["mask_expansion"].append(start.elapsed_time(end))

            # Timing for time_sampling
            start.record()
            with torch.profiler.record_function("time_sampling"):
                t = torch.randint(0, diffusion.num_steps, (batch_size,), device=device).long()
                optimizer.zero_grad()
            end.record()
            torch.cuda.synchronize()
            timings["time_sampling"].append(start.elapsed_time(end))

            with autocast('cuda'):  # Updated for deprecation
                # Timing for forward_diffusion
                start.record()
                with torch.profiler.record_function("forward_diffusion"):
                    x_t, noise = diffusion.forward_diffusion(x_0, t, mask_expanded)
                end.record()
                torch.cuda.synchronize()
                timings["forward_diffusion"].append(start.elapsed_time(end))

                # Timing for model_inference
                start.record()
                with torch.profiler.record_function("model_inference"):
                    noise_pred = model(x_t, t)
                end.record()
                torch.cuda.synchronize()
                timings["model_inference"].append(start.elapsed_time(end))

                # Timing for loss_computation
                start.record()
                with torch.profiler.record_function("loss_computation"):
                    noise_pred[mask_expanded] = 0
                    mse_loss = (noise_pred - noise) ** 2
                    ocean_elements = (~mask_expanded).float().sum()
                    loss = mse_loss.sum() / ocean_elements
                    curr_epoch_loss += loss.item()
                end.record()
                torch.cuda.synchronize()
                timings["loss_computation"].append(start.elapsed_time(end))

            # Timing for backward_pass
            start.record()
            with torch.profiler.record_function("backward_pass"):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            end.record()
            torch.cuda.synchronize()
            timings["backward_pass"].append(start.elapsed_time(end))

            if batch_idx % 54 == 53:
                average_loss = curr_epoch_loss / 54
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {average_loss:.10f}")
                # Print average timings every 54 batches
                for key, times in timings.items():
                    if times:
                        print(f"Avg {key} time: {np.mean(times[-54:]):.3f} ms")
                curr_epoch_loss = 0.0

            profiler.step()

        avg_epoch_loss = curr_epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        cumulative_mse_loss += avg_epoch_loss
        cumulative_mses.append(cumulative_mse_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg MSE: {avg_epoch_loss:.10f}, Cumulative MSE: {cumulative_mse_loss:.10f}")

        # Timing for checkpoint_saving
        start.record()
        with torch.profiler.record_function("checkpoint_saving"):
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_losses': epoch_losses,
                    'cumulative_mse_loss': cumulative_mse_loss,
                    'cumulative_mses': cumulative_mses,
                }
                checkpoint_path = f'/scratch/your_username/checkpoint_epoch_{epoch+1}.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
                timings["checkpoint_saving"].append(start.elapsed_time(end))
                break
        end.record()
        torch.cuda.synchronize()
        if (epoch + 1) % 10 == 0:
            timings["checkpoint_saving"].append(start.elapsed_time(end))

        # Print profiler summary for verification
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
