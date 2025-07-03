import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal
import numpy as np
import torch
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import random
import math
import pickle
import os
from matplotlib.colors import ListedColormap, BoundaryNorm

# Directory containing your 1,000 .pt files
data_dir = "./preprocessed_data/"  # Replace with your directory path
mask = torch.load('mask.pth', map_location='cpu')

# List to store data for each channel
combined_data = [[] for _ in range(4)]  # One list per channel

# Step 1: Load and combine data from all 1,000 .pt files
for i in range(4000, 6000, 1):
    file_path = os.path.join(data_dir, f"data_{i+1:04d}.pt")  # Adjust filename pattern as needed
    if os.path.exists(file_path):
        # Load the .pt file (assumes tensor of shape 4 x 169 x 300)
        tensor = torch.load(file_path)
        
        # Ensure the tensor has the expected shape
        if tensor.shape == (4, 169, 300):
            tensor[mask] = np.nan
            # Append data for each channel
            for channel in range(4):
                combined_data[channel].append(tensor[channel])  # Store 169 x 300 tensor
        else:
            print(f"Warning: File {file_path} has unexpected shape {tensor.shape}")
    else:
        print(f"File {file_path} not found")

# Assuming combined_data is shaped [time, channel, lat, lon] after swapaxes
combined_data = np.swapaxes(combined_data, 0, 1)  # shape becomes [1000, 4, 169, 300]
print(combined_data.shape)

mean = np.array([35.7845, -0.0201, 0.0664, 0.0718])
std = np.array([1.7568, 0.2273, 0.2687, 0.2271])

for channel in range(4):
    combined_data[:, channel, :, :] = (combined_data[:, channel, :, :] * std[channel]) + mean[channel]
print(combined_data)

# Create an xarray DataArray
sample_index = 69 # or any index from 0 to 999
sample = combined_data[sample_index]  # shape: (4, 169, 300)

if hasattr(sample, 'numpy'):
    sample = sample.numpy()

x = np.linspace(17.0, 31.0, 169)
y = np.linspace(-99.0, -74.08, 300)
X, Y = np.meshgrid(y, x)
print(sample)

# For Salinity, sample 7 colors from the RdBu colormap
rd_bu_cmap = plt.get_cmap('Blues')  # Get the RdBu colormap
colors = [rd_bu_cmap(i / 6.0) for i in range(7)]  # Sample 7 colors (0.0, 0.1667, ..., 1.0)
cmap = ListedColormap(colors)  # Custom colormap for Salinity

colorbar_limits = [
    [32, 39],  # Salinity
    [-0.7, 0.7],  # Uo
    [-0.7, 0.7],  # Vo
    [-0.6, 0.6]   # Zos
]

colorbar_ticks = [
    np.arange(32, 40, 1),      # Salinity: 34, 35, 36, 37, 38, 39
    np.arange(-0.7, 0.9, 0.2), # Uo: -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6 (stop at 0.7 due to vmax)
    np.arange(-0.7, 0.9, 0.2), # Vo: -0.8, -0.4, 0, 0.4, 0.8
    np.arange(-0.6, 0.8, 0.2)  # Zos: -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6
]

levels = []
norms = []
for channel_idx in range(4):
    vmin, vmax = colorbar_limits[channel_idx]
    channel_norm = BoundaryNorm(colorbar_ticks[channel_idx], cmap.N, clip=True)
    levels.append(colorbar_ticks[channel_idx])
    norms.append(channel_norm)

# Define different colormaps for each channel
colormaps = [cmap for _ in range(4)]

# Plot each of the 4 channels
fig, axs = plt.subplots(2, 2, figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
variables = ["Salinity", "Uo", "Vo", "Zos"]

for i in range(2):
    for j in range(2):
        channel_idx = i * 2 + j
        axs[i, j].set_facecolor('green')
        axs[i, j].add_feature(cfeature.LAND, facecolor='darkgreen', edgecolor='black')
        cf = axs[i, j].contourf(
            X, Y, sample[channel_idx],
            levels=levels[channel_idx],  # Define the boundaries for intervals
            cmap=colormaps[channel_idx],  # Same colormap for all channels
            norm=norms[channel_idx],  # Use BoundaryNorm to map values to intervals
            vmin=colorbar_limits[channel_idx][0],
            vmax=colorbar_limits[channel_idx][1]
        )
        axs[i, j].coastlines()
        axs[i, j].set_title(f'{variables[i * 2 + j]}')
        axs[i, j].axis('off')
        cbar = plt.colorbar(cf, ax=axs[i, j], orientation='vertical')
        cbar.set_ticks(colorbar_ticks[channel_idx])
        # Explicitly set the colorbar limits to vmin and vmax
        # Adjust tick label font size to ensure all ticks are visible
        cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()