import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math

# Directory containing your 1,000 .pt files
data_dir = "./preprocessed_data/"  # Replace with your directory path
mask = torch.load('mask.pth', map_location='cpu')

# List to store data for each channel
combined_data = [[] for _ in range(4)]  # One list per channel

# Step 1: Load and combine data from all 1,000 .pt files
for i in range(1000):
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

# Step 2: Stack the data for each channel and flatten
for channel in range(4):
    # Stack along a new dimension (e.g., 1000 x 169 x 300) and flatten
    combined_data[channel] = torch.stack(combined_data[channel]).flatten()
    # Convert to NumPy for plotting
    combined_data[channel] = combined_data[channel].numpy()

mean = np.array([35.7845, -0.0201, 0.0664, 0.0718])
std = np.array([1.7568, 0.2273, 0.2687, 0.2271])

for channel in range(4):
    combined_data[channel] = (combined_data[channel] * std[channel]) + mean[channel]

# Step 3: Plot histograms for each channel
fig, axes = plt.subplots(2, 2, figsize=(10, 5))  # 2x2 grid for 4 channels
axes = axes.flatten()  # Flatten for easy indexing
print(combined_data)

for channel in range(4):
    axes[channel].hist(combined_data[channel], edgecolor='black', bins=200, color=f'C{channel}', alpha=0.7)
    axes[channel].set_xlabel('Value')
    axes[channel].set_ylabel('Frequency')

axes[0].set_xlim([34, 37])
axes[1].set_xlim([-0.85, 0.75])
axes[2].set_xlim([-1, 1.3])
axes[3].set_xlim([-0.5, 0.7])

axes[0].set_title(f'Salinity Histogram')
axes[1].set_title(f'Uo Histogram')
axes[2].set_title(f'Vo Histogram')
axes[3].set_title(f'Zos Histogram')

plt.tight_layout()
plt.show()