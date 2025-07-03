import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import wasserstein_distance

data_dir_exact = "./preprocessed_data/"
data_dir_new = "./new_data_200/"
mask = torch.load('mask.pth', map_location='cpu')

combined_data_exact = [[] for _ in range(4)]
combined_data_new = [[] for _ in range(4)]

for i in range(1000):
    file_path_exact = os.path.join(data_dir_exact, f"data_{i+1:04d}.pt")
    if os.path.exists(file_path_exact):
        # Load the .pt file (assumes tensor of shape 4 x 169 x 300)
        tensor = torch.load(file_path_exact)
        
        # Ensure the tensor has the expected shape
        if tensor.shape == (4, 169, 300):
            tensor[mask] = np.nan
            # Append data for each channel
            for channel in range(4):
                combined_data_exact[channel].append(tensor[channel])  # Store 169 x 300 tensor
        else:
            print(f"Warning: File {file_path_exact} has unexpected shape {tensor.shape}")
    else:
        print(f"File {file_path_exact} not found")

for i in range(140):
    file_path_new = os.path.join(data_dir_new, f"sample_{i+1:04d}.pt")  # Adjust filename pattern as needed
    if os.path.exists(file_path_new):
        # Load the .pt file (assumes tensor of shape 4 x 169 x 300)
        tensor = torch.load(file_path_new)
        
        # Ensure the tensor has the expected shape
        if tensor.shape == (4, 169, 300):
            tensor[mask] = np.nan
            # Append data for each channel
            for channel in range(4):
                combined_data_new[channel].append(tensor[channel])  # Store 169 x 300 tensor
        else:
            print(f"Warning: File {file_path_new} has unexpected shape {tensor.shape}")
    else:
        print(f"File {file_path_new} not found")

# Step 2: Stack the data for each channel and flatten
for channel in range(4):
    # Stack along a new dimension (e.g., 1000 x 169 x 300) and flatten
    combined_data_exact[channel] = torch.stack(combined_data_exact[channel]).flatten()
    combined_data_exact[channel] = combined_data_exact[channel].numpy()

    combined_data_new[channel] = torch.stack(combined_data_new[channel]).flatten()
    combined_data_new[channel] = combined_data_new[channel].numpy()

mean = np.array([35.7845, -0.0201, 0.0664, 0.0718])
std = np.array([1.7568, 0.2273, 0.2687, 0.2271])

for channel in range(4):
    combined_data_exact[channel] = (combined_data_exact[channel] * std[channel]) + mean[channel]

bins = 200
# Remove NaNs
exact_clean = combined_data_exact[0][~np.isnan(combined_data_exact[0])]
new_clean   = combined_data_new[0][~np.isnan(combined_data_new[0])]

# Get safe range
#range_min = min(exact_clean.min(), new_clean.min())
#range_max = max(exact_clean.max(), new_clean.max())
range_min = min(np.percentile(exact_clean, 5), np.percentile(new_clean, 5))
range_max = max(np.percentile(exact_clean, 95), np.percentile(new_clean, 95))
#range_min = -0.5
#range_max = 0.7

# Compute histograms
counts1, bin_edges = np.histogram(exact_clean, bins=bins, range=(range_min, range_max))
counts2, _         = np.histogram(new_clean,   bins=bins, range=(range_min, range_max))

# Compute bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Normalize the counts
weights1 = counts1 / counts1.sum()
weights2 = counts2 / counts2.sum()

# Compute Wasserstein Distance
wd = wasserstein_distance(bin_centers, bin_centers, u_weights=weights1, v_weights=weights2)

print(f"Wasserstein Distance: {wd:.4f}")
