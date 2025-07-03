import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import piq
from skimage.metrics import structural_similarity as ssim

data_dir_exact = "./preprocessed_data/"
data_dir_new = "./new_data/"
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

for i in range(1000):
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
    combined_data_exact[channel] = torch.stack(combined_data_exact[channel])
    combined_data_exact[channel] = combined_data_exact[channel].numpy()

    combined_data_new[channel] = torch.stack(combined_data_new[channel])
    combined_data_new[channel] = combined_data_new[channel].numpy()

mean = np.array([35.7845, -0.0201, 0.0664, 0.0718])
std = np.array([1.7568, 0.2273, 0.2687, 0.2271])
print(combined_data_exact)
print(np.nanmin(combined_data_exact))

for channel in range(4):
    combined_data_new[channel] = (combined_data_new[channel] -  mean[channel]) / std[channel]

combined_data_exact = np.swapaxes(combined_data_exact, 0, 1)  # shape becomes [1000, 4, 169, 300]
combined_data_new = np.swapaxes(combined_data_new, 0, 1)  # shape becomes [1000, 4, 169, 300]

sample_number = 30
exact, new = torch.from_numpy(combined_data_exact[sample_number]).unsqueeze(0), torch.from_numpy(combined_data_new[sample_number]).unsqueeze(0)
exact, new = torch.nan_to_num(exact, nan=0.0), torch.nan_to_num(new, nan=0.0)

print(np.nanmax(new))

shift = 30.0  # Assumes values are mostly in [-3, 3]
exact_shifted = exact + shift
new_shifted = new + shift

ssim = piq.ssim(new_shifted, exact_shifted, data_range=30.0)

#ssim_scores = piq.ssim(new, exact, data_range=6.0)
#print(ssim_scores)
