import xarray as xr
import os
from tqdm import tqdm
import torch
import numpy as np
import math


# Get sorted list of files
data_dir = "./original_data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
file_list = sorted([
    os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")
])

# Use open_dataset for each file and add dummy dimension for stacking
datasets = []
batch_size = 100
tensor_batches = []
mask = torch.load('./mask.pth')
sum = 0
#mask_expanded = mask.expand(8766, -1, -1, -1)

for i in tqdm(range(0, len(file_list), batch_size)):
    batch_files = file_list[i:i+batch_size]
    batch_datasets = []
    for file in batch_files:
        ds = xr.open_dataset(file)
        ds = ds.expand_dims(dim="sample")
        batch_datasets.append(ds)

    ds_batch = xr.concat(batch_datasets, dim="sample")

    # Align and stack features
    data_vars = [
        ds_batch['so'],
        ds_batch['uo'],
        ds_batch['vo'],
        ds_batch['zos'].expand_dims(dim='depth')
    ]
    xr_data = xr.concat(data_vars, dim='feature')
    xr_data = xr_data.squeeze(dim='depth')
    xr_data = xr_data.transpose('sample', 'feature', 'latitude', 'longitude')

    tensor_batch = torch.tensor(xr_data.values, dtype=torch.float32)
    tensor_batches.append(tensor_batch)

# Stack all the batches (final shape: [8766, 4, 169, 300])
tensor_data = torch.cat(tensor_batches, dim=0).to(device)

running_sum = torch.zeros(4).to(device)
running_sqsum = torch.zeros(4).to(device)
running_count = torch.zeros(4).to(device)

batch_size = 100  # You can tune this depending on available memory
num_samples = tensor_data.shape[0]

for i in range(0, num_samples, batch_size):
    batch = tensor_data[i:i+batch_size]  # shape: [batch, 4, 169, 300]

    mask_expanded = ~torch.isnan(batch)
    batch_filled = torch.nan_to_num(batch, nan=0.0)

    # Sum, squared sum, count per feature
    sum_per_feature = (batch_filled * mask_expanded).sum(dim=(0, 2, 3))
    sqsum_per_feature = ((batch_filled ** 2) * mask_expanded).sum(dim=(0, 2, 3))
    count_per_feature = mask_expanded.sum(dim=(0, 2, 3))

    running_sum += sum_per_feature
    running_sqsum += sqsum_per_feature
    running_count += count_per_feature

# Final mean and std over ocean points (ignoring NaNs)
mean = running_sum / running_count
std = torch.sqrt(running_sqsum / running_count - mean ** 2)

print("Mean per feature:", mean)
print("Std per feature:", std)
print(sum)
