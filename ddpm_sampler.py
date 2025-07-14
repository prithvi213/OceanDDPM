import torch
import os
from diffusion_model import DiffusionModel, Diffusion
from ocean_dataset import OceanDataset
from torch.utils.data import DataLoader

# Setup on GPU and directory for where newly created samples will be saved
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = "./new_sparse_samples"

# Calculated mean and std and load mask data
mean = torch.tensor([35.7845, -0.0201, 0.0664, 0.0718]).view(4, 1, 1).to(device)
std = torch.tensor([1.7568, 0.2273, 0.2687, 0.2271]).view(4, 1, 1).to(device)
land_mask = torch.load('mask.pth', map_location=device, weights_only=True)
sparse_mask = torch.load('sparse.pth', map_location=device, weights_only=True)

# Load State of the Training Model into Diffusion Model
model = DiffusionModel().to(device)
training_file = torch.load('checkpoint100_withsparsity.pth', map_location=device, weights_only=False)
model.load_state_dict(training_file['model_state_dict'])
diffusion = Diffusion(model, num_steps=1000, beta_0=1e-4, beta_f=0.02, device=device)
dataset = OceanDataset(data_dir='./preprocessed_data_test/')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
model.eval()

with torch.no_grad():
    # Generate 1,000 Samples
    for batch_idx, data in enumerate(dataloader):
        x_0 = data.to(device).float()
        # Create and Normalize the Sample, then save it into directory
        sampled_output = diffusion.sample(shape=(1, 4, 169, 300), x_0=x_0, sparse_mask=sparse_mask, land_mask=land_mask).squeeze(0)
        original_data = (sampled_output * std) + mean
        original_data[land_mask] = float('nan')
        save_path = os.path.join(save_dir, f"sample_{batch_idx+1:04d}.pt")
        torch.save(original_data.cpu(), save_path)

        # Displays progress with sample creation
        if batch_idx % 10 == 9 and batch_idx > 0:
            print(f"{batch_idx} samples created")

        # Free up memory
        del sampled_output
        torch.cuda.empty_cache()
