import torch
import os
from diffusion_model import DiffusionModel, Diffusion

# Setup on GPU and directory for where newly created samples will be saved
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = "./new_data_newmodel"

# Calculated mean and std and load mask data
mean = torch.tensor([35.7845, -0.0201, 0.0664, 0.0718]).view(4, 1, 1).to(device)
std = torch.tensor([1.7568, 0.2273, 0.2687, 0.2271]).view(4, 1, 1).to(device)
mask = torch.load('mask.pth', map_location=device, weights_only=True)

# Load State of the Training Model into Diffusion Model
model = DiffusionModel().to(device)
training_file = torch.load('checkpoint100_newmodel.pth', map_location=device, weights_only=False)
model.load_state_dict(training_file['model_state_dict'])
diffusion = Diffusion(model, num_steps=1000, beta_0=1e-4, beta_f=0.02, device=device)
model.eval()

with torch.no_grad():
    # Generate 1,000 Samples
    for idx in range(1000):
        # Create and Normalize the Sample, then save it into directory
        sampled_output = diffusion.sample(shape=(1, 4, 169, 300)).squeeze(0)
        original_data = (sampled_output * std) + mean
        original_data[mask] = float('nan')
        save_path = os.path.join(save_dir, f"sample_{idx+1:04d}.pt")
        torch.save(original_data.cpu(), save_path)

        # Displays progress with sample creation
        if idx % 50 == 0 and idx > 0:
            print(f"{idx} samples created")

        # Free up memory
        del sampled_output
        torch.cuda.empty_cache()
