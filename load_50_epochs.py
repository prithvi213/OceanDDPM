import os
import torch

#location = './checkpoint100_newmodel.pth'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#s = torch.load(location, map_location=device, weights_only=False)

"""
for key, value in s.items():
    print(f"\nKey: {key}")
    if isinstance(value, dict):
        print(f"  Dict with keys: {list(value.keys())}")
    elif isinstance(value, torch.Tensor):
        print(f"  Tensor with shape: {value.shape}")
    else:
        print(f"  Type: {type(value)} | Value: {value}")
"""

mask = torch.load('mask.pth')
data_dir = './preprocessed_data/'
files = [file for file in os.listdir(data_dir)]

tensor_list = [torch.load(os.path.join(data_dir, file)) for file in files]
combined_tensor = torch.stack(tensor_list, dim=0)
print(combined_tensor)

