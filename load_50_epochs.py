import os
import torch

location = './preprocessed_data/data_0242.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s = torch.load(location, map_location=device, weights_only=False)
print(s)
