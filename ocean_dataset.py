import os
import torch
from torch.utils.data import Dataset

class OceanDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        sample = torch.load(file_path, map_location='cpu', weights_only=True)
        return sample
    