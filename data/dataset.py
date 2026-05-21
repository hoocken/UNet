import json
import os
from pathlib import Path
import random
import torch

from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import torchio as tio

class UNet_Dataset(data.Dataset):
    def __init__(self, filepath: str, image_dir: str, labels_dir: str):
        self.filepath = filepath
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        
        self.dataset = self._load_data(filepath)

    def __len__(self):
        return len(self.dataset.keys())
    
    def __getitem__(self, idx):
        idx_k = list(self.dataset.keys())[idx]
        items = self.dataset[idx_k]

        image = tio.ScalarImage(items['image'])
        label = tio.ScalarImage(items['labels'])
        return image, label
        
    def _load_data(self, path):
        items = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                id = data['id']
                image = data['file_name']
                data_name = Path(image).stem
                items[id] = {
                    'name': data_name,
                    'image': Path(self.image_dir) / data_name + '.nii.gz',
                    'label': Path(self.labels_dir) / data_name + '.nii.gz'
                }

        return items
    
def build_loader(filepath, image_dir, labels_dir, batch_size=42):
    dataset = UNet_Dataset(filepath, image_dir, labels_dir)
    generator = torch.Generator().manual_seed(200)
    train_set, validation_set = data.random_split(dataset, [0.7, 0.3], generator)

    train_ld = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    validation_ld = data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return train_ld, validation_ld