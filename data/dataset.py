import json
import os
from pathlib import Path
import random
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from tqdm import tqdm
from data.dataloader import MultiEpochsDataLoader
from torch.utils import data

class UNet_Dataset(data.Dataset):
    def __init__(self, filepath: str, image_dir: str, labels_dir: str, limit_files=None):
        self.filepath = filepath
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        
        self.limit_files = limit_files
        self.dataset = self._load_data(filepath)

    def __len__(self):
        return len(self.dataset.keys())
    
    def __getitem__(self, idx):
        idx_k = list(self.dataset.keys())[idx]
        items = self.dataset[idx_k]

        image = Image.open(items['image']).convert('L')
        image = pil_to_tensor(image).to(torch.float32)

        label = np.load(items['label'])
        label = torch.tensor(label).to(torch.float32)
        return image, label
        
    def _load_data(self, path):
        items = {}
        with open(path, 'r', encoding='utf-8') as f:
            line = f.readline()
            data = json.loads(line)
            images = data['images']
            print("Loading images json...")
            num = 0
            for image in tqdm(images):
                id = image['id']
                file_name = image['file_name']
                data_name = Path(file_name).stem
                items[id] = {
                    'name': data_name,
                    'image': Path(self.image_dir) / (data_name + '.png'),
                    'label': Path(self.labels_dir) / (data_name + '.npy')
                }

                num += 1
                if num >= self.limit_files:
                    break

        return items
    
def build_loader(filepath, image_dir, labels_dir, batch_size=42, limit_files=None):
    dataset = UNet_Dataset(filepath, image_dir, labels_dir, limit_files)
    generator = torch.Generator().manual_seed(200)

    train_set, validation_set = data.random_split(dataset, [0.7, 0.3], generator)

    train_ld = MultiEpochsDataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    validation_ld = MultiEpochsDataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return train_ld, validation_ld