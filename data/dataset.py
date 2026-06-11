import json
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from tqdm import tqdm
from data.dataloader import MultiEpochsDataLoader
from torch.utils import data

class UNet_Dataset(data.Dataset):
    def __init__(self, filepath: str, image_dir: str, labels_dir: str, limit_files: int=None, channels: list[int]=None):
        self.filepath = filepath
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        
        self.limit_files = limit_files
        self.channels = [0] + channels if not channels is None else None
        self.dataset = self._load_data(filepath)

    def __len__(self):
        return len(self.dataset.keys())
    
    def __getitem__(self, idx):
        idx_k = list(self.dataset.keys())[idx]
        items = self.dataset[idx_k]

        image = Image.open(items['image']).convert('L')
        image = pil_to_tensor(image).to(torch.float32) / 255.0

        label = np.load(items['label'])

        if self.channels is None: 
            label = torch.tensor(label).to(torch.float32)
        else:
            label = torch.tensor(label[self.channels, :, :]).to(torch.float32)
        
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

                # Skip lateral views
                if 'lateral' in file_name:
                    continue

                data_name = Path(file_name).stem
                items[id] = {
                    'name': data_name,
                    'image': Path(self.image_dir) / (data_name + '.png'),
                    'label': Path(self.labels_dir) / (data_name + '.npy')
                }

                num += 1
                if not self.limit_files is None and num >= self.limit_files:
                    break

        return items
    
def build_loader(filepath, image_dir, labels_dir, batch_size=42, limit_files=None, channels=None):
    dataset = UNet_Dataset(filepath, image_dir, labels_dir, limit_files, channels)
    generator = torch.Generator().manual_seed(200)

    train_set, validation_set = data.random_split(dataset, [0.8, 0.2], generator)
    
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