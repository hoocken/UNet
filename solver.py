from collections import deque
from datetime import datetime
import os
from pathlib import Path
import sys

import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from model.loss import BinaryCrossEntropyLoss, GeneralizedDiceLoss
from model.unet import UNet
from data.dataloader import infinite_iterator
from data.transform import RandomAffine, RandomCrop, GaussianBlur, RandomHorizontalFlip, RandomResize, RandomRotation



class Solver():
    def __init__(self, train_ld, validation_ld, config):
        self.model_dir = config.model_dir

        self.train_ld = train_ld
        self.validation_ld = validation_ld

        self.train_iter = infinite_iterator(train_ld)
        self.valid_iter = infinite_iterator(validation_ld)

        self.lr = config.lr

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet = UNet(config.num_classes).to(self.device)
        self.dsc_loss = GeneralizedDiceLoss().to(self.device)
        self.bce_loss = BinaryCrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optimizer, config.decay) # LR decay
        
        self.moving_avg_alpha = config.moving_avg_alpha
        self.train_threshold = config.train_threshold
        self.valid_threshold = config.valid_threshold
        self.patience = config.patience
        self.valid_cutoff = config.valid_cutoff

        self.epoch_length = config.epoch_length
        self.total_epochs = config.total_epochs
        self.valid_epoch_length = config.valid_epoch_length
        self.start_epoch = 0

        self.transform = v2.Compose([
            RandomHorizontalFlip(),
            RandomCrop(),
            RandomAffine(),
            GaussianBlur(),
        ])

        if config.load_state:
            checkpoint = torch.load(config.load_state, weights_only=True)
            self.unet.load_state_dict(checkpoint['unet_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # LR decay
            self.start_epoch = checkpoint['epoch']

        self.save = config.save_every

        self.checkpoints = Path() / config.model_dir / 'checkpoints' / datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            os.makedirs(self.checkpoints)
        except FileExistsError:
            print(f'{self.checkpoints} already exists!')

        self.writer = SummaryWriter(Path() / config.model_dir / 'logs' / datetime.now().strftime("%Y%m%d-%H%M%S"))

    @torch.no_grad 
    def validate(self):
        valid_loss_list = []
        bce_loss_list = []
        dsc_loss_list = []

        self.unet.eval()

        pbar = tqdm(total=self.valid_epoch_length, ncols=0, desc="Valid Epoch", file=sys.stdout)
        for _ in range(self.valid_epoch_length):
            images, labels = next(self.valid_iter)
            images, labels = images.to(self.device), labels.to(self.device)

            pred = self.unet(images)
            bce_loss = self.bce_loss(pred, labels)
            dsc_loss = self.dsc_loss(pred, labels)
            loss = bce_loss + dsc_loss

            valid_loss_list.append(loss.detach().item())
            bce_loss_list.append(bce_loss.detach().item())
            dsc_loss_list.append(dsc_loss.detach().item())
           
            pbar.update(1)
            pbar.set_postfix(loss=self._calculate_mean_loss(valid_loss_list), bce=self._calculate_mean_loss(bce_loss_list), dsc=self._calculate_mean_loss(dsc_loss_list))

        pbar.close()

        return self._calculate_mean_loss(valid_loss_list), self._calculate_mean_loss(bce_loss_list), self._calculate_mean_loss(dsc_loss_list)

    def train(self, train_loss_list, bce_loss_list, dsc_loss_list):
        pbar = tqdm(total=self.epoch_length, ncols=0, desc="Train Epoch", file=sys.stdout)
        self.unet.train()
            
        for _ in range(self.epoch_length):
            images, labels = next(self.train_iter)
            images = images.to(self.device)
            labels = labels.to(self.device)

            images, labels = self.transform(images, labels)

            pred = self.unet(images)
            bce_loss = self.bce_loss(pred, labels)
            dsc_loss = self.dsc_loss(pred, labels)
            loss = bce_loss + dsc_loss

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            train_loss_list.append(loss.detach().item())
            bce_loss_list.append(bce_loss.detach().item())
            dsc_loss_list.append(dsc_loss.detach().item())

            mean_train_loss = self._calculate_mean_loss(train_loss_list) 
            mean_bce_loss = self._calculate_mean_loss(bce_loss_list)
            mean_dsc_loss = self._calculate_mean_loss(dsc_loss_list)
            
            pbar.update(1)
            pbar.set_postfix(loss=mean_train_loss, bce=mean_bce_loss, dsc=mean_dsc_loss)

        pbar.close()
        
        return mean_train_loss, mean_bce_loss, mean_dsc_loss
        
        
    def _calculate_mean_loss(self, x):
        return sum(x) / len(x)
    
    def training(self):
        max_train_loss = None
        max_valid_loss = None

        patience = 0
        cutoff = 0
        saved_model = None

        ema_train_loss = None
        ema_valid_loss = None

        train_loss_list = deque(maxlen=self.epoch_length)
        bce_loss_list = deque(maxlen=self.epoch_length)
        dsc_loss_list = deque(maxlen=self.epoch_length)

        print("Start training...")
        for i in range(self.start_epoch, self.total_epochs):
            mean_train_loss, mean_train_bce_loss, mean_train_dsc_loss = self.train(train_loss_list, bce_loss_list, dsc_loss_list)

            if ema_train_loss is None:
                ema_train_loss = mean_train_loss
            else:
                ema_train_loss = self.moving_avg_alpha * mean_train_loss + (1 - self.moving_avg_alpha) * ema_train_loss

            # Check EMA of training loss
            if max_train_loss is None:
                max_train_loss = ema_train_loss
            elif max_train_loss - self.train_threshold < ema_train_loss:
                patience += 1
            else:
                patience = 0
                max_train_loss = ema_train_loss
            
            if patience >= self.patience:
                self.scheduler.step()
                patience = 0
                max_train_loss = ema_train_loss

            self.writer.add_scalar('train/ema_loss', ema_train_loss, i)
            self.writer.add_scalar('train/loss', mean_train_loss, i)
            self.writer.add_scalar('train/bce_loss', mean_train_bce_loss, i)
            self.writer.add_scalar('train/dsc_loss', mean_train_dsc_loss, i)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], i)
            
            mean_valid_loss, mean_valid_bce_loss, mean_valid_dsc_loss = self.validate()

            if ema_valid_loss is None:
                ema_valid_loss = mean_valid_loss
            else:
                ema_valid_loss = self.moving_avg_alpha * mean_valid_loss + (1 - self.moving_avg_alpha) * ema_valid_loss

            self.writer.add_scalar('eval/ema_loss', ema_valid_loss, i)
            self.writer.add_scalar('eval/loss', mean_valid_loss, i)
            self.writer.add_scalar('eval/bce_loss', mean_valid_bce_loss, i)
            self.writer.add_scalar('eval/dsc_loss', mean_valid_dsc_loss, i)
            
            tqdm.write(f'[TRAIN: {i + 1}] loss = {ema_train_loss}', file=sys.stdout)
            tqdm.write(f'[EVAL: {i + 1}] loss = {ema_valid_loss}\n', file=sys.stdout)

            # Check EMA of validation loss
            if max_valid_loss is None:
                max_valid_loss = ema_valid_loss
            elif max_valid_loss - self.valid_threshold < ema_valid_loss and self.scheduler.get_last_lr()[0] < 1e-6:
                cutoff += 1
            else:
                cutoff = 0
                max_valid_loss = ema_valid_loss
                saved_model = self.unet.state_dict()

            if i % self.save == 0 or cutoff >= self.valid_cutoff or i == self.total_epochs - 1:
                name = 'final-weights.pt' if cutoff >= self.valid_cutoff or i == self.total_epochs - 1 else f'unet-epoch{i}.pt'
                checkpoint = self.checkpoints / name
                self.unet.cpu()
                torch.save({
                    'epoch': i,
                    'unet_state_dict': self.unet.state_dict() if i % self.save == 0 else saved_model,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': ema_valid_loss,
                    }, checkpoint)
                self.unet.to(self.device)

                if cutoff >= self.valid_cutoff:
                    return
            
            
            

