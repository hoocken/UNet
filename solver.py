from collections import deque
from datetime import datetime
import os
from pathlib import Path

import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from model.loss import UNetLoss
from model.unet import UNet
from data.dataloader import infinite_iterator


class Solver():
    def __init__(self, train_ld, validation_ld, config):
        self.model_dir = config.model_dir

        self.train_iter = infinite_iterator(train_ld)
        self.validation_iter = iter(validation_ld)

        self.lr = config.lr

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet = UNet(config.num_classes).to(self.device)
        self.criteria = UNetLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optimizer, config.decay) # LR decay
        
        self.moving_avg_alpha = config.moving_avg_alpha
        self.train_threshold = config.train_threshold
        self.valid_threshold = config.valid_threshold
        self.patience = config.patience
        self.valid_cutoff = config.valid_cutoff

        self.epoch_length = config.epoch_length
        self.total_epochs = config.total_epochs
        self.start_epoch = 0

        if config.load_state:
            checkpoint = torch.load(config.load_state, weights_only=True)
            self.unet.load_state_dict(checkpoint['unet_state_dict'])
            self.criteria.load_state_dict(checkpoint['criteria_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # LR decay
            self.start_epoch = config.start_epoch

        self.save = config.save_every

        self.checkpoints = Path() / config.model_dir / 'checkpoints' / datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            os.makedirs(self.checkpoints)
        except FileExistsError:
            print(f'{self.checkpoints} already exists!')

        self.writer = SummaryWriter(Path() / config.model_dir / 'logs' / datetime.now().strftime("%Y%m%d-%H%M%S"))

    @torch.no_grad 
    def validate(self):
        ema_valid_loss = None
        for images, labels in self.validation_iter:
            images, labels = images.to(self.device), labels.to(self.device)

            pred = self.unet(images)
            loss = self.criteria(pred, labels)
            if ema_valid_loss:
                ema_valid_loss = self.moving_avg_alpha * loss.item() + (1 - self.moving_avg_alpha) * ema_valid_loss
            else:
                ema_valid_loss = loss.item()

        return ema_valid_loss

    def train(self):
        max_train_loss = None
        max_valid_loss = None

        patience = 0
        cutoff = 0

        print("Start training...")
        for i in range(self.start_epoch, self.total_epochs):
            pbar = tqdm(total=self.epoch_length, ncols=0, desc="Train Epoch")
            ema_train_loss = None

            for _ in tqdm(range(self.epoch_length)):
                images, labels = next(self.train_iter)
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred = self.unet(images)
                loss = self.criteria(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                
                if ema_train_loss:
                    ema_train_loss = self.moving_avg_alpha * loss.item() + (1 - self.moving_avg_alpha) * ema_train_loss
                else:
                    ema_train_loss = loss.item()
                
                pbar.update(1)
                pbar.set_postfix(loss=ema_train_loss)
            
            # Check EMA of training loss
            if max_train_loss is None:
                max_train_loss = ema_train_loss
            elif max_train_loss - self.train_threshold < ema_train_loss:
                patience += 1
            
            if patience >= self.patience:
                self.scheduler.step()
                patience = 0

            self.writer.add_scalar('train/loss', ema_train_loss, i)
            
            # ----------------------------------------------------------
            # Validation
            # ----------------------------------------------------------
            ema_valid_loss = self.validate()
            tqdm.write(f'[EVAL: {i}] loss = {ema_valid_loss}')
            self.writer.add_scalar('eval/loss', ema_valid_loss, i)

            # Check EMA of validation loss
            if max_valid_loss is None:
                max_valid_loss = ema_valid_loss
            elif ema_valid_loss - self.valid_threshold < ema_valid_loss:
                cutoff += 1

            if i % self.save == 0 or cutoff >= self.valid_cutoff:
                checkpoint = self.checkpoints / (f'/unet-epoch{i}.pt' if i % self.save == 0 else '/final-weights.pt')
                self.unet.cpu()
                torch.save({
                    'epoch': i,
                    'unet_state_dict': self.unet.state_dict(),
                    'criteria_state_dict': self.criteria.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': ema_valid_loss,
                    }, checkpoint)
                self.unet.to(self.device)
            
            
            

