from omegaconf import DictConfig

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy

from models import get_model


class Module(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg.trainer
        if cfg.apply_resizer_model:
            self.resizer_model = get_model('resizer', cfg)
        else:
            self.resizer_model = None

        self.base_model = get_model('base_model', cfg)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        if self.resizer_model is not None:
            x = self.resizer_model(x)
        x = self.base_model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=False,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(-1) == y).sum().item()

        self.log('val_loss', loss, on_step=True)
        return acc

    def validation_epoch_end(self, validation_step_outputs):
        acc = 0
        for pred in validation_step_outputs:
            acc += pred
        acc = acc / len(self.datamodule.valid_dataloader)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.cfg.lr,
                                 weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.8, min_lr=1e-5)

        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
            }
        }
