from omegaconf import DictConfig

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from models import get_model


class Module(LightningModule):
    def __init__(self, cfg: DictConfig, val_length):
        super().__init__()
        self.cfg = cfg.trainer
        self.val_length = val_length
        if cfg.apply_resizer_model:
            self.resizer_model = get_model('resizer', cfg)
        else:
            self.resizer_model = None

        self.base_model = get_model('base_model', cfg)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        if self.resizer_model is not None:
            x = self.resizer_model(x)
        x = self.base_model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = (y_hat.argmax(-1) == y).sum().item()
        return acc

    def validation_epoch_end(self, validation_step_outputs):
        acc = 0
        for pred in validation_step_outputs:
            acc += pred
        acc = acc / self.val_length
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.cfg.lr,
                                momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=50, gamma=0.8)

        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
