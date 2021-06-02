from omegaconf import DictConfig
import hydra

import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Module
from data import DataModule


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    dm = DataModule(cfg)
    dm.setup()
    model = Module(cfg, dm.val_length)

    if cfg.apply_resizer_model:
        path = hydra.utils.to_absolute_path(cfg.trained_path)
        path = os.path.abspath(path)
        path = os.path.join(path, f"{cfg.data.name}.ckpt")
        ckpt = torch.load(path)['state_dict']
        state_dict = {}
        for k, v in ckpt.items():
            k = k[k.find('.')+1:]
            state_dict[k] = v
        model.base_model.load_state_dict(state_dict)

    cfg = cfg.trainer
    callback = ModelCheckpoint(filename="{epoch}-{val_acc}",
                               monitor='val_acc',
                               save_last=True,
                               mode='max')

    trainer = Trainer(gpus=cfg.gpus,
                      benchmark=True,
                      callbacks=[callback],
                      check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                      max_epochs=cfg.epochs,
                      precision=cfg.precision,
                      gradient_clip_val=cfg.gradient_clip_value)

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
