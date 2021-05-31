from omegaconf import DictConfig
import hydra

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Module
from data import DataModule


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    dm = DataModule(cfg)
    model = Module(cfg)

    cfg = cfg.trainer
    callback = ModelCheckpoint(filename='{epoch}-{acc:.4f}',
                               monitor='acc',
                               save_last=True,
                               mode='max')

    trainer = Trainer(gpus=cfg.gpus,
                      benchmark=True,
                      callbacks=[callback],
                      check_val_every_n_epoch=5,
                      max_epochs=cfg.epochs,
                      precision=cfg.precision)

    if cfg.tune.tune:
        trainer.tune(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
