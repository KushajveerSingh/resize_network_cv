from omegaconf import DictConfig

from .base_model import get_base_model
from .resizer import Resizer


def get_model(name: str, cfg: DictConfig):
    if name == "resizer":
        return Resizer(cfg)
    elif name == "base_model":
        if cfg.apply_resizer_model:
            in_channels = cfg.resizer.out_channels
        else:
            in_channels = cfg.resizer.in_channels
        return get_base_model(in_channels, cfg.data.num_classes)
    else:
        raise ValueError(f"Incorrect name={name}. The valid options are"
                         "('resizer', 'base_model')")
