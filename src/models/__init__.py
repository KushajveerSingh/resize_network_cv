from omegaconf import DictConfig

from .base_model import get_base_model
from .resizer import Resizer


def get_model(name: str, cfg: DictConfig):
    if name == "resizer":
        return Resizer(cfg)
    elif name == "base_model":
        return get_base_model(cfg.pretrained, cfg.data.num_classes)
    else:
        raise ValueError(f"Incorrect name={name}. The valid options are"
                         "('resizer', 'base_model')")
