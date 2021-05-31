from hydra.experimental import initialize, compose
from data import DataModule
from models import get_model


if __name__ == "__main__":
    initialize(config_path="config", job_name="test_app")
    cfg = compose(config_name="config")

    dm = DataModule(cfg)
    dm.setup()
    loader = iter(dm.train_dataloader())
    batch = next(loader)
    img, y = batch

    resizer = get_model('resizer', cfg)
    base_model = get_model('base_model', cfg)
