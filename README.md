# resize_network_cv
PyTorch implementation of the paper "Learning to Resize Images for Computer Vision Tasks" on Imagenette and Imagewoof datasets

## Results

## Details of config file

## Download datasets
`Imagenette` and `Imagewoof` datasets are used. You can learn more about the datasets at [fastai/imagenette](https://github.com/fastai/imagenette). The instructions to download and setup the data are provided below or you can use [download_data.sh](download_data.sh) script to do all of this for you (`./download_data.sh`).

### Imagenette
Download [link](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz) or run the following commands from the root directory of this repo

```
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xzf imagenette2.tgz -C data/
rm imagenette2.tgz
```

### Imagewoof
Download [link](https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz) or run the following commands from the root directory of this repo

```
wget https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz
tar -xzf imagewoof2.tgz -C data/
rm imagewoof2.tgz
```

## Repository structure
- [download_data.sh](download_data.sh) - Script to download `Imagenette` and `Imagewoof` datasets. Usage `./download_data.sh`
- [src](src)
    - [config.yaml](config.yaml) - The hydra config file to handle anything in the repository. All the options in the config file are well documented. Check the [Details of Config files](#details-of-config-files) section for all the details about the config file.
    - [data.py](src/data.py) - Contains the code to create `pytorch_lightning.LightningDataModule` for the specified dataset.
    - [models](src/models)
        - [resizer.py](src/models/resizer.py) - Contains the implementation of the Resizer model proposed in the paper
        - [base_model.py](src/models/base_model.py) - It loads torchvision Resnet50 model which is used as the base model in this repo. You can specify your own base model here.
        - [__init__.py](src/models/__init__.py) - Provides a utility function `get_model` to load the above two models by providing the corresponding name (*resizer*, *base_model*)
    - [model.py](src/model.py) - Contains the code to create `pytorch_lightning.LightningModule`. This loads the above models and specifies the training/validation steps, optimizers, learning rate scheduler
    - [trainer.py](src/trainer.py) - The main python script that you should call to train your models. It reads the arguments from [config.yaml](src/config.yaml) and does the specified training, while saving all the outputs to `outputs/{date}/{time}` directory.

## Requirements
- Python = 3.8.8
- hydra-core = 1.0.6
- matplotlib = 3.4.2
- pytorch = 1.8.1
- torchvision = 0.9.1
- pytorch-lightning = 1.3.3
- torchmetrics = 0.3.2

## License
[Apache License 2.0](LICENSE)