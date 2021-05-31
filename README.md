# resize_network_cv
PyTorch implementation of the paper "Learning to Resize Images for Computer Vision Tasks" on Imagenette and Imagewoof datasets

## Details of config files

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
    - [config](src/config) - Contains the hydra config files to change anything in the repository. The main config file is [config.yaml](src/config/config.yaml). All the options in the config file are well documented. Check the [Details of Config files](#details-of-config-files) section for all the details about the config files.
    - [data.py](src/data.py) - Contains the code to create `pytorch_lightning.LightningDataModule` for the specified dataset. 
## Requirements
- Python = 3.8.8
- hydra-core = 1.0.6
- matplotlib = 3.4.2
- pyTorch = 1.8.1
- torchvision = 0.9.1
- pytorch-lightning = 1.3.3
- torchmetrics = 0.2.0

## License
[Apache License 2.0](LICENSE)