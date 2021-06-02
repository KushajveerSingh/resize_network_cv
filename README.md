# Resizer network
This repository contains the PyTorch implementation of the Resizer model proposed in the paper [Learning to Resize Images for Computer Vision Tasks](https://arxiv.org/abs/2103.09950). The model is tested on two datasets: Imagenette and Imagewoof using ResNet-50 as the baseline model. Check the accompanying [blob](TODO) for details on the model.

## Table of Contents
- [Results](#results)
- [Details of config file](#details-of-config-file)
- [Download datasets](#download-datasets)
    - [Imagenette](#imagenette)
    - [Imagewoof](#imagewoof)
- [Reproducing experiments](#reproducing-experiments)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [License](#license)

## Results

<table style="text-align:center">
    <tr>
        <th style="text-align:center"> Dataset </th>
        <th style="text-align:center"> Model </th>
        <th style="text-align:center"> Acc </th>
    </tr>
    <tr>
        <td rowspan=2> Imagenette </td>
        <td> ResNet-50 </td>
        <td> 81.07 </td>
    </tr>
    <tr>
        <td> Resizer + ResNet-50 </td>
        <td> 82.16 </td>
    </tr>
    <tr>
        <td rowspan=2> Imagewoof </td>
        <td> ResNet-50 </td>
        <td> 58.13 </td>
    </tr>
    <tr>
        <td> Resizer + ResNet-50 </td>
        <td> 65.20 </td>
    </tr>
</table>

**Note**:- Due to compute limitation I stopped the training of models early. If you want to get better results increase the number of epochs to `300` and change the learning rate scheduler to reduce learning rate every `50` epochs with a factor of `0.8`.

## Details of config file
If you are unfamiliar with [hydra](https://hydra.cc/), check my blog [Complete tutorial on how to use Hydra in Machine Learning projects](https://kushajveersingh.github.io/blog/general/2021/03/16/post-0014.html) for a quick guide on how to use hydra.

`cfg.data` contains the arguments to load the desired dataset. List of all arguments is shown below
```
data:
  root: ../data     # directory where data is downloaded (not including the folder name)
  name: imagenette2  # "imagenette2" or "imagewoof2" (folder name of dataset inside `root`)
  resizer_image_size: 448  # size of images passed to resizer model
  image_size: 224          # size of images passed to CNN model
  num_classes: 10          # number of labels in training dataset

  # Passed to torch.utils.data.DataLoader
  batch_size: 64
  num_workers: 8
```

The main arguments that you need to adjust are `root` where the dataset is downloaded and `name` (*imagenette2*, *imagewoof2*) which dataset to use for training.

To apply the resizer model use `apply_resizer_model: true` and it will apply the resizer before the base model. The arguments of the resizer are specified in `cfg.resizer`
```
resizer:
  in_channels: 3       # Number of input channels of resizer (for RGB images it is 3)
  out_channels: 3      # Number of output channels of resizer (for RGB images it is 3)
  num_kernels: 16      # Same as `n` in paper
  num_resblocks: 2     # Same as 'r' in paper
  negative_slope: 0.2  # Used by leaky relu
  interpolate_mode: bilinear  # Passed to torch.nn.functional.interpolate
```

`in_channels` and `out_channels` specify the number of input channels to the resizer and the number of channels outputted by resizer respectively. In most scenarios, both thse values should be same (and equal to 3 for RGB images).

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

## Reproducing experiments
The config files to reproduce the experiments are provided in [config_files](config_files) folder. Simply copy the config file to `src/config.yaml` and run `python trainer.py`

- ResNet50 on Imagenette: [config_files/imagenette_resnet50.yaml](config_files/imagenette_resnet50.yaml)
- ResNet50 + Resizer on Imagenette: [config_files/imagenette_resnet50_resizer.yaml](config_files/imagenette_resnet50_resizer.yaml)
- ResNet50 on Imagewoof: [config_files/imagewoof_resnet50.yaml](config_files/imagewoof_resnet50.yaml)
- ResNet50 + resizer on Imagewoof: [config_files/imagewoof_resnet50_resizer.yaml](config_files/imagewoof_resnet50_resizer.yaml)

**Note**:- I trained the models on RTX 2080TI, so you may have to adjust the batch size depending on your GPU.

An example of how to use the config files is shown below (from the root of this repo)
```
cd src

# For Resnet50 on Imagenette
mv ../config_files/imagenette_resnet50.yaml config.yaml
python trainer.py

# For ResNet50 + Resizer on Imagenette
mv ../config_files/imagenette_resnet50_resizer.yaml config.yaml
python trainer.py
```
## Repository structure
- [download_data.sh](download_data.sh) - Script to download `Imagenette` and `Imagewoof` datasets. Usage `./download_data.sh`
- [src](src)
    - [config.yaml](src/config.yaml) - The hydra config file to handle anything in the repository. All the options in the config file are well documented. Check the [Details of Config file](#details-of-config-file) section for all the details about the config file.
    - [data.py](src/data.py) - Contains the code to create `pytorch_lightning.LightningDataModule` for the specified dataset.
    - [models](src/models)
        - [resizer.py](src/models/resizer.py) - Contains the implementation of the Resizer model proposed in the paper
        - [base_model.py](src/models/base_model.py) - It loads torchvision Resnet50 model which is used as the base model in this repo. You can specify your own base model here.
        - [\_\_init\_\_.py](src/models/__init__.py) - Provides a utility function `get_model` to load the above two models by providing the corresponding name (*resizer*, *base_model*)
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