from torchvision.models import resnet50
import torch.nn as nn


def get_base_model(in_channels: int, num_classes: int = 10):
    model = resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)
    model.fc = nn.Linear(2048, num_classes)
    return model
