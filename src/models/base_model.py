from torchvision.models import resnet50
from torch.nn import Linear


def get_base_model(pretrained: bool = False, num_classes: int = 10):
    model = resnet50(pretrained)
    model.fc = Linear(2048, num_classes)
    return model
