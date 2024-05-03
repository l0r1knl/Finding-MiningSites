import timm
import torch.nn as nn
from torchvision.models import (
    regnet_y_400mf,
    regnet_x_3_2gf,
    efficientnet_v2_s
)


def get_model(
    model_name: str,
    pretrained: bool = True,
    pretrained_cfg: str | None = None,
    in_chans: int = 3,
    num_classes: int = 1000
) -> nn.Module:
    """Load a pretrained model from PyTorch Image Models (timm) or torchvision.models.

    Args:
        model_name (str): Name of the model to load.
            (e.g., 'resnet50', 'vit_base_patch16_224').
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        pretrained_cfg (str | None): Path to a pretrained configuration file.
            Defaults to None.
        in_chans (int): Number of input channels. Defaults to 3(RGB).
        num_classes (int): Number of output classes. Defaults to 1000.

    Returns:
        nn.Module: The requested model instance.
    """

    if pretrained:
        weights = "IMAGENET1K_V2" if pretrained_cfg is None else pretrained_cfg
    else:
        weights = None

    model_loaded = False
    if model_name == "regnet_y_400mf":
        model = regnet_y_400mf(weights)
        model_loaded = True

    if model_name == "regnet_x_3_2gf":
        model = regnet_x_3_2gf(weights)
        model_loaded = True

    if model_name == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights)
        model_loaded = True

    if model_loaded:
        __edit_model(model, in_chans, num_classes)

    else:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes
        )

    return model


def __edit_model(
    model: nn.Module,
    in_chans: int = 3,
    num_classes: int = 1000
) -> None:
    """Edit the input or output layer of a pre-trained model.

    Args:
        model (nn.Module): Model to edit input and output layers.
        in_chans (int, optional): Number of input channels. Defaults to 3(RGB).
        num_classes (int, optional): Number of output classes. Defaults to 1000.
    """
    architecture = model._get_name()

    if architecture == "RegNet":
        num_ftrs = model.fc.in_features

        if not in_chans == 3:
            model.stem[0] = nn.Conv2d(
                in_channels=in_chans,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            )

        if not num_ftrs == 1000:
            model.fc = nn.Linear(num_ftrs, num_classes)

    if architecture == "EfficientNet":
        num_ftrs = model.classifier[1].in_features
        if not in_chans == 3:
            model.features[0][0] = nn.Conv2d(
                in_channels=in_chans,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            )

        if not num_ftrs == 1000:
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
