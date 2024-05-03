import torch.nn as nn
import torchvision
from torch import Tensor
import lightning as L

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms


class Transform():
    """A class to manage transformations for train and validation / test.

    This class allows you to define transformations for different versions,
    and generates appropriate transformations for training and testing based
    on the given version.

    Args:
        version (int): Version of transformations to use.
        resize (tuple[int, int]): Image size to be used in the model.
        mean (list[float]): List of mean value for each channel of the image.
        std (list[float]): List of std value for each channel of the image.
        probability (float): Probability of applying data expansion.
            Defaults to 0.5.

    Attributes:
        train (nn.Module): Transformations for train.
        test (nn.Module): Transformations for test.

    Example:
        >>> transform= Transform(version=1)
        >>> train_dataset = MyDataset(train_data, transform=transform.train)
        >>> test_dataset = MyDataset(test_data, transform=transform.test)
    """

    def __init__(
        self,
        version: int,
        resize: tuple[int, int],
        mean: list,
        std: list,
        probability: float = 0.5
    ) -> None:

        self.version = version
        self._resize = resize
        self._mean = mean
        self._std = std
        self._p = probability
        self.train: nn.Module = self._get_train_transforms()
        self.test: nn.Module = self._get_test_transforms()

    def _get_train_transforms(self) -> nn.Module:
        """Get train transforms based on the version.

        Returns:
            nn.Module: Requested version of train transform.
        """
        if self.version == 1:
            return nn.Sequential(
                transforms.Resize(
                    self._resize,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.RandomHorizontalFlip(p=self._p),
                transforms.RandomVerticalFlip(p=self._p),
                transforms.ConvertDtype(),
                transforms.Normalize(mean=self._mean, std=self._std),
            )

        if self.version == 2:
            return nn.Sequential(
                transforms.Resize(
                    self._resize,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.RandomHorizontalFlip(p=self._p),
                transforms.RandomVerticalFlip(p=self._p),
                transforms.ConvertDtype(),
            )

        if self.version == 3:
            return nn.Sequential(
                transforms.RandomRotation(degrees=90),
                transforms.RandomHorizontalFlip(p=self._p),
                transforms.RandomVerticalFlip(p=self._p),
                transforms.ConvertDtype(),
            )

        if self.version == 4:
            return nn.Sequential(
                transforms.Resize(
                    self._resize,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.RandomRotation(degrees=90),
                transforms.RandomHorizontalFlip(p=self._p),
                transforms.RandomVerticalFlip(p=self._p),
                transforms.ConvertDtype(),
            )

    def _get_test_transforms(self) -> nn.Module:
        """Get test transforms based on the version.

        Returns:
            nn.Module: Requested version of test transform.
        """
        if self.version == 1:
            return nn.Sequential(
                transforms.Resize(
                    self._resize,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.ConvertDtype(),
                transforms.Normalize(mean=self._mean, std=self._std),
            )

        if self.version == 2:
            return nn.Sequential(
                transforms.Resize(
                    self._resize,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.ConvertDtype(),
            )

        if self.version == 3:
            return nn.Sequential(
                transforms.ConvertDtype(),
            )

        if self.version == 4:
            return nn.Sequential(
                transforms.Resize(
                    self._resize,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.ConvertDtype(),
            )


def predict_proba_tta(model:L.LightningModule, x: Tensor) -> float:
    hflipper = transforms.RandomHorizontalFlip(1.0)
    vflipper = transforms.RandomVerticalFlip(1.0)
    rotator = transforms.RandomRotation(degrees=90)

    x_list = [
        x,
        hflipper(x),
        vflipper(x),
        vflipper(hflipper(x)),
        rotator(x),
        rotator(hflipper(x)),
        rotator(vflipper(x)),
        rotator(vflipper(hflipper(x))),
    ]

    weights = [
        0.300,
        0.100,
        0.100,
        0.100,
        0.100,
        0.100,
        0.100,
        0.100,
    ]

    proba = 0.0
    for _x, _w in zip(x_list, weights):
        proba += (
            model.predict_proba(_x).to("cpu").detach().numpy()[0][0] * _w
        )

    return proba
