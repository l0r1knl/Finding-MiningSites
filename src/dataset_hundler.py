from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import lightning as L
from sklearn.model_selection import StratifiedKFold
from tifffile import TiffFile
import torch
from torch import Tensor
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms


def kfold_dataset(
    train_data_path: Path | str,
    output_dir: Path,
    k: int = 5,
    random_state: int = 42
) -> None:
    """Saves the index from the train data list for split into train and 
    validation data.

    Args:
        train_data_path (Path | str): Train data list.
            it's provided as answer.csv
        output_dir (Path): Save path of KFold Index
        k (int, optional): Number of folds. Defaults to 5.
        random_state (int, optional): To control random numbers. Defaults to 42.
    """

    if not output_dir.exists():
        output_dir.mkdir()

    train_data = pd.read_csv(train_data_path, header=None)
    train_data.columns = ["image_path", "label"]
    train_data["fold"] = None

    skf = StratifiedKFold(
        n_splits=k,
        random_state=random_state,
        shuffle=True
    )
    for i, (_, valid_index) in enumerate(skf.split(train_data.image_path, train_data.label)):
        train_data.loc[valid_index, "fold"] = i

    train_data.loc[:, ["fold"]].to_csv(
        output_dir / f"kfold{k:02}.csv", index=False
    )


class FindingMiningSitesDataset(Dataset):
    """A Dataset class for a FindingMiningSites dataset.

    Attributes:
        data (pd.DataFrame): DataFrame contains the path of image file 
            and its corresponding label.
        image_paths (np.ndarray): An array stores the path of image file.
        labels (Tensor): Tensor stores labels corresponding to each image file.
        transform (transforms.Transform): It transforms input data.
        device (str, optional): Specify the device for the dataset. 
            Defaults to "cuda".
    """

    def __init__(
        self,
        data: pd.DataFrame,
        transform: transforms.Transform,
        device: str = "cuda"
    ) -> None:
        """Initializes an instance of the FindingMiningSitesDataset class.

        Args:
            data (pd.DataFrame): DataFrame to be used as the dataset.
            transform (transforms.Transform): It transforms input data.
            device (str, optional): Specify the device for the data set. 
                Defaults to "cuda".
        """

        super().__init__()
        self.device = device
        self.data: pd.DataFrame = data
        self.image_paths = data["image_path"].values
        self.labels: Tensor = self.__get_labels()
        self.transform = transform.to(self.device)

    def __read_tiff_image(self, path: Path | str) -> Tensor:
        """Reads Tiff image into N dimensional Tensor.

        Args:
            path (Path | str): Path of the Tiff image.

        Returns:
            Tensor[image_channels, image_height, image_width]: Tensor of image.
        """
        with TiffFile(path) as tif:
            tiff = tif.asarray()

        tiff = np.transpose(tiff, (2, 0, 1))

        return torch.from_numpy(tiff).clone().to(self.device)

    def __get_labels(self) -> Tensor:
        """Get label stored in DataFrame with Tensor type.

        Returns:
            Tensor[N, 1]: Tensor of labels.
        """
        try:
            labels = torch.from_numpy(
                self.data["label"].values[:, np.newaxis]
            ).clone()

        except KeyError:
            labels = torch.zeros(self.data.shape[0])

        return labels.to(self.device)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return the image and its target corresponding to the specified index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            tuple[Tensor, Tensor]: The image and its corresponding target.
        """
        image = self.__read_tiff_image(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        """Return the total number of data in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.data.shape[0]


class FindingMiningSitesDataModule(L.LightningDataModule):
    """A PyTorch Lightning DataModule for FindingMiningSites dataset.

    This DataModule manages training and test datasets, handles data preprocessing,
    data loading, and KFold cross-validation settings.

    Args:
        train_data_path (Path): Path to the CSV containing the list of train data.
        kfold_data_path (Path): Path to the KFold index list.
        train_transform (Transform): Transform applied to train data.
        test_transform (Transform): Transform applied to validation/test data.
        fold_index (int): Index of the KFold to use.
        batch_size (int): Size of the data batch.
        num_workers (int): Number of subprocesses used by the data loader.
        predict_data: (None | pd.DataFrame) DataFrame for test or predict.
    """

    def __init__(
        self,
        train_data_path: Path,
        kfold_data_path: Path,
        train_transform: transforms.Transform,
        test_transform: transforms.Transform,
        fold_index: int = 0,
        batch_size: int = 32,
        num_workers: int = 0,
        predict_data: None | pd.DataFrame = None,
    ) -> None:

        super().__init__()

        self.train_data_path = train_data_path
        self.kfold_data_path = kfold_data_path
        self.columns = ["image_path", "label"]

        self.fold_index = fold_index
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.predict_data = predict_data

    def setup(self, stage: Literal["fit", "test", "predict", None] = None) -> None:
        """Sets up the DataModule according to the given stage.

        This method prepares the datasets for the 'fit' or 'test' stage.
        For the 'fit' stage, it creates the train and validation datasets,
        and for the 'test' and 'predict' stage, it prepares the dataset.

        Args:
            stage (Literal['fit', 'test', 'predict', None]): The stage to set up.
        """
        if stage == "fit":
            train_data = pd.read_csv(self.train_data_path)
            kfold = pd.read_csv(self.kfold_data_path)
            train_data["fold"] = kfold["fold"]

            train_idx = train_data[train_data.fold != self.fold_index].index
            valid_idx = train_data[train_data.fold == self.fold_index].index

            self.train_dataset = FindingMiningSitesDataset(
                train_data.loc[train_idx, self.columns].reset_index(drop=True),
                self.train_transform,
            )

            self.valid_dataset = FindingMiningSitesDataset(
                train_data.loc[valid_idx, self.columns].reset_index(drop=True),
                self.test_transform,
            )

        if stage == "test":
            self.test_dataset = FindingMiningSitesDataset(
                self.predict_data,
                self.test_transform,
            )

        if stage == "predict":
            self.predict_dataset = FindingMiningSitesDataset(
                self.predict_data,
                self.test_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns Train DataLoader.

        Returns:
            DataLoader: Train DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns Validation DataLoader.

        Returns:
            DataLoader: Valdation DataLoader
        """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns Test DataLoader.

        Returns:
            DataLoader: Test DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,

        )

    def predict_dataloader(self) -> DataLoader:
        """Returns Predict DataLoader.

        Returns:
            DataLoader: Predict DataLoader
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,

        )
