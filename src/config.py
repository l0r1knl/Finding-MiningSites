import yaml
from pathlib import Path
from typing import Literal

import torch


class CFG:
    """A class to manage hyperparameters for machine learning.

    This class reads hyperparameters from a yaml file and expands them as instance variables.

    Attributes:
        BATCH_SIZE (int): Size of the data batch.
        NUM_WORKERS (int): Number of subprocesses used by the data loader.
        RS (int): Seed value for the random number generator.
        kfold (int): Number of folds used for cross-validation.
        dataset_name (str): Name of the dataset to use.
        dataset_root_dir (Path | str): Path of Dataset's root directory.
        transform_version (int): Version of preprocessing and data augmentation.
        model_name (str): Name of the model to use.
        resize (tuple[int, int]): Image size to be used in the model.
        task (Literal['binary']): Select a classification task.
        num_classes (int): Number of classes.
        epoch (int): Number of epoch.
        optimizer (Literal['AdamW']): The name of the optimizer.
        learning_rate (float): Learning rate for the optimizer.
        optimizer_kwargs (dict): Keyword argument for optimizer.
        scheduler (Literal['CosineAnnealingLR']): The name of the learning scheduler.
        scheduler_kwargs (dict): Keyword argument for scheduler.
        loss_function (Literal['CE', 'BCE']): The name of the loss funcion.
        loss_function_kwargs (dict): Keyword argument for loss function.

    Example:
        >>> config = CFG('./config/001.yaml')
        >>> print(config.BATCH_SIZE)

    Note:
        The format of the yaml file is as follows Be sure to include all Attributes.

        ```yaml
        BATCH_SIZE: 32
        dataset_name: xxxx
        scheduler: CosineAnnealingLR
        scheduler_kwargs:
          eta_min: 0.000001
          T_max: 4960
        ```
    """

    def __init__(
        self,
        yaml_path: Path | str,
        device: Literal["cpu", "cuda"] = "cuda"
    ) -> None:
        self.__dict__.update(**self.load(yaml_path))
        self.train()
        self.device = device

    def train(self) -> None:
        """Converts type of a specific parameter for train / validation.
        """
        self.dataset_root_dir = Path(self.dataset_root_dir)
        self.resize = tuple(self.resize)

        for k in ["weight", "pos_weight"]:
            try:
                self.loss_function_kwargs[k] = torch.Tensor(
                    self.loss_function_kwargs[k]
                ).to("cuda")

            except KeyError:
                pass

    def format_save(self) -> None:
        """Converts type of a specific parameter for Save.
        """
        self.dataset_root_dir = str(self.dataset_root_dir)
        self.resize = list(self.resize)

        for k in ["weight", "pos_weight"]:
            try:
                self.loss_function_kwargs[k] = (
                    self.loss_function_kwargs[k]
                    .to("cpu")
                    .detach()
                    .numpy()
                    .copy()
                    .tolist()
                )
            except KeyError:
                pass

    def load(self, filepath: str) -> None:
        """Reads the yaml file.

        Args:
            filepath (str): Path of configuration file.
        """
        return yaml.load(
            Path(filepath).read_text(),
            Loader=yaml.Loader
        )

    def save(self, output_path: Path | str) -> None:
        """Saves hyperparameters to yaml file.

        Args:
            output_path (Path | str): Path to save yaml file.
        """
        with open(output_path, "w") as file:
            self.format_save()
            yaml.safe_dump(
                self.__dict__,
                file,
                default_flow_style=False,
                sort_keys=False
            )
            self.train()
