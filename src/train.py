"""
train.py

This script trains a model using the specified configuration file and fold number.

Usage:
    python train.py -c CONFIG_PATH -f FOLD_NUMBER

Arguments:
    -c, --config CONFIG_PATH: Path to the configuration file (e.g., config.yaml).
    -f, --fold FOLD_NUMBER: The fold number to use for training (e.g., 0, 1, 2).

Example:
    python train.py -c configs/config.yaml -f 0

Notes:
    - The configuration file should contain all the necessary parameters for training.
    - The fold number is used to split the data into train and validation sets.
    - This script assumes the presence of a `main` function that handles the training process.
"""

import argparse

import pandas as pd
import seaborn as sns
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from config import CFG
from dataset_hundler import FindingMiningSitesDataModule
from lit_modules import Classifier
from models import get_model
from transform import Transform


def main(config_path, fold):
    """Main function to train the model.

    Args:
        config_path (str): Path to the configuration file.
        fold (int): The fold number to use for training.
    """
    # ----- initialize train parameters ----- #
    cfg = CFG(config_path)
    cfg.fold = fold
    seed_everything(cfg.RS, workers=True)
    # ----- --------------------------- ----- #

    train_data_path = cfg.dataset_root_dir / cfg.dataset_name
    kfold_data_path = cfg.dataset_root_dir / f"kfold{cfg.kfold:02}.csv"

    status = pd.read_csv(train_data_path / "statistic.csv")
    mean = status["mean"]
    std = status["std"]
    n_features = status.shape[0]

    transform = Transform(
        cfg.transform_version,
        cfg.resize,
        mean,
        std
    )

    net = get_model(
        cfg.model_name,
        in_chans=n_features,
        num_classes=cfg.num_classes,
    )

    output_dir = f"./models/{cfg.model_name}/{train_data_path.stem}/kfold5-{fold}/"

    model = Classifier(
        net=net,
        lr=cfg.learning_rate,
        loss_function=cfg.loss_function,
        loss_function_kwargs=cfg.loss_function_kwargs,
        optimizer_kwargs=cfg.optimizer_kwargs,
        scheduler_kwargs=cfg.scheduler_kwargs,
        task=cfg.task,
        num_classes=cfg.num_classes,
    )

    print(f"train transform\n{transform.train}")
    print(f"test transform\n{transform.test}")

    fms_dm = FindingMiningSitesDataModule(
        train_data_path / "train.csv",
        kfold_data_path,
        train_transform=transform.train,
        test_transform=transform.test,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        fold_index=fold,
    )

    trainer = Trainer(
        max_epochs=cfg.epoch,
        accelerator="gpu",
        logger=CSVLogger(save_dir=output_dir),
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="val_loss",
                filename="{epoch:02d}-{val_loss:.03f}",
                save_top_k=2,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_accuracy",
                filename="{epoch:02d}-{val_accuracy:.03f}",
                save_top_k=2,
                mode="max",
            ),
            ModelCheckpoint(
                monitor="val_f1score",
                filename="{epoch:02d}-{val_f1score:.03f}",
                save_top_k=2,
                mode="max",
            ),
            ModelCheckpoint(
                monitor="val_best_f1score",
                filename="{epoch:02d}-{val_best_f1score:.03f}-{threshold:.03f}",
                save_top_k=2,
                mode="max",
            )
        ],
        deterministic=True
    )

    trainer.fit(model, fms_dm)

    # History Plot
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]

    metrics.set_index("epoch", inplace=True)
    lr = metrics.filter(like="lr-").ffill()
    losses = metrics.loc[:, ["train_loss", "val_loss"]]
    scores = metrics.loc[
        :,
        ["val_accuracy", "val_f1score", "val_best_f1score"]
    ]

    lr_history = sns.relplot(data=lr, kind="line", height=6, aspect=1.5)
    loss_history = sns.relplot(data=losses, kind="line", height=6, aspect=1.5)
    score_history = sns.relplot(data=scores, kind="line", height=6, aspect=1.5)
    score_history.set(ylim=(0.55, 1.025))

    lr_history.savefig(f"{trainer.logger.log_dir}/lr_history.png")
    score_history.savefig(f"{trainer.logger.log_dir}/score_history.png")
    loss_history.savefig(f"{trainer.logger.log_dir}/loss_history.png")

    cfg.save(f"{trainer.logger.log_dir}/parameters.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-f", "--fold")

    args = parser.parse_args()

    main(args.config, int(args.fold))
