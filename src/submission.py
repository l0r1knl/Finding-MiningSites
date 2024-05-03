"""
submission.py

This script make a submission file using trained models.

Usage:
    python submission.py --models_info MODELS_INFO_PATH --threshold THRESHOLD [--apply_tta]

Arguments:
    --models_info MODELS_INFO_PATH: Path to the models infomation file (e.g., models_info.csv).
                                    The file should have the following format:

                                    weight,parameter
                                    /path/to/model_1/weights.ckpt,/path/to/model_1/config.yaml
                                    /path/to/model_2/weights.ckpt,/path/to/model_2/config.yaml
                                    ...

    --threshold THRESHOLD: Threshold value to classify for presence or absence of  Minig Sites.
                           Images with predicted probabilities above this threshold will be
                           classified as containing the Minig Sites.

    --apply_tta: If provided, Whether to apply Test Time Augmentation during evaluation.
                 Default is not to apply tta.
Example:
    # Apply TTA
    python submission.py --models_info models_info.csv --threshold 0.5 --apply_tta

    # Don't apply TTA
    python submission.py --models_info --threshold 0.5 models_info.csv

Notes:
    - This script assumes the presence of a `main` function that handles to make submission files process.
    - Submission will be saved in the same directory as the models information file.
"""


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from lightning import seed_everything

from config import CFG
from dataset_hundler import FindingMiningSitesDataModule
from lit_modules import Classifier
from models import get_model
from transform import Transform, predict_proba_tta


def main(trained_models_info_path, threshold, apply_tta):

    trained_models_info = pd.read_csv(trained_models_info_path)
    result = pd.read_csv(
        r"D:\Finding-MiningSites\data\uploadsample.csv",
        header=None
    )
    result.columns = ["image_path", "label"]
    result["predict_proba"] = 0.0
    seed_everything(0, workers=True)

    for row in trained_models_info.itertuples():

        weight = row.weight
        parameter = row.parameter

        cfg = CFG(parameter)

        train_data_path = cfg.dataset_root_dir / cfg.dataset_name
        kfold_data_path = cfg.dataset_root_dir / f"kfold{cfg.kfold:02}.csv"

        data = result.copy()
        data["image_path"] = data.image_path.apply(
            lambda x: Path(
                r"D:\Finding-MiningSites\data\out\features"
            ) / cfg.dataset_name / x
        )

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

        test_dm = FindingMiningSitesDataModule(
            train_data_path / "train.csv",
            kfold_data_path,
            train_transform=transform.train,
            test_transform=transform.test,
            batch_size=1,
            num_workers=0,
            fold_index=cfg.fold,
            predict_data=data
        )

        net = get_model(
            cfg.model_name,
            in_chans=n_features,
            num_classes=cfg.num_classes
        )

        model = Classifier.load_from_checkpoint(
            weight,
            net=net,
            num_classes=cfg.num_classes
        )

        test_dm.setup("predict")
        model.to("cuda")
        model.eval()

        pred, label = [], []
        for x, y in tqdm(test_dm.predict_dataloader()):
            if not apply_tta:
                pred.append(
                    model.predict_proba(x).to("cpu").detach().numpy()[0][0]
                )

            else:
                pred.append(predict_proba_tta(model, x))

            label.append(y.to("cpu").detach().numpy()[0][0])

        result.loc[:, "predict_proba"] += pred

    result["predict"] = (
        (result.loc[:, "predict_proba"] /
         trained_models_info.shape[0]) > threshold
    ).astype(np.uint8)

    # save submission file
    if not apply_tta:
        result[["image_path", "predict"]].to_csv(
            trained_models_info_path.parent /
            f"{trained_models_info_path.stem}_predict_th{threshold}.csv",
            header=None,
            index=None
        )

    else:
        result[["image_path", "predict"]].to_csv(
            trained_models_info_path.parent /
            f"{trained_models_info_path.stem}_predict_th{threshold}_TTA.csv",
            header=None,
            index=None
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--models_info", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--apply_tta", action="store_true")

    args = parser.parse_args()

    main(Path(args.models_info), float(args.threshold), args.apply_tta)
