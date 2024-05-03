"""
evaluate.py

This script evaluates models using the specified models infomation file.

Usage:
    python evaluate.py --models_info MODELS_INFO_PATH [--apply_tta]

Arguments:
    --models_info MODELS_INFO_PATH: Path to the models infomation file (e.g., models_info.csv).
                                    The file should have the following format:

                                    weight,parameter
                                    /path/to/model_1/weights.ckpt,/path/to/model_1/config.yaml
                                    /path/to/model_2/weights.ckpt,/path/to/model_2/config.yaml
                                    ...

    --apply_tta: If provided, Whether to apply Test Time Augmentation during evaluation.
                 Default is not to apply tta.
Example:
    # Apply TTA
    python evaluate.py --models_info models_info.csv --apply_tta

    # Don't apply TTA
    python evaluate.py --models_info models_info.csv

Notes:
    - This script assumes the presence of a `main` function that handles the evaluation process.
    - Evaluation results will be saved in the same directory as the models information file.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from lightning import seed_everything

from config import CFG
from dataset_hundler import FindingMiningSitesDataModule
from lit_modules import Classifier
from models import get_model
from plot import (
    plot_confusion_matrix,
    plot_classification_report,
)
from transform import Transform, predict_proba_tta


def main(trained_models_info_path: Path, apply_tta: bool):
    """Main function to evaluate the models.

    Args:
        models_info_path (Path): Path to the models information file.
        apply_tta (bool): Whether to apply Test Time Augmentation or not.
    """
    trained_models_info = pd.read_csv(trained_models_info_path)

    result = pd.read_csv(
        r"D:\Finding-MiningSites\data\answer.csv",
        header=None
    )
    result.columns = ["image_path", "label"]
    result["predict_proba"] = 0.0
    result["eval_count"] = 0

    seed_everything(0, workers=True)

    ths = []
    for row in trained_models_info.itertuples():

        weight = row.weight
        parameter = row.parameter

        cfg = CFG(parameter)

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

        data = pd.read_csv(train_data_path / "train.csv")
        kfold_idx = pd.read_csv(kfold_data_path)
        data = pd.concat([data, kfold_idx], axis=1)
        data = data[data.fold == cfg.fold].copy()
        eval_idx = data.index

        eval_dm = FindingMiningSitesDataModule(
            train_data_path / "train.csv",
            kfold_data_path,
            train_transform=transform.train,
            test_transform=transform.test,
            batch_size=1,
            num_workers=0,
            fold_index=cfg.fold,
            predict_data=data.copy()
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

        eval_dm.setup("predict")
        model.to("cuda")
        model.eval()

        pred, label = [], []
        for x, y in tqdm(eval_dm.predict_dataloader()):
            if not apply_tta:
                pred.append(
                    model.predict_proba(x).to("cpu").detach().numpy()[0][0]
                )

            else:
                pred.append(predict_proba_tta(model, x))

            label.append(y.to("cpu").detach().numpy()[0][0])

        th = Path(row.weight).stem.split("=")[-1]

        result.loc[eval_idx, "predict_proba"] += pred
        result.loc[eval_idx, "predict"] = (
            np.array(pred) > float(th)
        ).astype(np.int8)
        result.loc[eval_idx, "eval_count"] += 1

        ths.append(float(th))

    print(result["predict_proba"] / result["eval_count"])
    print(ths)
    print(np.mean(ths))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    plot_classification_report(
        axes[0], 
        result.label, 
        result.predict, 
        fontsize=12
    )
    plot_confusion_matrix(
        axes[1], 
        result.label, 
        result.predict, 
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.close()

    if not apply_tta:
        fig.savefig(
            trained_models_info_path.parent /
            f"{trained_models_info_path.stem}.png"
        )

    else:
        fig.savefig(
            trained_models_info_path.parent /
            f"{trained_models_info_path.stem}_TTA.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--models_info", required=True)
    parser.add_argument("--apply_tta", action="store_true")

    args = parser.parse_args()

    main(Path(args.models_info), args.apply_tta)
