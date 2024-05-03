# Finding-MiningSites

This repository contains work done in the **Finiding Mining Sites** competition at Solafune. It includes various modules for model training, evaluation, and prediction for submission.

## Structure

- `config.py`: Module for defining configurations related to training and data processing.
- `dataset_handler.py`: Module for loading and preprocessing datasets.
- `evaluate.py`: Script for evaluating trained models.
- `features.py`: Script to make some features.
                 Includes modules to make features and some util functions for them.
- `lit_modules.py`: File defining PyTorch Lightning modules.
- `loss.py`: Module defining loss functions.
- `metric.py`: Module defining evaluation metrics.
- `models.py`: Module defining machine learning models.
- `plot.py`: Module for visualization and plotting.
- `submission.py`: Script to predict test data and make submission file.
- `train.py`: Script for training models.
- `transform.py`: Module for data transformations and augmentations.

## Usage

1. Install the required libraries.
pip install -r requirements.txt

2. Run `features.py` to make some features to train the model that predicta presense or absense for Mining Sites.

3. Edit the configuration file (`config.py`) to set for training and data processing.

4. Run `train.py` to train the models.

5. Run `evaluate.py` to evaluate the trained models.

6. Run `submission.py` to make submission file!
