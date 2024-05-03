# Finding-MiningSites

This repository contains the work done for a data analysis competition. It includes various modules with functionalities such as model training, evaluation, and generating predictions.

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
- `submission.py`: Script for make submission file  predictions.
- `train.py`: Script for training models.
- `transform.py`: Module for data transformations and augmentations.

## Usage

1. Install the required libraries.
pip install -r requirements.txt

2. Run `features.py` to make some features to train the model that predicta presense or absense for Mining Sites.

3. Edit the configuration file (`config.py`) to adjust the settings for training and data processing.

4. Run `train.py` to train the models.

5. Run `evaluate.py` to evaluate the trained models.

6. Run `submission.py` to generate the final predictions.
