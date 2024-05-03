"""
features.py

This script make features to train the model predict presense or absense for Mining Sites from Sentinel-2 data.

Usage:
    python features.py

Example:
    python features.py

Notes:
    - This script assumes the presence of some `make_xxx` functions  to make and save features.
    - If you want to make a new feature, implement a new function with reference to the `make_baseline` and call that function in `main`.
"""

from pathlib import Path
from tifffile import TiffFile, imsave
from tqdm import tqdm
import numpy as np
import pandas as pd


def calc_statistic(features: np.ndarray) -> pd.DataFrame:
    """Calculate statistics for each channel and return them as a DataFrame.

    Args:
        features (np.ndarray): Input feature array.

    Returns:
        pd.DataFrame: DataFrame containing statistics for each channel.
    """
    mean, std, min, max = [], [], [], []
    for c in tqdm(range(features.shape[-1])):
        mean.append(features[..., c].mean())
        std.append(features[..., c].std())
        min.append(features[..., c].min())
        max.append(features[..., c].max())

    return pd.DataFrame(
        [mean, std, min, max],
        index=["mean", "std", "min", "max"]
    ).T


def save_features(
        feature_name: str,
        output_root_dir: str,
        data: pd.DataFrame,
        features: np.ndarray
) -> None:
    """Save features to disk.

    Args:
        feature_name (str): Name of the feature.
        output_root_dir (str): Root directory to save features.
        data (pd.DataFrame): DataFrame contains the path of image file 
            and its corresponding label.
        features (np.ndarray): Features to be saved.
    """
    basename = data.image_path[0].stem[:-1]

    # --- save features --- #
    out_features_dir = Path(output_root_dir) / "features" / feature_name
    if not out_features_dir.exists():
        out_features_dir.mkdir(exist_ok=True, parents=True)

    image_paths = []
    for i, feature in tqdm(enumerate(features), total=features.shape[0]):
        path = out_features_dir / f"{basename}{i}.tif"
        image_paths.append(path)
        imsave(path, feature)

    if basename[:-1] == "train":
        data.image_path = image_paths
        out_dataset_dir = Path(output_root_dir) / "dataset" / feature_name
        if not out_dataset_dir.exists():
            out_dataset_dir.mkdir(exist_ok=True, parents=True)

        print(f"Save list tiff file...")
        data.to_csv(out_dataset_dir / "train.csv")

        print(f"Calc statistic.")
        calc_statistic(features).to_csv(out_dataset_dir / f"statistic.csv")


def read_tiff(path: Path | str) -> np.ndarray:
    """Reads Tiff image into N dimensional np.ndarray.

    Args:
        path (Path | str): Path of the Tiff image.

    Returns:
        np.ndarray: np.ndarray of image.
    """
    with TiffFile(path) as tif:
        tiff = tif.asarray()

    return tiff


def load_data(data_path: Path, img_dir: Path) -> pd.DataFrame:
    """Load a list of images from a CSV file and return as a DataFrame.

    Args:
        data_path (Path): Path to the CSV file containing the list of data.
        img_dir (Path): Directory where the images are stored.

    Returns:
        pd.DataFrame: DataFrame containing the path of image file 
            and its corresponding label.
    """

    data = pd.read_csv(
        data_path,
        header=None
    )

    if data.shape[1] == 2:
        data.columns = ["image_path", "label"]

    else:
        data.columns = ["image_path"]

    data["image_path"] = img_dir / data.image_path

    return data


def devide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Perform element-wise division of two arrays, replacing zero-division with zero.

    Args:
        a (np.ndarray): Numerator array.
        b (np.ndarray): Denominator array.

    Returns:
        np.ndarray: Result of element-wise division, with zero-division replaced by zero.
    """
    return np.divide(a, b, out=np.zeros(b.shape), where=(b != 0))


def clay_mineral(bands: np.ndarray) -> np.ndarray:
    """Calculate a feature representing clay mineral content from Sentinel-2 data.

    Args:
        bands (np.ndarray): Input Sentinel-2 data.

    Returns:
        np.ndarray: Feature representing clay mineral content.
    """
    swir1 = bands[..., 10]
    swir2 = bands[..., 11]

    cmr = devide(swir1, swir2)

    return cmr


def iron_oxide(bands: np.ndarray) -> np.ndarray:
    """Calculate a feature representing iron oxide content from Sentinel-2 data.

    Args:
        bands (np.ndarray): Input Sentinel-2 data.

    Returns:
        np.ndarray: Feature representing iron oxide content.
    """
    red = bands[..., 3]
    blue = bands[..., 1]

    io = devide((red-blue), (red+blue))

    return io


def ferrous_minerals(bands: np.ndarray) -> np.ndarray:
    """Calculate a feature representing ferrous mineral content from Sentinel-2 data.

    Args:
        bands (np.ndarray): Input Sentinel-2 data.

    Returns:
        np.ndarray: Feature representing ferrous mineral content.
    """
    swir = bands[..., 11]
    nir = bands[..., 7]

    fm = devide((swir - nir), (swir + nir))

    return fm


def nd_vegetation_index(bands: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Vegetation Index (NDVI) from Sentinel-2 data.

    Args:
        bands (np.ndarray): Input Sentinel-2 data.

    Returns:
        np.ndarray: Normalized Difference Vegetation Index.
    """
    red = bands[..., 3]
    nir = bands[..., 7]

    ndvi = devide((nir - red), (nir + red))

    return ndvi


def enmdi(bands: np.ndarray) -> np.ndarray:
    """Calculate Enhanced Normalized Difference Moisture Index from Sentinel-2 data.
    
    Args:
        bands (np.ndarray): Input Sentinel-2 data.

    Returns:
        np.ndarray: Enhanced Normalized Difference Moisture
    
    Notes:
        This implementation is based on the discussion and implementation
        described in [1], by Michal.

        [1]: Michal, "Pytorch - CV, feature engineering and more (LB 0.927+)", Solafune, Publish Date.
             URL: https://example.com/article
             Author's Profile: https://solafune.com/account/Michal

        The author, Michal, can be found at the above profile link.
    """
    nir = bands[..., 7]
    re4 = bands[..., 8]
    swir1 = bands[..., 10]
    swir2 = bands[..., 11]

    nmdi = devide(
        ((nir + re4) - (swir1 + swir2)), ((nir + re4) + (swir1 + swir2))
    )

    return nmdi


def ndoi1(bands: np.ndarray) -> np.ndarray:
    """Calculate a custom feature index from Sentinel-2 data.

    Args:
        bands (np.ndarray): Input Sentinel-2 data.

    Returns:
        np.ndarray: Custom feature index calculated from the input bands.
    """
    swir1 = bands[..., 10]  # 1610
    re4 = bands[..., 8]  # 864

    ndoi1 = devide((swir1 - re4), (swir1 + re4))

    return ndoi1


def ndoi2(bands: np.ndarray) -> np.ndarray:
    """Calculate a custom feature index from Sentinel-2 data.

    Args:
        bands (np.ndarray): Input Sentinel-2 data.

    Returns:
        np.ndarray: Custom feature index calculated from the input bands.
    """
    blue = bands[..., 1]  # 490
    re3 = bands[..., 6]  # 779

    ndoi2 = devide((blue - re3), (blue + re3))

    return ndoi2


def make_baseline(data: pd.DataFrame, out_dir: Path) -> None:
    """Generate and save features of baseline.

    Args:
        data (pd.DataFrame): DataFrame contains the path of image file 
            and its corresponding label.
        out_dir (Path): Directory to save the generated features.
    """
    feature_name = "baseline"

    allbands = np.array(
        [
            read_tiff(row.image_path)
            for row in data.itertuples()
        ]
    )

    save_features(feature_name, out_dir, data, allbands)


def make_baseline_log2(data: pd.DataFrame, out_dir: Path) -> None:
    """Generate and save features of baseline_log2.

    Args:
        data (pd.DataFrame): DataFrame contains the path of image file 
            and its corresponding label.
        out_dir (Path): Directory to save the generated features.
    """
    feature_name = "baseline_log2"

    allbands = np.array(
        [
            read_tiff(row.image_path)
            for row in data.itertuples()
        ]
    )

    # log transformation.
    features = np.log2((allbands + 0.5))

    save_features(feature_name, out_dir, data, features)


def make_alpha_features_v0(data: pd.DataFrame, out_dir: Path) -> None:
    """Generate and save features of alpha_features_v0.

    Args:
        data (pd.DataFrame): DataFrame contains the path of image file 
            and its corresponding label.
        out_dir (Path): Directory to save the generated features.
    """
    feature_name = "alpha_features_v0"

    allbands = np.array(
        [
            read_tiff(row.image_path)
            for row in data.itertuples()
        ]
    )

    n, h, w, b = allbands.shape

    # Feature engineering for bands.
    band_features_values = np.zeros((n, h, w, 5), np.float32)
    band_features_values[..., 0] = enmdi(allbands)
    band_features_values[..., 1] = nd_vegetation_index(allbands)
    band_features_values[..., 2] = np.log2(clay_mineral(allbands) + 0.5)
    band_features_values[..., 3] = ferrous_minerals(allbands)
    band_features_values[..., 4] = iron_oxide(allbands)

    # Add band features to existing 12 bands.
    features = np.zeros((n, h, w, b + 5), np.float32)

    # log transformation.
    features[..., : b] = np.log2((allbands + 0.5))
    features[...,  b:] = band_features_values

    save_features(feature_name, out_dir, data, features)


def make_alpha_features_v1(data: pd.DataFrame, out_dir: Path) -> None:
    """Generate and save features of alpha_features_v0.

    Args:
        data (pd.DataFrame): DataFrame contains the path of image file 
            and its corresponding label.
        out_dir (Path): Directory to save the generated features.
    """
    feature_name = "alpha_features_v1"

    allbands = np.array(
        [
            read_tiff(row.image_path)
            for row in data.itertuples()
        ]
    )

    n, h, w, b = allbands.shape

    # Feature engineering for bands.
    band_features_values = np.zeros((n, h, w, 4), np.float32)
    band_features_values[..., 0] = enmdi(allbands)
    band_features_values[..., 1] = nd_vegetation_index(allbands)
    band_features_values[..., 2] = ndoi1(allbands)
    band_features_values[..., 3] = ndoi2(allbands)

    # Add band features to existing 12 bands.
    features = np.zeros((n, h, w, b + 4), np.float32)

    # log transformation.
    features[..., : b] = np.log2((allbands + 0.5))
    features[...,  b:] = band_features_values

    save_features(feature_name, out_dir, data, features)


if __name__ == "__main__":
    data_dir = Path(r"D:\Finding-MiningSites\data")
    output_dir = data_dir / "out"

    data_paths = [data_dir / "answer.csv", data_dir / "uploadsample.csv"]
    img_dirs = [data_dir / "train", data_dir / "evaluation_images"]

    for data_path, img_dir in zip(data_paths, img_dirs):

        data = load_data(data_path, img_dir)

        make_baseline(data.copy(), output_dir)
        make_baseline_log2(data.copy(), output_dir)
        make_alpha_features_v0(data.copy(), output_dir)
        make_alpha_features_v1(data.copy(), output_dir)
