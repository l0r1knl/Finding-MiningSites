from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


def remove_axes_decorations(ax: Axes) -> None:
    """Remove axis decorations (frame and ticks) from a matplotlib plot.

    This function removes the frame and ticks from the specified matplotlib
    Axes object, while leaving other elements such as the title and color bar
    intact.

    Args:
        ax (Axes): The Axes object from which to remove the decorations.

    Example:
        >>> import matplotlib.pyplot as plt

        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
        >>> ax.set_title('Sample Plot')

        >>> remove_axes_decorations(ax)
        >>> plt.close()
        >>> display(fig)
    """
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_image(
    fig: Figure,
    ax: Axes,
    image: np.ndarray,
    title: str | None = None,
    cmap: str = "viridis",
    crange: tuple | None = None,
    plot_cbar: bool = True,
    axes_decorations: bool = False,
) -> None:
    """Plot an image using Matplotlib.

    Args:
        fig (matplotlib.figure.Figure, optional): Figure object to plot on.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on.
        image (numpy.ndarray): Image data to be plotted (height, width, channels).
        title (str, optional): Title of the image plot.
        cmap (str, optional): Colormap to use for the image.
        crange (tuple, optional) : Range of values to use for the colormap. 
            If None, the full range of the image is used.
        plot_cbar (bool): Whether to plot a colorbar for the image.
        axes_decorations (bool): Whether to plot a axes decorations for the image.

    Example:
        >>> import numpy as np
        >>> image = np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)
        >>> fig, ax = plot_image(image, title='Random RGB Image')
        >>> plt.show()
    """
    im = ax.imshow(
        image,
        cmap=cmap if plot_cbar else None,
    )

    channels = image.shape[2] if len(image.shape) == 3 else 1

    if title:
        ax.set_title(title)

    if plot_cbar and not (channels == 3):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size="7.5%",
            pad=0.05
        )
        fig.colorbar(im, ax=ax, cax=cax)

    if not (axes_decorations):
        remove_axes_decorations(ax)

    if crange:
        im.set_clim(crange)


def plot_sentinel_allbands(
    bands: np.ndarray,
    title: str | None = None,
    min: float = None,
    max: float = None,
    tight_layout: bool = False,
) -> Figure:
    """Plot all bands in Sentinel-2 using Matplotlib.

    Args:
        image (numpy.ndarray): Sentinel-2 data to be plotted (height, width, channels).
        title (str, optional): Title of the plot.
        min (float, optional): Min value of color bar.
        max (float, optional): Max value of color bar.
        plot_cbar (bool): Whether to apply tight_layout to figure object.

    Returns:
        fig (Figure): Figure object containing the plotted all bands in Sentinel-2.

    """
    band_names = [
        "B1", "B2", "B3", "B4", "B5", "B6",
        "B7", "B8", "B8A", "B9", "B11", "B12"
    ]

    fig, axes = plt.subplots(2, 6, figsize=(3.5*6, 3.5*2))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        if min and max:
            crange = min[i], max[i]
        plot_image(
            fig,
            ax,
            bands[:, :, i],
            crange=crange,
            title=band_names[i]
        )

    plt.close()

    if tight_layout:
        fig.tight_layout(
            rect=[0, 0, 1, 0.96] if title else None
        )
        fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1
        )

    return fig


def plot_classification_report(
    ax: Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
    fontsize: int = 16
) -> None:
    """Plots a classification report.

    Args:
        ax (Axes): Matplotlib Axes object to plot the classification report on.
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        labels (list[str], optional): List of class labels. 
            If not provided, numerical class values will be used.
        fontsize (int, optional): Font size for the labels and values in the plot.

    Returns:
        None

    Example:
        import matplotlib.pyplot as plt

        # Assuming y_true and y_pred are already computed
        fig, ax = plt.subplots()
        class_labels = ['class_0', 'class_1', 'class_2']
        plot_classification_report(ax, y_true, y_pred, labels=class_labels)
        plt.show()
    """
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names=labels,
    ) -> pd.DataFrame:

        cr = pd.DataFrame(
            classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                output_dict=True,
            )
        ).transpose()

        supports = ("\n(" + cr.iloc[:-3, -1].astype(int).astype(str) + ")")
        cr.index = (
            (cr.index[:-3] + supports).to_list()
            + cr.index[-3:].to_list()
        )
        cr = cr.iloc[:, :-1].copy()
        cr.iloc[-3:, :-1] = np.nan

        return cr

    cr = get_classification_report(y_true, y_pred, labels)

    sns.heatmap(
        cr.iloc[:-2, :],
        vmin=0.0,
        vmax=1.0,
        cmap=sns.color_palette("Blues", 24),
        annot=True,
        linewidths=2.0,
        fmt="0.4g",
        annot_kws=dict(fontsize=fontsize),
        square=True,
        ax=ax,
    )

    ax.tick_params(axis="x", labelsize=fontsize * 1.10)
    ax.tick_params(axis="y", labelsize=fontsize * 1.10)

    return None


def plot_confusion_matrix(
    ax: Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] = None,
    fontsize: int = 16,
) -> None:
    """Plots a confusion matrix.

    Args:
        ax (Axes): Matplotlib Axes object to plot the confusion matrix on.
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        labels (list[str], optional): List of class labels. 
            If not provided, numerical class values will be used.
        fontsize (int, optional): Font size for the labels and values in the plot.

    Returns:
        None

    Example:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        # Assuming y_true and y_pred are already computed
        fig, ax = plt.subplots()
        class_labels = ['class_0', 'class_1', 'class_2']
        plot_confusion_matrix(ax, y_true, y_pred, labels=class_labels)
        plt.show()
    """
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=labels,
        columns=labels,
    )

    sns.heatmap(
        cm,
        cmap=sns.color_palette("Blues", 24),
        annot=True,
        linewidths=2.0,
        fmt="g",
        annot_kws=dict(fontsize=fontsize),
        square=True,
        ax=ax,
    )

    ax.set_ylim(len(cm), 0)
    ax.tick_params(axis="x", labelsize=fontsize * 1.10)
    ax.tick_params(axis="y", labelsize=fontsize * 1.10)

    padding = fontsize * (0.2) * 4.0 
    ax.set_xlabel("Predict", fontsize=fontsize * 1.20, labelpad = padding)
    ax.set_ylabel("Actual", fontsize=fontsize * 1.20, labelpad = padding)
