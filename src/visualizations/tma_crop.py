import sys
from tqdm import tqdm
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('..')
import settings


def visualize_crop_by_stds(image, title, std_threshold=10.0, path=None):

    """
    Visualize vertical and horizontal standard deviations of an image, the image and its cropped version

    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, channel)
        Image array

    std_threshold: float
        Standard deviation threshold to drop row and columns

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig = plt.figure(figsize=(16, 12), dpi=100)
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    vertical_stds = image.std(axis=(1, 2))
    horizontal_stds = image.std(axis=(0, 2))
    cropped_image = image[vertical_stds > std_threshold, :, :]
    cropped_image = cropped_image[:, horizontal_stds > std_threshold, :]

    ax1.plot(vertical_stds, label='Vertical Stds')
    ax1.plot(horizontal_stds, label='Horizontal Stds')
    ax1.axhline(std_threshold, label=f'Std Threshold: {std_threshold}', color='r', linewidth=2, linestyle='--')
    ax2.imshow(image)
    ax3.imshow(cropped_image)
    for i in range(2):
        ax1.tick_params(axis='x', labelsize=15)
        ax1.tick_params(axis='y', labelsize=15)

    ax1.set_xlabel('Column/Row', size=15, labelpad=12.5)
    ax1.set_ylabel('Std', size=15, labelpad=12.5)
    ax1.legend(loc='lower right', prop={'size': 14})
    ax1.set_title(title, size=15, pad=12.5)
    ax2.set_title(f'Image Shape: {image.shape}', size=15, pad=12.5)
    ax3.set_title(f'Cropped Image Shape: {cropped_image.shape}', size=15, pad=12.5)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':

    raw_image_directory = settings.HDD / 'ubc_ocean' / 'train_images'
    df_train = pd.read_csv(settings.HDD / 'ubc_ocean' / 'train.csv')

    output_directory = settings.EDA / 'crop_tma'
    output_directory.mkdir(parents=True, exist_ok=True)

    df_train_tmas = df_train.loc[df_train['is_tma'], :].reset_index(drop=True)

    for idx, row in tqdm(df_train_tmas.iterrows(), total=df_train_tmas.shape[0]):

        image_path = str(raw_image_directory / f'{row["image_id"]}.png')
        image = cv2.imread(image_path)

        visualize_crop_by_stds(
            image=image,
            std_threshold=10,
            title=f'Image ID: {row["image_id"]} - Label: {row["label"]} - Type: TMA',
            path=output_directory / f'{row["image_id"]}_crop.png'
        )
