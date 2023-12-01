import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import pyvips

sys.path.append('..')
import settings


def read_image(image_path):

    """
    Read image using libvips

    Parameters
    ----------
    image_path: str
        Path of the image

    Returns
    -------
    image: numpy.ndarray of shape (height, width, 3)
        Image array
    """

    image = pyvips.Image.new_from_file(image_path, access='sequential')

    return np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=[image.height, image.width, image.bands]
    )


if __name__ == '__main__':

    raw_image_directory = settings.HDD / 'ubc_ocean' / 'train_images'
    df_crops = pd.read_csv(settings.DATA / 'model_datasets' / 'ubc_ocean' / 'train_crops_info.csv')

    output_image_directory = settings.DATA / 'model_datasets' / 'ubc_ocean' / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)

    metadata = []

    for image_id, df_image in tqdm(df_crops.groupby('image_id'), total=df_crops['image_id'].nunique()):

        image = read_image(str(raw_image_directory / f'{image_id}.png'))

        for crop_idx, row in tqdm(df_image.iterrows(), total=df_image.shape[0]):

            crop_id = row['crop_id']
            label = row['label']
            image_path = str(output_image_directory / f'{image_id}_{crop_id}.png')

            image_crop = image[
                row['h1']:row['h2'],
                row['w1']:row['w2'],
                :
            ].copy()

            cv2.imwrite(image_path, image_crop)

            metadata.append({
                'image_id': f'{image_id}_{crop_id}',
                'label': row['label'],
                'image_path': image_path
            })

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(settings.DATA / 'model_datasets' / 'ubc_ocean' / 'metadata.csv')
