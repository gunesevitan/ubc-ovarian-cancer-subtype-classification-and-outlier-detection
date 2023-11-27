import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import cv2

sys.path.append('..')
import settings
import image_utilities


if __name__ == '__main__':

    raw_thumbnail_directory = settings.HDD / 'ubc_ocean' / 'train_thumbnails'
    raw_image_directory = settings.HDD / 'ubc_ocean' / 'train_images'
    df = pd.read_csv(settings.DATA / 'raw_datasets' / 'ubc_ocean' / 'train.csv')

    output_image_directory = settings.DATA / 'model_datasets' / 'ubc_ocean' / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)

    metadata = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        image_type = 'tma' if row['is_tma'] else 'wsi'

        if image_type == 'tma':
            file_name = f'{row["image_id"]}.png'
            file_path = raw_image_directory / file_name
        elif image_type == 'wsi':
            file_name = f'{row["image_id"]}_thumbnail.png'
            file_path = raw_thumbnail_directory / file_name

        image_path = str(file_path)
        image = cv2.imread(image_path)
        raw_image_shape = image.shape[:2]

        if image_type == 'tma':

            # Drop low standard deviation rows and columns (white areas with less tissue)
            image = image_utilities.drop_low_std(image=image, threshold=10)

        elif image_type == 'wsi':

            # Crop the largest contour on WSIs
            largest_contour_bounding_box = image_utilities.get_largest_contour(image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold=20)
            image = image[
                largest_contour_bounding_box[1]:largest_contour_bounding_box[3] + 1,
                largest_contour_bounding_box[0]:largest_contour_bounding_box[2] + 1,
                :
            ]

            # Standardize the background
            image[np.all(image == 0, axis=-1)] = 255

        else:
            raise ValueError(f'Invalid image type {image_type}')

        cv2.imwrite(str(output_image_directory / file_name), image)

        processed_image_shape = image.shape[:2]
        settings.logger.info(f'Image ID {row["image_id"]} - Raw Shape: {raw_image_shape} Processed Shape: {processed_image_shape}')

        metadata.append({
            'image_id': row['image_id'],
            'label': row['label'],
            'image_width': processed_image_shape[1],
            'image_height': processed_image_shape[0],
            'image_type': image_type,
            'organ': 'ovary',
            'cancer_subtype': np.nan,
            'dataset': 'ubc_ocean',
            'image_path': output_image_directory / file_name,
        })

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(settings.DATA / 'model_datasets' / 'ubc_ocean' / 'metadata.csv', index=False)
