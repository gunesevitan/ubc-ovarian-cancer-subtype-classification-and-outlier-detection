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

    raw_image_directory = settings.HDD / 'ubc_ocean' / 'train_images'
    df = pd.read_csv(settings.DATA / 'raw_datasets' / 'ubc_ocean' / 'train.csv')

    output_image_directory = settings.DATA / 'model_datasets' / 'ubc_ocean' / 'images'
    output_image_directory.mkdir(parents=True, exist_ok=True)

    output_instance_directory = settings.DATA / 'model_datasets' / 'ubc_ocean' / 'instances'
    output_instance_directory.mkdir(parents=True, exist_ok=True)

    wsi_longest_edge = 8192
    metadata = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        image_type = 'tma' if row['is_tma'] else 'wsi'

        image_path = str(raw_image_directory / f'{row["image_id"]}.png')
        image = cv2.imread(image_path)
        raw_image_shape = image.shape[:2]

        if image_type == 'tma':

            # Drop low standard deviation rows and columns (white areas with less tissue)
            image = image_utilities.drop_low_std(image=image, threshold=10)

        elif image_type == 'wsi':

            # Crop the largest contour for WSIs
            largest_contour_bounding_box = image_utilities.get_largest_contour(image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold=20)
            image = image[
                largest_contour_bounding_box[1]:largest_contour_bounding_box[3] + 1,
                largest_contour_bounding_box[0]:largest_contour_bounding_box[2] + 1,
                :
            ]
            image = image_utilities.resize_with_aspect_ratio(image=image, longest_edge=wsi_longest_edge)
            image[image == 0] = 255

        else:
            raise ValueError(f'Invalid image type {image_type}')

        processed_image_shape = image.shape[:2]
        settings.logger.info(f'Image ID {row["image_id"]} - Raw Shape: {raw_image_shape} Processed Shape: {processed_image_shape}')
        metadata.append({
            'image_id': row['image_id'],
            'label': row['label'],
            'image_width': processed_image_shape[1],
            'image_height': processed_image_shape[0],
            'image_type': 'tma',
            'organ': 'ovary',
            'cancer_subtype': np.nan,
            'dataset': 'ubc_ocean',
            'image_path': output_image_directory / f'{row["image_id"]}.png'
        })
        cv2.imwrite(str(output_image_directory / f'{row["image_id"]}.png'), image)

        instances = image_utilities.create_instances(image=image, n_instances=16, instance_size=1024)
        for instance_idx, instance in enumerate(instances, start=1):
            cv2.imwrite(str(output_instance_directory / f'{row["image_id"]}_{instance_idx}.png'), instance)

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(settings.DATA / 'model_datasets' / 'ubc_ocean' / 'metadata.csv', index=False)
